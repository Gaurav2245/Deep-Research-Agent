"""
Conversation Orchestration Module

Orchestrates the layered conversational memory flow.

This is the "glue" that ties together:
- Message storage (raw chat history)
- ConversationState (working memory)
- ConversationalKnowledge (relational facts)
- FollowUpResolver (pronoun/reference resolution)
- KnowledgeExtractor (fact extraction)
- ConversationMemoryRetriever (memory-first retrieval)

Usage:
    orchestrator = ConversationOrchestrator(db)
    
    # Before processing user query
    orchestrator.begin_turn(conversation_id, user_query)
    
    # Get memory-first retrieval results
    memory_results = orchestrator.get_memory_results()
    
    # After generating response
    orchestrator.end_turn(assistant_response)
    
This ensures proper flow through all layers.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from database.models import (
    Message, 
    Conversation, 
    ConversationState, 
    ConversationalKnowledge
)
from agents.knowledge_extractor import KnowledgeExtractor
from agents.follow_up_resolver import FollowUpResolver
from agents.conversation_memory_retriever import ConversationMemoryRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TurnContext:
    """Context for a single conversation turn."""
    conversation_id: UUID
    user_message_id: Optional[UUID] = None
    assistant_message_id: Optional[UUID] = None
    user_query: str = ""
    assistant_response: str = ""
    memory_results: Optional[Dict] = None
    resolution_result: Optional[Dict] = None
    extracted_facts: List[Dict] = None
    start_time: datetime = None
    end_time: datetime = None
    
    def elapsed_ms(self) -> Optional[float]:
        """Time taken for this turn (ms)."""
        if self.start_time and self.end_time:
            delta = (self.end_time - self.start_time).total_seconds()
            return delta * 1000
        return None


class ConversationOrchestrator:
    """
    Orchestrates the complete conversational flow with layered memory.
    
    Responsibilities:
    1. Manage conversation state (active topic, entities)
    2. Resolve follow-up references (pronouns, ellipsis)
    3. Retrieve from memory before research
    4. Extract facts from responses
    5. Store messages and facts
    6. Track conversation metrics
    
    Usage:
        orchestrator = ConversationOrchestrator(db)
        ctx = orchestrator.begin_turn(conv_id, "What is Klaasen's strike rate?")
        memory = orchestrator.get_memory_results()
        # ... generate response ...
        orchestrator.end_turn(response)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.extractor = KnowledgeExtractor(db)
        self.resolver = FollowUpResolver(db)
        self.retriever = ConversationMemoryRetriever(db)
        self.current_turn: Optional[TurnContext] = None
    
    def begin_turn(
        self,
        conversation_id: UUID,
        user_query: str,
    ) -> TurnContext:
        """
        Begin a new conversation turn.
        
        Steps:
        1. Store user message
        2. Load/update conversation state
        3. Resolve follow-up references
        4. Retrieve from memory
        
        Args:
            conversation_id: ID of conversation
            user_query: User's input query
        
        Returns:
            TurnContext with all initialization done
        """
        ctx = TurnContext(
            conversation_id=conversation_id,
            user_query=user_query,
            start_time=datetime.utcnow(),
            extracted_facts=[],
        )
        
        try:
            # Step 1: Store user message
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=user_query,
            )
            self.db.add(user_msg)
            self.db.flush()
            ctx.user_message_id = user_msg.id
            
            logger.info(f"Stored user message {user_msg.id} for conv {conversation_id}")
            
            # Step 2: Load/update conversation state
            state = self.db.query(ConversationState).filter_by(
                conversation_id=conversation_id
            ).first()
            
            if not state:
                state = ConversationState(conversation_id=conversation_id)
                self.db.add(state)
                self.db.flush()
                logger.info(f"Created new conversation state for {conversation_id}")
            
            # Step 3: Resolve follow-up references
            prev_query = self._get_previous_query(conversation_id)
            resolution = self.resolver.resolve(
                query=user_query,
                conversation_id=conversation_id,
                previous_query=prev_query,
            )
            ctx.resolution_result = {
                "is_follow_up": resolution.is_follow_up,
                "resolved_references": resolution.resolved_references,
                "primary_entity": resolution.primary_entity,
                "active_entities": resolution.active_entities,
                "resolution_confidence": resolution.resolution_confidence,
            }
            
            # Update state with resolved info
            self.resolver.update_state_with_resolution(
                conversation_id,
                user_query,
                resolution
            )
            
            logger.info(
                f"Resolved follow-up | is_follow_up={resolution.is_follow_up} | "
                f"primary={resolution.primary_entity} | "
                f"confidence={resolution.resolution_confidence:.2f}"
            )
            
            # Step 4: Retrieve from memory
            memory_results = self.retriever.retrieve(
                conversation_id=conversation_id,
                query=user_query,
            )
            ctx.memory_results = {
                "should_use_memory": memory_results.should_use_memory,
                "memory_coverage": memory_results.memory_coverage,
                "retrieved_entities": memory_results.retrieved_entities,
                "memory_context": memory_results.memory_context,
                "num_facts": len(memory_results.retrieved_claims),
            }
            
            logger.info(
                f"Memory retrieval | coverage={memory_results.memory_coverage:.1%} | "
                f"entities={len(memory_results.retrieved_entities)} | "
                f"should_use={memory_results.should_use_memory}"
            )
            
            self.db.commit()
            self.current_turn = ctx
            return ctx
            
        except Exception as e:
            logger.error(f"Error in begin_turn: {e}", exc_info=True)
            self.db.rollback()
            raise
    
    def get_memory_results(self) -> Dict:
        """Get memory retrieval results from current turn."""
        if not self.current_turn:
            raise RuntimeError("No active turn. Call begin_turn first.")
        return self.current_turn.memory_results
    
    def get_resolution_result(self) -> Dict:
        """Get follow-up resolution results from current turn."""
        if not self.current_turn:
            raise RuntimeError("No active turn. Call begin_turn first.")
        return self.current_turn.resolution_result
    
    def should_research(self) -> bool:
        """
        Determine if external research is needed.
        
        Returns:
            True if memory coverage < 0.7 (insufficient)
            False if memory coverage >= 0.7 (sufficient)
        """
        if not self.current_turn or not self.current_turn.memory_results:
            return True  # Default: research
        
        coverage = self.current_turn.memory_results["memory_coverage"]
        return coverage < 0.7
    
    def end_turn(self, assistant_response: str) -> TurnContext:
        """
        End the current conversation turn.
        
        Steps:
        1. Extract knowledge from response
        2. Store assistant message
        3. Update conversation state
        4. Return turn summary
        
        Args:
            assistant_response: The assistant's response text
        
        Returns:
            Updated TurnContext with results
        """
        if not self.current_turn:
            raise RuntimeError("No active turn. Call begin_turn first.")
        
        ctx = self.current_turn
        ctx.assistant_response = assistant_response
        ctx.end_time = datetime.utcnow()
        
        try:
            # Step 1: Extract knowledge from response
            facts = self.extractor.extract_from_response(
                response=assistant_response,
                conversation_id=ctx.conversation_id,
                message_id=ctx.user_message_id,  # Source from user message
            )
            ctx.extracted_facts = [
                {
                    "entity": f.entity,
                    "attribute": f.attribute,
                    "value": f.value,
                    "confidence": f.confidence,
                }
                for f in facts
            ]
            
            logger.info(f"Extracted {len(facts)} facts from response")
            
            # Step 2: Store assistant message
            assistant_msg = Message(
                conversation_id=ctx.conversation_id,
                role="assistant",
                content=assistant_response,
                context_data={
                    "memory_coverage": ctx.memory_results["memory_coverage"],
                    "resolution_confidence": ctx.resolution_result["resolution_confidence"],
                    "facts_extracted": len(facts),
                    "entities_in_scope": ctx.resolution_result["active_entities"],
                    "was_follow_up": ctx.resolution_result["is_follow_up"],
                }
            )
            self.db.add(assistant_msg)
            self.db.flush()
            ctx.assistant_message_id = assistant_msg.id
            
            # Step 3: Update conversation metadata
            conversation = self.db.query(Conversation).filter_by(
                id=ctx.conversation_id
            ).first()
            if conversation:
                conversation.message_count = (conversation.message_count or 0) + 2
                conversation.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            logger.info(
                f"Turn completed | elapsed={ctx.elapsed_ms():.0f}ms | "
                f"facts={len(facts)} | "
                f"response_len={len(assistant_response)}"
            )
            
            self.current_turn = None
            return ctx
            
        except Exception as e:
            logger.error(f"Error in end_turn: {e}", exc_info=True)
            self.db.rollback()
            raise
    
    def _get_previous_query(self, conversation_id: UUID) -> Optional[str]:
        """Get the previous user query for context."""
        try:
            prev_msg = self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.role == "user"
            ).order_by(Message.created_at.desc()).offset(1).first()
            
            return prev_msg.content if prev_msg else None
        except Exception as e:
            logger.error(f"Error getting previous query: {e}")
            return None
    
    def get_conversation_summary(self, conversation_id: UUID) -> Dict:
        """
        Get a summary of the conversation so far.
        
        Returns:
            - total_turns: Number of Q&A pairs
            - active_entities: Currently tracked entities
            - top_topics: Most discussed topics
            - facts_extracted: Total facts learned
            - memory_efficiency: Avg memory coverage (higher = fewer searches)
        """
        try:
            # Count messages
            messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).all()
            turns = len(messages) // 2
            
            # Get active entities
            state = self.db.query(ConversationState).filter_by(
                conversation_id=conversation_id
            ).first()
            entities = state.active_entities if state else []
            
            # Count facts
            facts = self.db.query(ConversationalKnowledge).filter(
                ConversationalKnowledge.conversation_id == conversation_id,
                ConversationalKnowledge.is_active == True,
            ).all()
            
            # Get unique attributes (topics)
            topics = list(set(f.attribute for f in facts))
            
            # Calculate memory efficiency
            coverages = [
                m.context_data.get("memory_coverage", 0)
                for m in messages
                if m.role == "assistant" and m.context_data
            ]
            avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
            
            return {
                "total_turns": turns,
                "active_entities": entities,
                "unique_entities": len(set(f.entity for f in facts)),
                "facts_extracted": len(facts),
                "top_topics": topics[:10],
                "memory_efficiency": avg_coverage,
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {}


def make_orchestrator_node(conversation_id_source: str = "state"):
    """
    Create a graph node that orchestrates the memory flow.
    
    Can be inserted into the research graph as:
        graph.add_node("orchestrate", orchestrator_node)
        
    Args:
        conversation_id_source: Where to get conversation_id from
            ("state" = state.conversation_id, "memory" = local memory, etc.)
    
    Usage:
        from agents.orchestration import make_orchestrator_node
        
        # In your graph building:
        orchestrator = make_orchestrator_node()
        graph.add_node("orchestrate", orchestrator)
        graph.add_edge("resolve_state", "orchestrate")
        graph.add_edge("orchestrate", "query_planner")
    """
    def orchestrator_node(state, config, db):
        """Orchestrate memory retrieval and state management."""
        orchestrator = ConversationOrchestrator(db)
        
        turn_ctx = orchestrator.begin_turn(
            conversation_id=state.conversation_id,
            user_query=state.query,
        )
        
        # Store results in state for downstream nodes
        state.memory_results = turn_ctx.memory_results
        state.resolution_result = turn_ctx.resolution_result
        state.should_research = orchestrator.should_research()
        
        # Add orchestrator to state for end_turn() later
        state._orchestrator = orchestrator
        state._turn_context = turn_ctx
        
        return state
    
    return orchestrator_node
