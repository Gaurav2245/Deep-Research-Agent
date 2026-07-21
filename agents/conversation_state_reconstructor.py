"""
Conversation State Reconstructor

Loads and reconstructs full conversational context from memory BEFORE planning.
This is the bridge between memory storage and active cognition.

Problem solved:
- Agent had memory storage but couldn't query it
- Follow-ups treated as standalone queries
- Previous entities/facts invisible to planner
- No conversation grounding in planning

Solution:
- Load prior messages from DB
- Extract entities/topics from prior answers
- Build conversational context
- Inject into state before query planning
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import json
from datetime import datetime
from agents.state import ResearchState
from database.connection import get_db
from database.models import Conversation, Message
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PriorEntity:
    """Entity extracted from prior conversation"""
    name: str
    entity_type: str  
    first_mentioned_message: int
    last_mentioned_message: int
    mention_count: int
    context_snippet: str  
    confidence: float = 0.9


@dataclass
class ConversationStateContext:
    """Full reconstructed conversation state"""
    conversation_id: str
    turn_number: int
    
    # Prior exchange
    prior_message_count: int
    prior_messages_summary: str  # Summary of all prior messages
    
    # Entities from prior conversation
    prior_entities: List[PriorEntity] = field(default_factory=list)
    
    # Topics/themes
    conversation_topics: List[str] = field(default_factory=list)  # ["IPL 2026", "player stats"]
    focus_domain: str = ""  # "cricket_ipl"
    
    # Last answer context
    last_answer_text: str = ""
    last_answer_entities: List[str] = field(default_factory=list)
    
    # Follow-up signals
    is_follow_up: bool = False
    follow_up_type: str = ""  # "clarification", "expansion", "comparison"
    references_resolved: Dict[str, str] = field(default_factory=dict)  # "him" -> "Klaasen"
    
    # Query hints from context
    likely_entity_scope: List[str] = field(default_factory=list)  # ["Klaasen", "SRH", "IPL 2026"]
    
    def to_prompt_context(self) -> str:
        """Format as context for LLM injection"""
        lines = [
            "=== CONVERSATION CONTEXT ===",
            f"Turn: {self.turn_number}",
            f"Prior messages: {self.prior_message_count}",
            f"Topics: {', '.join(self.conversation_topics)}",
            "",
            "=== PRIOR ENTITIES ===",
        ]
        
        for entity in self.prior_entities:
            lines.append(f"- {entity.name} ({entity.entity_type}): mentioned {entity.mention_count}x")
            lines.append(f"  Context: {entity.context_snippet}")
        
        if self.last_answer_text:
            lines.extend([
                "",
                "=== LAST ANSWER ===",
                self.last_answer_text[:500],  # First 500 chars
            ])
        
        if self.references_resolved:
            lines.extend([
                "",
                "=== RESOLVED REFERENCES ===",
            ])
            for ref, resolved in self.references_resolved.items():
                lines.append(f"- '{ref}' → {resolved}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Safe serialization to dict"""
        return {
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "prior_message_count": self.prior_message_count,
            "prior_messages_summary": self.prior_messages_summary,
            "prior_entities": [asdict(e) for e in self.prior_entities],
            "conversation_topics": self.conversation_topics,
            "focus_domain": self.focus_domain,
            "last_answer_text": self.last_answer_text,
            "last_answer_entities": self.last_answer_entities,
            "is_follow_up": self.is_follow_up,
            "follow_up_type": self.follow_up_type,
            "references_resolved": self.references_resolved,
            "likely_entity_scope": self.likely_entity_scope,
        }


class ConversationStateReconstructor:
    """
    Loads conversational state from memory database.
    
    Usage:
        reconstructor = ConversationStateReconstructor()
        context = reconstructor.reconstruct(
            conversation_id="conv_123",
            current_query="tell strike rate of all batsmen mentioned above"
        )
        
        # Use context
        state.conversation_context = context
        state.conversation_context_str = context.to_prompt_context()
    """
    
    def __init__(self):
        self.db = get_db()
    
    def reconstruct(
        self,
        conversation_id: str,
        current_query: str,
        look_back_messages: int = 10
    ) -> ConversationStateContext:
        """
        Reconstruct full conversation state from memory.
        
        Args:
            conversation_id: ID of conversation
            current_query: Current user query
            look_back_messages: How many prior messages to consider
        
        Returns:
            ConversationStateContext with all prior knowledge
        """
        try:
            # Load conversation
            conversation = self.db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found")
                return ConversationStateContext(
                    conversation_id=conversation_id,
                    turn_number=0,
                    prior_message_count=0,
                    prior_messages_summary=""
                )
            
            # Load prior messages
            prior_messages = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).all()
            
            prior_message_count = len(prior_messages)
            
            # Build context
            context = ConversationStateContext(
                conversation_id=conversation_id,
                turn_number=prior_message_count + 1,
                prior_message_count=prior_message_count,
                prior_messages_summary=self._summarize_messages(prior_messages)
            )
            
            # Extract entities from prior messages
            context.prior_entities = self._extract_entities_from_messages(prior_messages)
            
            # Infer topics
            context.conversation_topics = self._infer_topics(prior_messages)
            context.focus_domain = self._infer_domain(prior_messages)
            
            # Get last answer
            if prior_messages:
                last_assistant_message = None
                for msg in reversed(prior_messages):
                    if msg.role == "assistant":
                        last_assistant_message = msg
                        break
                
                if last_assistant_message:
                    context.last_answer_text = last_assistant_message.content[:1000]
                    context.last_answer_entities = self._extract_entities_from_text(
                        last_assistant_message.content
                    )
            
            # Detect follow-up
            context.is_follow_up, context.follow_up_type = self._detect_follow_up(
                current_query,
                prior_messages
            )
            
            # Resolve references
            context.references_resolved = self._resolve_references(
                current_query,
                context.prior_entities
            )
            
            # Determine entity scope for query
            context.likely_entity_scope = [
                e.name for e in context.prior_entities[:5]
            ]
            
            logger.info(
                f"Reconstructed state | turn={context.turn_number} | "
                f"prior_entities={len(context.prior_entities)} | "
                f"is_follow_up={context.is_follow_up}"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error reconstructing state: {e}", exc_info=True)
            return ConversationStateContext(
                conversation_id=conversation_id,
                turn_number=0,
                prior_message_count=0,
                prior_messages_summary=""
            )
    
    def _summarize_messages(self, messages: List[Message]) -> str:
        """Summarize prior messages into context string"""
        if not messages:
            return ""
        
        lines = []
        for msg in messages[-10:]:  # Last 10 messages
            prefix = "User:" if msg.role == "user" else "Assistant:"
            lines.append(f"{prefix} {msg.content[:200]}")
        
        return "\n".join(lines)
    
    def _extract_entities_from_messages(self, messages: List[Message]) -> List[PriorEntity]:
        """Extract named entities mentioned in prior messages"""
        entities = {}
        
        for i, msg in enumerate(messages):
            text_entities = self._extract_entities_from_text(msg.content)
            for entity_name in text_entities:
                if entity_name not in entities:
                    entities[entity_name] = {
                        "name": entity_name,
                        "entity_type": self._infer_entity_type(entity_name),
                        "first_mentioned_message": i,
                        "last_mentioned_message": i,
                        "mention_count": 1,
                        "context_snippet": msg.content[:200],
                    }
                else:
                    entities[entity_name]["mention_count"] += 1
                    entities[entity_name]["last_mentioned_message"] = i
        
        # Convert to PriorEntity objects, sorted by recency
        result = [
            PriorEntity(**e) for e in entities.values()
        ]
        result.sort(key=lambda x: x.last_mentioned_message, reverse=True)
        
        return result
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction - can be enhanced with NER"""
        # Look for capitalized proper nouns
        import re
        
        # Simple pattern: Capitalized words (likely proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter out common non-entities
        stopwords = {'The', 'This', 'That', 'And', 'But', 'For', 'IPL'}
        return [m for m in set(matches) if m not in stopwords and len(m) > 2]
    
    def _infer_entity_type(self, entity_name: str) -> str:
        """Infer entity type from context"""
        # Simple heuristics - can be enhanced
        if any(x in entity_name.lower() for x in ['team', 'unit', 'kings', 'royals', 'warriors']):
            return "team"
        elif any(x in entity_name.lower() for x in ['ipl', '2024', '2025', '2026']):
            return "tournament"
        else:
            return "player"  # Default for IPL context
    
    def _infer_topics(self, messages: List[Message]) -> List[str]:
        """Infer conversation topics from messages"""
        topics = set()
        
        for msg in messages:
            if "strike rate" in msg.content.lower():
                topics.add("strike_rate")
            if "runs" in msg.content.lower():
                topics.add("scoring")
            if "ipl" in msg.content.lower():
                topics.add("IPL_2026")
        
        return list(topics)
    
    def _infer_domain(self, messages: List[Message]) -> str:
        """Infer conversation domain"""
        full_text = " ".join(m.content.lower() for m in messages)
        
        if "ipl" in full_text:
            return "cricket_ipl"
        if "cricket" in full_text:
            return "cricket"
        
        return "general"
    
    def _detect_follow_up(self, query: str, prior_messages: List[Message]) -> tuple[bool, str]:
        """Detect if query is follow-up"""
        if not prior_messages:
            return False, ""
        
        query_lower = query.lower()
        
        # Signals
        is_follow_up = False
        follow_up_type = ""
        
        # Pronoun reference
        if any(p in query_lower for p in ['his', 'her', 'their', 'that', 'above']):
            is_follow_up = True
            follow_up_type = "reference"
        
        # Clarification signals
        if any(c in query_lower for c in ['what about', 'tell me about', 'more about', 'also']):
            is_follow_up = True
            follow_up_type = "clarification"
        
        # Comparison signals
        if any(c in query_lower for c in ['compare', 'vs', 'difference', 'similar']):
            is_follow_up = True
            follow_up_type = "comparison"
        
        return is_follow_up, follow_up_type
    
    def _resolve_references(
        self,
        query: str,
        prior_entities: List[PriorEntity]
    ) -> Dict[str, str]:
        """Resolve pronouns/references to entities"""
        resolved = {}
        
        query_lower = query.lower()
        
        # Resolve pronouns to most recent entity
        if prior_entities:
            primary_entity = prior_entities[0].name  # Most recent
            
            pronouns = {
                'he': primary_entity,
                'his': primary_entity,
                'she': primary_entity,
                'her': primary_entity,
                'their': primary_entity,
            }
            
            for pronoun, entity in pronouns.items():
                if pronoun in query_lower:
                    resolved[pronoun] = entity
        
        return resolved


def make_conversation_state_reconstructor_node():
    """
    Create node that reconstructs conversation state before planning.
    
    Placement in graph: RIGHT AFTER entity_extractor, BEFORE query_planner
    
    This ensures that when query planner runs, it has full conversational context.
    """
    def conversation_state_reconstructor(state: ResearchState) -> ResearchState:
        if not hasattr(state, 'conversation_id') or not state.conversation_id:
            logger.debug("No conversation_id in state, skipping state reconstruction")
            return state
        
        reconstructor = ConversationStateReconstructor()
        context = reconstructor.reconstruct(
            conversation_id=state.conversation_id,
            current_query=state.query
        )
        
        # Store in state
        state.conversation_state_context = context
        state.conversation_context_str = context.to_prompt_context()
        
        # Update state fields
        state.is_follow_up = context.is_follow_up
        state.prior_entities = context.prior_entities
        state.conversation_topics = context.conversation_topics
        
        logger.info(
            f"State reconstructed | topics={len(context.conversation_topics)} | "
            f"entities={len(context.prior_entities)} | is_follow_up={context.is_follow_up}"
        )
        
        return state
    
    return conversation_state_reconstructor
