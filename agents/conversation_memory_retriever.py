"""
Conversation Memory Retriever

Implements the retrieval hierarchy:
1. Relational conversational memory (ConversationalKnowledge)
2. Recent assistant answers (semantic relevance)
3. Vector semantic search over messages
4. Cached evidence
5. External retrieval (if memory insufficient)

This PREVENTS redundant web searches for previously answered questions.

Problem solved:
- Follow-up "What is his strike rate?" → already in memory
- No unnecessary external research
- Conversation becomes truly stateful and efficient

Solution:
- Query ConversationalKnowledge first (structured facts)
- Fall back to message similarity search
- Only trigger external research if memory coverage is insufficient
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func
from agents.state import ResearchState
from database.connection import SessionLocal
from database.models import Message, ConversationalKnowledge, ConversationState
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryClaim:
    """Fact retrieved from conversational memory"""
    claim_text: str
    entity_name: str
    source_message_index: int
    source_date: datetime
    confidence: float = 0.9  


@dataclass
class ConversationMemoryRetrievalResult:
    """Result of memory retrieval"""
    query: str
    retrieved_entities: List[str]  
    retrieved_claims: List[MemoryClaim]  
    should_use_memory: bool  
    memory_coverage: float  
    memory_context: str  


class ConversationMemoryRetriever:
    """
    Retrieves facts and entities from conversation memory with proper hierarchy.
    
    Flow:
    1. Search ConversationalKnowledge (extracted facts)
    2. Search recent assistant messages (recency)
    3. Semantic search over message embeddings
    4. Return coverage score (0-1: how well memory covers query)
    
    If coverage >= 0.7, answer from memory only (no external research).
    If coverage < 0.7, blend memory + research.
    
    Usage:
        retriever = ConversationMemoryRetriever(db)
        result = retriever.retrieve(
            conversation_id="conv_123",
            query="What is Klaasen's strike rate?",
            top_k=5
        )
        
        if result.should_use_memory:
            # Use retrieved facts directly
            answer = result.memory_context
        else:
            # Need to do external research
            external_results = do_research(query)
    """
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db or SessionLocal()
    
    def retrieve(
        self,
        conversation_id: UUID,
        query: str,
        top_k: int = 5,
    ) -> ConversationMemoryRetrievalResult:
        """
        Retrieve relevant information from conversation memory using hierarchy.
        
        Implements retrieval priority:
        1. Relational knowledge (ConversationalKnowledge table)
        2. Recent answers (last 10 assistant messages)
        3. Semantic search over embeddings
        
        Args:
            conversation_id: ID of current conversation
            query: Current user query
            top_k: Max entities/facts to retrieve
        
        Returns:
            Retrieval result with entities, claims, and memory coverage
        """
        try:
            # Step 1: Search structured conversational knowledge (HIGHEST PRIORITY)
            knowledge_facts = self._search_conversational_knowledge(
                conversation_id, query, top_k
            )
            
            if knowledge_facts:
                logger.info(f"Found {len(knowledge_facts)} facts in conversational knowledge")
            
            # Step 2: Search recent assistant answers for context
            recent_answers = self._search_recent_answers(conversation_id, top_k=3)
            
            # Step 3: Semantic search (if embeddings available)
            semantic_matches = self._semantic_search(conversation_id, query, top_k=5)
            
            # Convert knowledge facts to MemoryClaims
            claims = [
                MemoryClaim(
                    claim_text=f"{fact['entity']}.{fact['attribute']} = {fact['value']}",
                    entity_name=fact['entity'],
                    source_message_index=0,
                    source_date=fact['updated_at'],
                    confidence=fact['confidence']
                )
                for fact in knowledge_facts
            ]
            
            # Extract entities from retrieved facts
            entities = list(set(f['entity'] for f in knowledge_facts))
            
            # Calculate coverage
            coverage = self._calculate_coverage(query, claims, knowledge_facts, semantic_matches)
            
            # Should we use memory?
            # Yes if: (1) found relevant facts, OR (2) explicit memory query, OR (3) coverage > 0.6
            should_use = len(claims) > 0 or coverage > 0.6
                    
            # Format memory context for injection
            context = self._format_memory_context(entities, claims, recent_answers)
            
            logger.info(
                f"Memory retrieval hierarchy | "
                f"knowledge_facts={len(knowledge_facts)} | "
                f"recent_answers={len(recent_answers)} | "
                f"semantic_matches={len(semantic_matches)} | "
                f"entities={len(entities)} | "
                f"coverage={coverage:.1%} | "
                f"should_use={should_use}"
            )
            
            return ConversationMemoryRetrievalResult(
                query=query,
                retrieved_entities=entities,
                retrieved_claims=claims,
                should_use_memory=should_use,
                memory_coverage=coverage,
                memory_context=context,
            )
            
        except Exception as e:
            logger.error(f"Error in memory retrieval: {e}", exc_info=True)
            return ConversationMemoryRetrievalResult(
                query=query,
                retrieved_entities=[],
                retrieved_claims=[],
                should_use_memory=False,
                memory_coverage=0.0,
                memory_context="",
            )
    
    def _search_conversational_knowledge(
        self,
        conversation_id: UUID,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search the ConversationalKnowledge table (HIGHEST PRIORITY).
        
        Returns extracted facts matching the query intent.
        
        Example:
            Query: "What is Klaasen's strike rate?"
            Result: [
                {entity: "Heinrich Klaasen", attribute: "strike_rate", value: "153.93", ...}
            ]
        """
        try:
            # Extract potential entities from query
            import re
            entities_in_query = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-z]+)*\b', query)
            
            # Query knowledge base
            facts = self.db.query(ConversationalKnowledge).filter(
                ConversationalKnowledge.conversation_id == conversation_id,
                ConversationalKnowledge.is_active == True,
            )
            
            # Filter by entities mentioned in query (if any)
            if entities_in_query:
                facts = facts.filter(ConversationalKnowledge.entity.in_(entities_in_query))
            
            facts = facts.order_by(ConversationalKnowledge.updated_at.desc()).limit(top_k).all()
            
            # Convert to dicts for easier handling
            return [
                {
                    'entity': f.entity,
                    'attribute': f.attribute,
                    'value': f.value,
                    'value_type': f.value_type,
                    'confidence': f.confidence,
                    'updated_at': f.updated_at,
                    'extraction_method': f.extraction_method,
                }
                for f in facts
            ]
            
        except Exception as e:
            logger.error(f"Error searching conversational knowledge: {e}")
            return []
    
    def _search_recent_answers(
        self,
        conversation_id: UUID,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search recent assistant answers for context and facts.
        
        Retrieves the last N assistant messages, which are likely to contain
        recent context and answers.
        """
        try:
            recent = self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
            ).order_by(Message.created_at.desc()).limit(top_k).all()
            
            return [
                {
                    'content': m.content,
                    'created_at': m.created_at,
                    'context_data': m.context_data or {},
                }
                for m in recent
            ]
            
        except Exception as e:
            logger.error(f"Error searching recent answers: {e}")
            return []
    
    def _semantic_search(
        self,
        conversation_id: UUID,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over message embeddings.
        
        Requires embeddings to be computed on messages.
        Uses vector similarity to find relevant prior messages.
        
        TODO: Implement when embeddings are available
        """
        # For now, this is a placeholder
        # Implement when you have embedding infrastructure
        return []
    
    def _calculate_coverage(
        self,
        query: str,
        claims: List[MemoryClaim],
        knowledge_facts: List[Dict],
        semantic_matches: List[Dict],
    ) -> float:
        """
        Calculate how well memory covers the query.
        
        Heuristics:
        - 0.9+: Direct fact match (entity.attribute in knowledge)
        - 0.7-0.9: Entity mentioned with some attributes
        - 0.5-0.7: Related entities but missing specific fact
        - 0.0-0.5: No relevant memory
        """
        if not knowledge_facts:
            return 0.0 if not claims else 0.3
        
        # Extract query intent
        import re
        has_numeric_query = bool(re.search(r'\d+|strike|runs|average|ratio', query, re.IGNORECASE))
        has_numeric_facts = any(f['value_type'] == 'number' for f in knowledge_facts)
        
        # Base score from fact count
        coverage = min(0.9, len(knowledge_facts) * 0.3)
        
        # Boost if we have the right type of fact
        if has_numeric_query and has_numeric_facts:
            coverage += 0.2
        
        # Boost if high confidence facts
        avg_confidence = sum(f['confidence'] for f in knowledge_facts) / len(knowledge_facts)
        coverage += avg_confidence * 0.1
        
        return min(1.0, coverage)
    
    def _format_memory_context(
        self,
        entities: List[str],
        claims: List[MemoryClaim],
        recent_answers: List[Dict],
    ) -> str:
        """
        Format retrieved memory for injection into prompts.
        
        Creates a structured summary that can be added to the system prompt
        or included in the query to the LLM.
        """
        if not claims and not entities:
            return ""
        
        lines = ["## Memory from Conversation\n"]
        
        if entities:
            lines.append(f"**Mentioned entities:** {', '.join(entities)}\n")
        
        if claims:
            lines.append("**Known facts:**")
            for claim in claims[:5]:  # Show top 5 claims
                lines.append(f"- {claim.entity_name}: {claim.claim_text}")
            lines.append("")
        
        if recent_answers:
            lines.append("**Recent context:**")
            for answer in recent_answers[:2]:
                lines.append(f"- {answer['content'][:100]}...")
            lines.append("")
        
        return "\n".join(lines)


def make_conversation_memory_retriever_node():
    """
    Create node that retrieves from conversation memory BEFORE web search.
    
    This is a critical node in the agentic flow:
    
    Flow:
    1. Load conversation state
    2. Search ConversationalKnowledge (extracted facts)
    3. Search recent answers
    4. Determine if memory is sufficient
    5. If coverage >= 0.7, mark for memory-only response
    6. If coverage < 0.7, mark for memory + research blending
    
    Placement in graph: RIGHT AFTER state reconstruction, BEFORE query_planner
    
    This ensures planner knows what was discussed before, avoiding redundant web search.
    """
    def conversation_memory_retriever(state: ResearchState) -> ResearchState:
        if not hasattr(state, 'conversation_id') or not state.conversation_id:
            logger.debug("No conversation_id, skipping memory retrieval")
            return state
        
        retriever = ConversationMemoryRetriever()
        result = retriever.retrieve(
            conversation_id=state.conversation_id,
            query=state.query
        )
        
        # Store retrieval result
        state.memory_retrieval_result = result
        
        # Store for use in subsequent nodes
        state.memory_grounded_entities = result.retrieved_entities
        state.memory_context = result.memory_context
        state.memory_coverage = result.memory_coverage
        
        logger.info(
            f"Memory retrieval complete | "
            f"entities={len(result.retrieved_entities)} | "
            f"coverage={result.memory_coverage:.1%} | "
            f"should_use={result.should_use_memory}"
        )
        
        return state
    
    return conversation_memory_retriever
