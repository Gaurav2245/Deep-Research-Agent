"""
Follow-Up Resolution Logic

Resolves pronouns and contextual references in follow-up queries.

Example:
    User: "Tell me about Heinrich Klaasen"
    Assistant: "Klaasen has a strike rate of 153.93..."
    
    User: "What is his average?"  # "his" → Heinrich Klaasen
    
This module:
1. Loads conversation state (active entities, recent mentions)
2. Resolves "his", "her", "the player", "this team" → specific entity
3. Updates ResearchState with resolved_entities
4. Enables memory-first retrieval (no search for "what is his average?")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import UUID
import re

from sqlalchemy.orm import Session
from agents.state import ResearchState
from database.models import ConversationState, ConversationalKnowledge
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResolutionResult:
    """Result of follow-up reference resolution."""
    is_follow_up: bool  # Is this a follow-up question?
    resolved_references: Dict[str, str]  # {pronoun/reference: entity}
    active_entities: List[str]  # All entities in scope
    primary_entity: Optional[str]  # The main subject (if any)
    scope_context: Optional[str]  # Domain context (e.g., "IPL 2026")
    resolution_confidence: float  # 0.0-1.0: how confident are we?


class FollowUpResolver:
    """
    Resolves pronouns and contextual references in conversation.
    
    Uses conversation state to determine what entities are "active" and recent,
    then maps pronouns to those entities.
    
    Critical for:
    - Follow-up questions: "What is his strike rate?"
    - Implicit references: "Tell me more" → repeat about current entity
    - Plural references: "What about them?"
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def resolve(
        self,
        query: str,
        conversation_id: UUID,
        previous_query: Optional[str] = None,
    ) -> ResolutionResult:
        """
        Resolve pronouns and references in the query.
        
        Args:
            query: Current user query
            conversation_id: ID of conversation
            previous_query: The previous user query (for ellipsis resolution)
        
        Returns:
            Resolution result with mapped pronouns to entities
        """
        try:
            # Load conversation state
            state = self.db.query(ConversationState).filter(
                ConversationState.conversation_id == conversation_id
            ).first()
            
            if not state:
                # No state yet - this is not a follow-up
                return ResolutionResult(
                    is_follow_up=False,
                    resolved_references={},
                    active_entities=[],
                    primary_entity=None,
                    scope_context=None,
                    resolution_confidence=0.0,
                )
            
            # Check if this is a follow-up
            is_follow_up = self._is_follow_up_query(query, previous_query, state)
            
            # Resolve references
            resolved_refs = self._resolve_pronouns(query, state)
            
            # Determine primary entity
            primary_entity = self._determine_primary_entity(query, state, resolved_refs)
            
            # Get all active entities
            active_entities = state.active_entities or []
            
            # Calculate confidence
            confidence = self._calculate_confidence(resolved_refs, query, state)
            
            logger.info(
                f"Follow-up resolution | "
                f"is_follow_up={is_follow_up} | "
                f"primary={primary_entity} | "
                f"resolved={resolved_refs} | "
                f"confidence={confidence:.2f}"
            )
            
            return ResolutionResult(
                is_follow_up=is_follow_up,
                resolved_references=resolved_refs,
                active_entities=active_entities,
                primary_entity=primary_entity,
                scope_context=state.scope_context,
                resolution_confidence=confidence,
            )
            
        except Exception as e:
            logger.error(f"Error resolving follow-up: {e}", exc_info=True)
            return ResolutionResult(
                is_follow_up=False,
                resolved_references={},
                active_entities=[],
                primary_entity=None,
                scope_context=None,
                resolution_confidence=0.0,
            )
    
    def _is_follow_up_query(
        self,
        query: str,
        previous_query: Optional[str],
        state: ConversationState,
    ) -> bool:
        """
        Determine if this is a follow-up to previous context.
        
        Signals:
        - Contains pronouns: "his", "her", "their"
        - Contains "tell me more", "what about", "also"
        - References "previous", "mentioned", "above"
        - Short query (ellipsis): "And Y?" → asking about context
        """
        query_lower = query.lower()
        
        # Explicit follow-up signals
        follow_up_markers = [
            'what about', 'tell me more', 'also', 'additionally',
            'furthermore', 'moreover', 'what else', 'anything else',
            'and him', 'and her', 'and them', 'and this',
            'mentioned', 'previously', 'above', 'earlier',
            'the other', 'another', 'the previous',
        ]
        
        # Pronouns (strong signal of follow-up)
        pronouns = ['his', 'her', 'their', 'its', 'his ', 'her ', 'their ', 'its ']
        
        has_marker = any(marker in query_lower for marker in follow_up_markers)
        has_pronoun = any(pronoun in query_lower for pronoun in pronouns)
        
        # Ellipsis: very short query with context available
        is_ellipsis = len(query.split()) <= 3 and len(state.active_entities or []) > 0
        
        return has_marker or has_pronoun or is_ellipsis
    
    def _resolve_pronouns(
        self,
        query: str,
        state: ConversationState,
    ) -> Dict[str, str]:
        """
        Map pronouns and references to actual entities.
        
        Priority:
        1. Stored pending_references (explicitly resolved before)
        2. Most recently mentioned entity (by recency)
        3. Most frequently mentioned entity (by count)
        """
        resolved = {}
        query_lower = query.lower()
        
        # List of pronouns and references to resolve
        pronouns_to_resolve = {
            'his': 'entity (male player)',
            'her': 'entity (female player)',
            'their': 'entity (plural)',
            'its': 'entity',
            'he': 'entity (male)',
            'she': 'entity (female)',
            'they': 'entity (plural)',
            'this player': 'entity',
            'that player': 'entity',
            'the player': 'entity',
            'the batter': 'entity',
            'the batsman': 'entity',
        }
        
        # First, check for explicit references in state
        if state.pending_references:
            for pronoun, entity in state.pending_references.items():
                if pronoun in query_lower:
                    resolved[pronoun] = entity
        
        # For remaining pronouns, use recency/frequency
        active_entities = state.active_entities or []
        recent_mentions = state.recent_entity_mentions or {}
        
        if not active_entities:
            return resolved
        
        # Find the most salient entity (most recent + highest mention count)
        if len(active_entities) == 1:
            # Only one entity? Easy choice
            primary = active_entities[0]
        else:
            # Multi-entity: prefer most frequently mentioned
            sorted_by_freq = sorted(
                active_entities,
                key=lambda e: recent_mentions.get(e, 0),
                reverse=True
            )
            primary = sorted_by_freq[0]
        
        # Resolve all pronouns to primary entity
        for pronoun in pronouns_to_resolve.keys():
            if pronoun in query_lower and pronoun not in resolved:
                resolved[pronoun] = primary
        
        return resolved
    
    def _determine_primary_entity(
        self,
        query: str,
        state: ConversationState,
        resolved_refs: Dict[str, str],
    ) -> Optional[str]:
        """
        Determine the primary entity being asked about.
        
        Heuristics:
        1. Explicit entity name in query
        2. Resolved pronoun
        3. Most frequently mentioned entity in conversation
        """
        # Extract proper nouns from query
        proper_noun_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        matches = re.findall(proper_noun_pattern, query)
        
        # Remove common words that aren't entities
        stopwords = {'The', 'This', 'That', 'And', 'But', 'For', 'IPL', 'Cricket', 'CSK', 'MI', 'RCB'}
        explicit_entities = [m for m in matches if m not in stopwords]
        
        if explicit_entities:
            return explicit_entities[0]
        
        # Try to get from resolved pronouns
        if resolved_refs:
            return list(resolved_refs.values())[0]
        
        # Fall back to most mentioned entity
        active = state.active_entities or []
        if active:
            recent_mentions = state.recent_entity_mentions or {}
            return max(active, key=lambda e: recent_mentions.get(e, 0))
        
        return None
    
    def _calculate_confidence(
        self,
        resolved_refs: Dict[str, str],
        query: str,
        state: ConversationState,
    ) -> float:
        """
        Calculate confidence in the resolution.
        
        - High (0.9+): Explicit entity or single entity in scope
        - Medium (0.6-0.9): Resolved pronoun with multiple entities
        - Low (0.0-0.6): Ambiguous reference, multiple possibilities
        """
        if not resolved_refs and len(state.active_entities or []) > 1:
            return 0.4  # Ambiguous
        
        if resolved_refs:
            return 0.85  # We resolved something
        
        if len(state.active_entities or []) == 1:
            return 0.9  # Only one entity, unambiguous
        
        # Check for explicit entity in query
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', query)
        if proper_nouns:
            return 0.95
        
        return 0.5
    
    def update_state_with_resolution(
        self,
        conversation_id: UUID,
        query: str,
        result: ResolutionResult,
    ) -> None:
        """
        Update conversation state based on resolution.
        
        - Track primary entity as most recent focus
        - Store pending references for next query
        - Update active topic
        """
        state = self.db.query(ConversationState).filter(
            ConversationState.conversation_id == conversation_id
        ).first()
        
        if not state:
            return
        
        # Update primary focus
        if result.primary_entity:
            state.current_focus = result.primary_entity
        
        # Store resolved references for next query
        if result.resolved_references:
            state.pending_references = result.resolved_references
        
        # Update active topic from query
        if not state.active_topic or result.is_follow_up:
            # Infer topic from query
            topic_keywords = self._extract_topic(query)
            if topic_keywords:
                state.active_topic = topic_keywords
        
        state.turn_count = (state.turn_count or 0) + 1
        
        from datetime import datetime
        state.last_updated_at = datetime.utcnow()
        
        self.db.commit()
    
    def _extract_topic(self, query: str) -> str:
        """Extract the main topic/intent from query."""
        # Simple heuristic: common sports keywords
        topics = {
            'strike': 'batting statistics',
            'average': 'player statistics',
            'runs': 'performance metrics',
            'wicket': 'bowling statistics',
            'score': 'match results',
            'ranking': 'rankings',
            'compare': 'player comparison',
        }
        
        query_lower = query.lower()
        for keyword, topic in topics.items():
            if keyword in query_lower:
                return topic
        
        return "general inquiry"
