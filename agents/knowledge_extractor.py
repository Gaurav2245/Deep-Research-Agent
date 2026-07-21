"""
Knowledge Extraction Pipeline

Extracts structured facts from assistant responses and stores them as conversational knowledge.

This is the MOST IMPORTANT missing piece. After EVERY assistant answer, we:
1. Parse the response for entity-attribute-value triples
2. Extract confidence scores
3. Store in conversational_knowledge table

This powers follow-up questions without re-searching.

Example:
    Input: "Heinrich Klaasen has a strike rate of 153.93 and scored 508 runs"
    
    Extracted knowledge:
    [
        {entity: "Heinrich Klaasen", attribute: "strike_rate", value: "153.93", confidence: 0.95},
        {entity: "Heinrich Klaasen", attribute: "runs", value: "508", confidence: 0.95},
    ]
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedFact:
    """A single extracted entity-attribute-value triple."""
    entity: str
    attribute: str
    value: str
    value_type: str = "string"
    confidence: float = 0.9
    source_text_snippet: str = ""
    extraction_method: str = "llm"


class KnowledgeExtractor:
    """
    Extracts structured facts from assistant responses.
    
    Critical workflow:
    1. Assistant generates response
    2. Extract facts: entity → attribute → value
    3. Store in conversational_knowledge
    4. These facts are used for future follow-up questions
    
    Usage:
        extractor = KnowledgeExtractor(db)
        facts = extractor.extract_from_response(
            response="Klaasen scored 508 runs with strike rate 153.93",
            conversation_id="...",
            message_id="...",
        )
        # Automatically stores in DB
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def extract_from_response(
        self,
        response: str,
        conversation_id: UUID,
        message_id: UUID,
        use_llm: bool = False,
    ) -> List[ExtractedFact]:
        """
        Extract structured facts from an assistant response.
        
        Args:
            response: The assistant's response text
            conversation_id: ID of the conversation
            message_id: ID of the message containing this response
            use_llm: If True, use LLM for extraction; if False, use regex patterns
        
        Returns:
            List of extracted facts (already stored in DB)
        """
        try:
            if use_llm:
                facts = self._extract_with_llm(response)
            else:
                facts = self._extract_with_patterns(response)
            
            # Store facts in conversational_knowledge
            self._persist_facts(facts, conversation_id, message_id)
            
            logger.info(
                f"Extracted {len(facts)} facts from response | "
                f"conv_id={conversation_id} | "
                f"facts={[(f.entity, f.attribute) for f in facts[:3]]}"
            )
            
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {e}", exc_info=True)
            return []
    
    def _extract_with_patterns(self, text: str) -> List[ExtractedFact]:
        """
        Use regex patterns to extract common entity-attribute-value patterns.
        
        Handles common sports stats format:
        - "Player has X runs"
        - "Player scored Y with Z strike rate"
        - "X is the highest/lowest"
        - "Player average is Z"
        """
        facts = []
        
        # Pattern 1: "Entity has/scored/averaged X attribute"
        # e.g., "Klaasen has a strike rate of 153.93"
        pattern1 = r'([A-Z][a-zA-Z\s]+?)\s+(?:has|scored|averaged|with)\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+of\s+([0-9.]+)'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches:
            entity = match.group(1).strip()
            attribute = match.group(2).strip()
            value = match.group(3).strip()
            facts.append(ExtractedFact(
                entity=entity,
                attribute=attribute,
                value=value,
                value_type="number" if self._is_number(value) else "string",
                confidence=0.85,
                source_text_snippet=match.group(0),
                extraction_method="regex"
            ))
        
        # Pattern 2: "Player runs: X, Strike Rate: Y"
        # e.g., "Klaasen - 508 runs, 153.93 strike rate"
        pattern2 = r'([A-Z][a-zA-Z\s]+?)\s*-?\s*([0-9]+)\s+(\w+)'
        matches = re.finditer(pattern2, text)
        for match in matches:
            entity = match.group(1).strip()
            value = match.group(2).strip()
            attribute = match.group(3).strip()
            
            # Avoid duplicates
            if not any(f.entity == entity and f.attribute == attribute for f in facts):
                facts.append(ExtractedFact(
                    entity=entity,
                    attribute=attribute,
                    value=value,
                    value_type="number",
                    confidence=0.80,
                    source_text_snippet=match.group(0),
                    extraction_method="regex"
                ))
        
        # Pattern 3: "Top/Highest/Most X: Entity"
        # e.g., "Highest strike rate: Klaasen (153.93)"
        pattern3 = r'(?:Top|Highest|Most|Lowest|Least)\s+(\w+(?:\s+\w+)?):?\s+([A-Z][a-zA-Z\s]+?)\s*\(?([0-9.]+)?\)?'
        matches = re.finditer(pattern3, text, re.IGNORECASE)
        for match in matches:
            attribute = match.group(1).strip()
            entity = match.group(2).strip()
            value = match.group(3) if match.group(3) else "—"
            
            facts.append(ExtractedFact(
                entity=entity,
                attribute=attribute,
                value=value,
                value_type="number" if value != "—" else "categorical",
                confidence=0.85,
                source_text_snippet=match.group(0),
                extraction_method="regex"
            ))
        
        # Remove duplicates by (entity, attribute)
        seen = set()
        unique_facts = []
        for fact in facts:
            key = (fact.entity, fact.attribute)
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        
        return unique_facts
    
    def _extract_with_llm(self, text: str) -> List[ExtractedFact]:
        """
        Use LLM to extract entity-attribute-value triples.
        
        More accurate but slower. Use for complex responses.
        """
        try:
            from llm.factory import create_llm
            from config import get_agent_config
            from agents.prompts import relational_knowledge_prompt
            from agents.conversational_knowledge import format_knowledge_for_prompt
            
            cfg = get_agent_config()
            llm = create_llm(cfg)
            
            # Get existing knowledge for context
            # (Note: we might not have the conversation_id here if called internally, 
            # but extract_from_response provides it)
            
            # For extraction, we don't necessarily need prior knowledge, 
            # but it helps with entity name consistency.
            
            response = llm.invoke(
                relational_knowledge_prompt.format(
                    prior_knowledge="{}", # Simplified for direct extraction
                    assistant_answer=text[:20000]
                )
            )
            
            content = response.content.strip()
            # Remove markdown fences
            for fence in ("```json", "```"):
                content = content.removeprefix(fence).removesuffix("```").strip()
            
            data = json.loads(content)
            entity_facts = data.get("entity_facts", {})
            
            facts = []
            for entity, attrs in entity_facts.items():
                for attr, value in attrs.items():
                    facts.append(ExtractedFact(
                        entity=entity,
                        attribute=attr,
                        value=str(value),
                        value_type="number" if self._is_number(str(value)) else "string",
                        confidence=0.95,
                        source_text_snippet=f"{entity}: {attr}={value}",
                        extraction_method="llm"
                    ))
            
            return facts
            
        except Exception as e:
            logger.error(f"LLM extraction failed, falling back to patterns: {e}")
            return self._extract_with_patterns(text)
    
    def _is_number(self, value: str) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _persist_facts(
        self,
        facts: List[ExtractedFact],
        conversation_id: UUID,
        message_id: UUID
    ) -> None:
        """
        Store extracted facts in conversational_knowledge table.
        
        Also handles:
        - Checking for superseded facts (newer value for same entity+attribute)
        - Updating conversation state with new entities
        """
        from database.models import ConversationalKnowledge, ConversationState
        
        for fact in facts:
            # Check if this entity-attribute already exists
            existing = self.db.query(ConversationalKnowledge).filter(
                ConversationalKnowledge.conversation_id == conversation_id,
                ConversationalKnowledge.entity == fact.entity,
                ConversationalKnowledge.attribute == fact.attribute,
                ConversationalKnowledge.is_active == True,
            ).first()
            
            if existing:
                # Mark old fact as superseded
                existing.is_active = False
                existing.superseded_by = None  # Will be set after new fact is created
            
            # Create new fact
            knowledge = ConversationalKnowledge(
                conversation_id=conversation_id,
                entity=fact.entity,
                attribute=fact.attribute,
                value=fact.value,
                value_type=fact.value_type,
                confidence=fact.confidence,
                extraction_method=fact.extraction_method,
                source_message_id=message_id,
                source_text_snippet=fact.source_text_snippet,
                is_active=True,
            )
            self.db.add(knowledge)
            
            # Mark old as superseded
            if existing:
                existing.superseded_by = knowledge.id
        
        # Update conversation state with new entities
        state = self.db.query(ConversationState).filter(
            ConversationState.conversation_id == conversation_id
        ).first()
        
        if state:
            # Add new entities to active_entities
            new_entities = list(set(f.entity for f in facts))
            for entity in new_entities:
                if entity not in state.active_entities:
                    state.active_entities.append(entity)
                
                # Track mention count
                if "recent_entity_mentions" not in state.recent_entity_mentions:
                    state.recent_entity_mentions = {}
                state.recent_entity_mentions[entity] = \
                    state.recent_entity_mentions.get(entity, 0) + 1
            
            state.last_updated_at = datetime.utcnow()
        
        self.db.commit()
    
    def get_entity_attributes(
        self,
        conversation_id: UUID,
        entity: str
    ) -> Dict[str, str]:
        """
        Get all known attributes for an entity in this conversation.
        
        Useful for follow-ups like "what else do we know about Player X?"
        
        Returns:
            {attribute: value, ...}
        """
        from database.models import ConversationalKnowledge
        
        facts = self.db.query(ConversationalKnowledge).filter(
            ConversationalKnowledge.conversation_id == conversation_id,
            ConversationalKnowledge.entity == entity,
            ConversationalKnowledge.is_active == True,
        ).all()
        
        return {f.attribute: f.value for f in facts}
    
    def get_active_entities(self, conversation_id: UUID) -> List[str]:
        """Get all entities mentioned in this conversation."""
        from database.models import ConversationalKnowledge
        
        facts = self.db.query(ConversationalKnowledge).filter(
            ConversationalKnowledge.conversation_id == conversation_id,
            ConversationalKnowledge.is_active == True,
        ).all()
        
        return list(set(f.entity for f in facts))
    
    def get_fact(
        self,
        conversation_id: UUID,
        entity: str,
        attribute: str
    ) -> Optional[ExtractedFact]:
        """Retrieve a specific entity-attribute fact from memory."""
        from database.models import ConversationalKnowledge
        
        knowledge = self.db.query(ConversationalKnowledge).filter(
            ConversationalKnowledge.conversation_id == conversation_id,
            ConversationalKnowledge.entity == entity,
            ConversationalKnowledge.attribute == attribute,
            ConversationalKnowledge.is_active == True,
        ).first()
        
        if knowledge:
            return ExtractedFact(
                entity=knowledge.entity,
                attribute=knowledge.attribute,
                value=knowledge.value,
                value_type=knowledge.value_type,
                confidence=knowledge.confidence,
                source_text_snippet=knowledge.source_text_snippet,
                extraction_method=knowledge.extraction_method,
            )
        return None
