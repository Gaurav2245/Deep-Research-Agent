

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
from agents.state import ResearchState
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GroundedClaim:
    """A single claim with grounding assessment"""
    claim_text: str
    claim_index: int
    is_grounded: bool
    grounding_evidence: Optional[str] = None  # Source snippet
    grounding_score: float = 0.0  # 0-1
    entities_mentioned: List[str] = field(default_factory=list)
    requires_evidence: bool = True  # Some claims don't need evidence (definitions, methods)


@dataclass
class ClaimLevelGroundingResult:
    """Result of claim-level grounding validation"""
    answer_text: str
    claims: List[GroundedClaim]
    grounded_count: int
    ungrounded_count: int
    partial_count: int
    overall_grounding_score: float  # Weighted average
    
    # Actionable info
    grounded_claims: List[str]
    ungrounded_claims: List[str]
    needs_repair: List[Tuple[int, str]]  # (index, claim, reason)
    repairs: Dict = field(default_factory=dict)  # {"index": "corrected_claim"}
    
    def can_use_answer(self, threshold: float = 0.5) -> bool:
        """Can we use this answer if >= threshold% grounded?"""
        if not self.claims:
            return False
        return self.overall_grounding_score >= threshold


class ClaimLevelGroundingValidator:
    """
    Validates grounding at claim granularity.
    
    Usage:
        validator = ClaimLevelGroundingValidator()
        result = validator.validate_answer(
            answer=synthesized_answer,
            sources=retrieved_sources,
            prior_entities=prior_entities
        )
        
        # See which claims need repair
        for idx, claim, reason in result.needs_repair:
            print(f"Claim {idx}: {claim} - {reason}")
    """
    
    def __init__(self):
        self.min_grounding_score = 0.6
    
    def validate_answer(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        prior_entities: Optional[List[str]] = None,
        conversation_context: Optional[str] = None
    ) -> ClaimLevelGroundingResult:
        """
        Validate grounding at claim level.
        
        Args:
            answer: Synthesized answer text
            sources: Retrieved sources with text
            prior_entities: Entities from prior conversation
            conversation_context: Full conversation history
        
        Returns:
            ClaimLevelGroundingResult with per-claim assessment
        """
        if not answer:
            return self._empty_result(answer)
        
        # Break answer into claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return self._empty_result(answer)
        
        # Assess each claim
        grounded_claims = []
        ungrounded_claims = []
        needs_repair = []
        partial_count = 0
        
        for i, claim in enumerate(claims):
            assessed = self._assess_claim(
                claim=claim,
                claim_index=i,
                sources=sources or [],
                prior_entities=prior_entities or [],
                conversation_context=conversation_context or ""
            )
            
            grounded_claims.append(assessed)
            
            if assessed.is_grounded:
                pass  # Counted below
            elif 0.3 < assessed.grounding_score < 0.6:
                partial_count += 1
                needs_repair.append((i, claim, "Weak grounding"))
            else:
                ungrounded_claims.append(assessed)
                needs_repair.append((i, claim, "No evidence found"))
        
        # Calculate overall score
        grounded_count = sum(1 for c in grounded_claims if c.is_grounded)
        ungrounded_count = len(ungrounded_claims)
        overall_score = self._calculate_overall_score(
            grounded_count,
            ungrounded_count,
            partial_count,
            len(grounded_claims)
        )
        
        logger.info(
            f"Claim-level grounding | total={len(grounded_claims)} | "
            f"grounded={grounded_count} | ungrounded={ungrounded_count} | "
            f"partial={partial_count} | overall={overall_score:.2f}"
        )
        
        return ClaimLevelGroundingResult(
            answer_text=answer,
            claims=grounded_claims,
            grounded_count=grounded_count,
            ungrounded_count=ungrounded_count,
            partial_count=partial_count,
            overall_grounding_score=overall_score,
            grounded_claims=[c.claim_text for c in grounded_claims if c.is_grounded],
            ungrounded_claims=[c.claim_text for c in ungrounded_claims],
            needs_repair=needs_repair
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Break answer into claims (sentences)"""
        # Split by periods, then clean
        claims = re.split(r'(?<=[.!?])\s+', answer.strip())
        
        # Filter empty claims
        claims = [c.strip() for c in claims if c.strip()]
        
        return claims
    
    def _assess_claim(
        self,
        claim: str,
        claim_index: int,
        sources: List[Dict],
        prior_entities: List[str],
        conversation_context: str
    ) -> GroundedClaim:
        """Assess if a single claim is grounded"""
        
        # Extract entities from claim
        entities = self._extract_entities(claim)
        
        # Check if claim is grounded in sources
        grounding_score = 0.0
        grounding_evidence = None
        
        for source in sources:
            text = source.get('text', '') if isinstance(source, dict) else ''
            
            # Simple: check if claim text or key entities appear in source
            if claim.lower() in text.lower():
                grounding_score = 0.95
                grounding_evidence = text[:200]
                break
            
            # Partial: entities appear in source
            entities_in_source = sum(
                1 for entity in entities if entity.lower() in text.lower()
            )
            
            if entities_in_source > 0:
                entity_coverage = entities_in_source / len(entities) if entities else 0
                grounding_score = max(grounding_score, 0.7 * entity_coverage)
                if not grounding_evidence:
                    grounding_evidence = text[:200]
        
        # Check if entities exist in prior conversation
        if grounding_score < 0.5:
            for entity in entities:
                if entity in prior_entities:
                    grounding_score = max(grounding_score, 0.6)
                    break
        
        # Check if claim is verifiable knowledge (doesn't require sources)
        if self._is_knowledge_claim(claim):
            grounding_score = max(grounding_score, 0.7)
        
        # Verdict
        is_grounded = grounding_score >= self.min_grounding_score
        
        return GroundedClaim(
            claim_text=claim,
            claim_index=claim_index,
            is_grounded=is_grounded,
            grounding_evidence=grounding_evidence,
            grounding_score=grounding_score,
            entities_mentioned=entities,
            requires_evidence=not self._is_knowledge_claim(claim)
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract capitalized entities from text"""
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        return list(set(matches))
    
    def _is_knowledge_claim(self, claim: str) -> bool:
        """
        Check if claim is verifiable knowledge vs. analysis.
        
        Verifiable knowledge: "Strike rate is calculated as runs/balls"
        Analysis: "This shows strong form"
        """
        knowledge_patterns = [
            'is calculated',
            'is defined',
            'means',
            'refers to',
            'is a',
            'are',
        ]
        
        claim_lower = claim.lower()
        return any(p in claim_lower for p in knowledge_patterns)
    
    def _calculate_overall_score(
        self,
        grounded: int,
        ungrounded: int,
        partial: int,
        total: int
    ) -> float:
        """Calculate weighted overall grounding score"""
        if total == 0:
            return 0.0
        
        # Weights: grounded=1.0, partial=0.5, ungrounded=0.0
        score = (grounded * 1.0 + partial * 0.5 + ungrounded * 0.0) / total
        
        return score
    
    def _empty_result(self, answer: str) -> ClaimLevelGroundingResult:
        """Return empty result structure"""
        return ClaimLevelGroundingResult(
            answer_text=answer,
            claims=[],
            grounded_count=0,
            ungrounded_count=0,
            partial_count=0,
            overall_grounding_score=0.0,
            grounded_claims=[],
            ungrounded_claims=[],
            needs_repair=[]
        )
    
    def repair_answer(
        self,
        result: ClaimLevelGroundingResult,
        repair_source: str = "clarification"
    ) -> str:
        """
        Repair ungrounded claims.
        
        Args:
            result: ClaimLevelGroundingResult from validation
            repair_source: Where to get repairs ("clarification", "retry", "remove")
        
        Returns:
            Repaired answer
        """
        if repair_source == "remove":
            # Remove ungrounded claims
            repaired = " ".join(result.grounded_claims)
            return repaired or "Unable to provide grounded answer."
        
        elif repair_source == "clarification":
            # Add caveat about ungrounded parts
            if result.overall_grounding_score >= 0.5:
                caveat = (
                    f"\n\nNote: This answer is based on {result.grounded_count}/{len(result.claims)} "
                    f"supported claims. {result.ungrounded_count} claims could not be verified."
                )
                return result.answer_text + caveat
        
        return result.answer_text


def make_claim_level_grounding_node():
    """
    Create node that validates grounding at claim granularity.
    
    Placement in graph: After synthesis, BEFORE data_validator
    
    Instead of simple pass/fail, provides detailed grounding breakdown.
    """
    def claim_level_grounding(state: ResearchState) -> ResearchState:
        if not hasattr(state, 'final_answer') or not state.final_answer:
            return state
        
        validator = ClaimLevelGroundingValidator()
        
        # Gather context
        sources = getattr(state, 'scored_sources', [])
        prior_entities = [
            e.name if hasattr(e, 'name') else e
            for e in getattr(state, 'prior_entities', [])
        ]
        
        # Validate
        result = validator.validate_answer(
            answer=state.final_answer,
            sources=sources,
            prior_entities=prior_entities
        )
        
        # Store result
        state.claim_grounding_result = result
        
        # Update confidence based on grounding
        if hasattr(state, 'confidence_score'):
            # Factor in grounding to confidence
            grounding_factor = result.overall_grounding_score
            state.confidence_score = state.confidence_score * (0.5 + 0.5 * grounding_factor)
        
        logger.info(
            f"Claim grounding: {result.grounded_count} grounded, "
            f"{result.ungrounded_count} ungrounded, "
            f"overall={result.overall_grounding_score:.2f}"
        )
        
        return state
    
    return claim_level_grounding
