"""
Evidence Gating Layer

Prevents synthesis when evidence is insufficient.
Blocks "generate first, validate later" antipattern.

Problem solved:
- Synthesizer generated answers without checking if sources exist
- Grounding verifier detected hallucinations AFTER synthesis
- System committed to hallucination before catching it

Solution:
- Check evidence sufficiency BEFORE synthesis
- Block or constrain generation based on evidence
- Trigger retrieval retry if evidence low
- Request clarification if ambiguous
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from agents.state import ResearchState
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvidenceProfile:
    """Evidence sufficiency assessment"""
    total_sources: int
    relevant_sources: int
    relevance_score: float  # 0-1
    coverage_score: float  # How many query aspects covered
    evidence_gaps: List[str]  # What's missing
    should_generate: bool  # Should synthesis proceed?
    generation_constraints: str  # Constraints for LLM if generating


class EvidenceGatingValidator:
    """
    Evaluates if evidence is sufficient before synthesis.
    
    Prevents the "generate without evidence" problem.
    
    Usage:
        gating = EvidenceGatingValidator()
        profile = gating.evaluate_evidence(state)
        
        if not profile.should_generate:
            # Retry retrieval, request clarification, or abort
            return retry_retrieval(state)
    """
    
    # Thresholds
    MIN_SOURCES = 2
    MIN_RELEVANCE = 0.6
    MIN_COVERAGE = 0.5
    
    def evaluate_evidence(self, state: ResearchState) -> EvidenceProfile:
        """
        Evaluate if evidence is sufficient for synthesis.
        
        Returns:
            EvidenceProfile with verdict and constraints
        """
        total_sources = len(state.scored_sources) if hasattr(state, 'scored_sources') else 0
        
        # Count relevant sources
        relevant_sources = 0
        relevance_score = 0.0
        
        if hasattr(state, 'scored_sources') and state.scored_sources:
            for source in state.scored_sources:
                score = source.get('score', 0) if isinstance(source, dict) else 0.0
                if score > 0.5:  # Threshold for "relevant"
                    relevant_sources += 1
                relevance_score += score
            
            relevance_score = relevance_score / len(state.scored_sources) if state.scored_sources else 0.0
        
        # Evaluate coverage
        coverage_score = self._evaluate_coverage(state)
        
        # Identify gaps
        evidence_gaps = self._identify_gaps(state)
        
        # Verdict
        should_generate = self._should_generate_verdict(
            total_sources,
            relevant_sources,
            relevance_score,
            coverage_score
        )
        
        # Constraints for generation
        constraints = self._generate_constraints(
            should_generate,
            evidence_gaps,
            relevant_sources
        )
        
        logger.info(
            f"Evidence gating | sources={total_sources} | "
            f"relevant={relevant_sources} | relevance={relevance_score:.2f} | "
            f"coverage={coverage_score:.2f} | should_generate={should_generate}"
        )
        
        return EvidenceProfile(
            total_sources=total_sources,
            relevant_sources=relevant_sources,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            evidence_gaps=evidence_gaps,
            should_generate=should_generate,
            generation_constraints=constraints
        )
    
    def _evaluate_coverage(self, state: ResearchState) -> float:
        """
        Evaluate how well evidence covers query requirements.
        
        Returns:
            Score 0-1 of coverage
        """
        if not hasattr(state, 'scored_sources') or not state.scored_sources:
            return 0.0
        
        # Simple: average score of sources
        scores = [
            s.get('score', 0) if isinstance(s, dict) else 0.0
            for s in state.scored_sources
        ]
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        
        # In production: match entity coverage, aspect coverage
        return min(avg_score, 1.0)
    
    def _identify_gaps(self, state: ResearchState) -> List[str]:
        """Identify what evidence is missing"""
        gaps = []
        
        # No sources
        if not hasattr(state, 'scored_sources') or not state.scored_sources:
            gaps.append("No sources retrieved")
            return gaps
        
        # Low relevance sources
        relevant_count = sum(
            1 for s in state.scored_sources
            if (s.get('score', 0) if isinstance(s, dict) else 0) > 0.5
        )
        
        if relevant_count == 0:
            gaps.append("No relevant sources found")
        
        # Missing entity coverage
        if hasattr(state, 'prior_entities') and state.prior_entities:
            if not hasattr(state, 'memory_grounded_entities'):
                gaps.append("Missing grounding for mentioned entities")
        
        # Unresolved references
        if hasattr(state, 'unresolved_references') and state.unresolved_references:
            gaps.append(f"Unresolved references: {state.unresolved_references}")
        
        return gaps
    
    def _should_generate_verdict(
        self,
        total_sources: int,
        relevant_sources: int,
        relevance_score: float,
        coverage_score: float
    ) -> bool:
        """
        Determine if synthesis should proceed.
        
        Conservative: require reasonable evidence before generation.
        """
        # Hard requirement: at least some sources
        if total_sources < self.MIN_SOURCES:
            return False
        
        # Hard requirement: relevance
        if relevance_score < self.MIN_RELEVANCE:
            return False
        
        # Soft requirement: coverage
        if coverage_score < 0.3:
            logger.warning("Low coverage, but proceeding with generation")
        
        return True
    
    def _generate_constraints(
        self,
        should_generate: bool,
        evidence_gaps: List[str],
        relevant_sources: int
    ) -> str:
        """
        Generate constraints to inject into synthesis prompt.
        
        Controls LLM behavior based on evidence quality.
        """
        lines = []
        
        if not should_generate:
            lines.append("CRITICAL: Evidence is insufficient. REQUEST CLARIFICATION.")
            lines.extend([f"Gap: {gap}" for gap in evidence_gaps])
            return "\n".join(lines)
        
        if relevant_sources < 3:
            lines.append("WARNING: Limited sources available.")
            lines.append("Be conservative in claims.")
            lines.append("Only state claims directly supported by provided sources.")
        
        if evidence_gaps:
            lines.append(f"Gaps identified: {', '.join(evidence_gaps)}")
            lines.append("Address these gaps or note them explicitly in answer.")
        
        if lines:
            return "\n".join(lines)
        
        return ""


def make_evidence_gating_node():
    """
    Create node that validates evidence before synthesis.
    
    Placement in graph: RIGHT AFTER confidence_scorer, BEFORE conditional router
    
    If evidence insufficient:
    - Block synthesis
    - Trigger retrieval retry
    - Request clarification
    - Prevent hallucination
    """
    def evidence_gating(state: ResearchState) -> ResearchState:
        validator = EvidenceGatingValidator()
        profile = validator.evaluate_evidence(state)
        
        # Store profile
        state.evidence_profile = profile
        
        # If evidence insufficient, don't proceed to synthesis
        if not profile.should_generate:
            logger.warning(
                f"Evidence insufficient: gaps={profile.evidence_gaps} | "
                f"sources={profile.relevant_sources}"
            )
            
            # Option 1: Retry web_search
            if profile.total_sources == 0 or profile.coverage_score < 0.2:
                logger.info("Triggering retrieval retry due to insufficient evidence")
                # This would need router support to loop back to web_search
                state.should_retry_search = True
                state.retry_reason = "Insufficient evidence for synthesis"
            
            # Option 2: Request clarification (if ambiguous)
            if "Unresolved references" in str(profile.evidence_gaps):
                state.needs_clarification = True
                state.clarification_request = "Please clarify which entity you're asking about"
            
            # Block synthesis by setting flag
            state.synthesis_blocked = True
        else:
            # Inject constraints into state for synthesizer to use
            state.synthesis_constraints = profile.generation_constraints
        
        return state
    
    return evidence_gating


def make_evidence_gating_repair_node():
    """
    Create node that handles synthesis blocking and repair.
    
    If synthesis was blocked by gating, this node decides what to do.
    """
    def evidence_gating_repair(state: ResearchState) -> ResearchState:
        if not hasattr(state, 'synthesis_blocked') or not state.synthesis_blocked:
            return state  # Pass through if synthesis not blocked
        
        logger.info("Entering repair mode due to blocked synthesis")
        
        # Option 1: Retry retrieval
        if hasattr(state, 'should_retry_search') and state.should_retry_search:
            logger.info("Retrying web search with modified query")
            # Query planner should attempt broader search
            state.search_retry_count = getattr(state, 'search_retry_count', 0) + 1
            
            if state.search_retry_count <= 2:
                # Broaden the search
                if state.search_queries:
                    original_query = state.search_queries[0]
                    state.search_queries = [
                        original_query,
                        original_query.replace("specific", "general"),
                        f"overview of {original_query}",
                    ]
                return state
            else:
                # Give up on external search
                logger.warning("Search retries exhausted")
        
        # Option 2: Return clarification instead of synthesis
        if hasattr(state, 'needs_clarification') and state.needs_clarification:
            state.final_answer = (
                f"I need clarification: {state.clarification_request}\n\n"
                f"Please specify which entities you're asking about, "
                f"and I'll retrieve the information."
            )
            state.confidence_score = 0.1
            logger.info("Returning clarification request instead of synthesis")
        
        # Option 3: Synthesize with available evidence (conservative)
        else:
            state.final_answer = (
                "I found limited information on this topic. "
                "Please provide more specific details or ask about a different aspect."
            )
            state.confidence_score = 0.2
            logger.info("Returning conservative fallback answer")
        
        return state
    
    return evidence_gating_repair
