"""Confidence scoring engine to determine research completion."""
from __future__ import annotations

from typing import List, Dict, Tuple
import statistics

from database.embedding_service import EmbeddingService
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceScorer:

    @staticmethod
    def score_source_diversity(sources: List[Dict]) -> float:
        
        if not sources:
            return 0.0
        
        domains = set()
        for source in sources:
            url = source.get("url", "")
            if url:
                from database.source_scorer import SourceScorer
                domain = SourceScorer.extract_domain(url)
                domains.add(domain)
        
        diversity_ratio = len(domains) / len(sources)
        # Ideal: 1 unique domain per source (diversity_ratio = 1.0)
        # If all from same domain: diversity_ratio = 1/n
        
        return min(diversity_ratio, 1.0)

    @staticmethod
    def score_source_quality(sources: List[Dict]) -> float:
        """
        Score average quality of sources.
        
        Uses domain authority and overall scores.
        Returns: 0.0-1.0
        """
        if not sources:
            return 0.0
        
        scores = [s.get("overall_score", 0.0) for s in sources if s.get("overall_score")]
        if not scores:
            return 0.0
        
        return statistics.mean(scores)

    @staticmethod
    def score_data_consistency(embeddings: List[List[float]], sources: List[Dict] = None) -> float:
        """
        Score consistency of information using embeddings with corroboration boost.
        
        Similar embeddings = consistent information.
        If similar embeddings come from DIFFERENT domains = Corroboration Boost.
        Returns: 0.0-1.0 (higher = more consistent/corroborated)
        """
        if len(embeddings) < 2:
            return 0.5
        
        sources = sources or []
        embedding_service = EmbeddingService()
        similarities = []
        corroboration_count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = embedding_service.cosine_similarity(
                    embeddings[i],
                    embeddings[j]
                )
                similarities.append(sim)
                
                # Check for cross-domain corroboration
                if sim > 0.85 and i < len(sources) and j < len(sources):
                    from urllib.parse import urlparse
                    domain_i = urlparse(sources[i].get("url", "")).netloc
                    domain_j = urlparse(sources[j].get("url", "")).netloc
                    
                    if domain_i and domain_j and domain_i != domain_j:
                        corroboration_count += 1
        
        if not similarities:
            return 0.5
            
        avg_similarity = statistics.mean(similarities)
        
        # Corroboration Boost: Up to 0.2 bonus for high-agreement across different domains
        boost = min(corroboration_count * 0.05, 0.2)
        final_score = avg_similarity + boost
        
        if avg_similarity < 0.3:
            return 0.2
            
        return min(final_score, 1.0)

    @staticmethod
    def score_answer_completeness(
        final_answer: str,
        query: str,
        expected_aspects: List[str] = None
    ) -> float:
        """
        Score completeness of answer.
        
        Checks:
        - Answer length (must be substantive)
        - Coverage of query aspects
        - Specific details/numbers
        
        Returns: 0.0-1.0
        """
        if not final_answer:
            return 0.0
        
        # Length check (substantive answer)
        answer_words = len(final_answer.split())
        if answer_words < 50:
            return 0.3
        elif answer_words < 200:
            return 0.6
        elif answer_words < 500:
            return 0.8
        else:
            length_score = 1.0
        
        # Coverage check
        coverage_score = 0.5
        if expected_aspects:
            answer_lower = final_answer.lower()
            covered = sum(
                1 for aspect in expected_aspects
                if aspect.lower() in answer_lower
            )
            coverage_score = covered / len(expected_aspects) if expected_aspects else 0.5
        
        # Specificity check (numbers, dates, entities)
        import re
        numbers = re.findall(r"\d+", final_answer)
        dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", final_answer)
        specific_keywords = ["according to", "data shows", "research indicates", "statistics"]
        specificity_count = len(numbers) + len(dates) + sum(
            1 for kw in specific_keywords if kw in final_answer.lower()
        )
        specificity_score = min(specificity_count / 5, 1.0)  # Expecting 5+ specifics
        
        # Weighted combination
        completeness = (
            length_score * 0.4 +
            coverage_score * 0.4 +
            specificity_score * 0.2
        )
        
        return min(completeness, 1.0)

    @staticmethod
    def score_no_hallucination(
        final_answer: str,
        sources: List[Dict],
        validation_issues: List[Dict] = None
    ) -> float:
        """
        Score confidence that answer has no hallucinations.
        
        Checks:
        - All claims are cited/sourced
        - No contradictions flagged
        - Factual consistency checks passed
        
        Returns: 0.0-1.0 (higher = more trustworthy)
        """
        if not final_answer or not sources:
            return 0.3
        
        validation_issues = validation_issues or []
        
        # Check for hallucination flags
        hallucination_count = sum(
            1 for issue in validation_issues
            if issue.get("validation_type") == "hallucination"
        )
        
        contradiction_count = sum(
            1 for issue in validation_issues
            if issue.get("validation_type") == "contradiction"
        )
        
        # Base score
        base_score = 1.0
        
        # Deduct for each issue
        base_score -= hallucination_count * 0.2
        base_score -= contradiction_count * 0.15
        
        # Check source citation patterns
        import re
        citation_pattern = r"\(https?://[^\)]+\)"
        citations = re.findall(citation_pattern, final_answer)
        citation_score = min(len(citations) / 10, 0.5)  # Max 0.5 bonus for citations
        
        base_score += citation_score
        
        return max(min(base_score, 1.0), 0.0)

    @classmethod
    def calculate_information_gain(
        cls,
        current_embeddings: List[List[float]],
        previous_embeddings: List[List[float]]
    ) -> float:
        """
        Calculate how much 'new' semantic information was gained in this round.
        Returns: 0.0-1.0 (1.0 = completely new, 0.0 = redundant)
        """
        if not current_embeddings or not previous_embeddings:
            return 1.0
            
        embedding_service = EmbeddingService()
        similarities = []
        
        # Check each new embedding against all previous ones
        for curr in current_embeddings:
            max_sim = 0.0
            for prev in previous_embeddings:
                sim = embedding_service.cosine_similarity(curr, prev)
                max_sim = max(max_sim, sim)
            similarities.append(max_sim)
            
        # Information gain = 1 - average maximum similarity
        # If all new sources are 90% similar to old ones, gain is 0.1
        avg_max_sim = statistics.mean(similarities)
        gain = 1.0 - avg_max_sim
        
        return max(min(gain, 1.0), 0.0)

    @classmethod
    def calculate_context_confidence(
        cls,
        sources: List[Dict],
        embeddings: List[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate confidence based ONLY on gathered context (pre-synthesis).
        Focuses on reliability: Source Agreement (50%) and Authority (40%).
        """
        embeddings = embeddings or []
        
        diversity = cls.score_source_diversity(sources)
        quality = cls.score_source_quality(sources)
        consistency = cls.score_data_consistency(embeddings) if embeddings else 0.5
        
        # Reliability-Centric Weights:
        # Agreement (Consistency) is king. Authority is queen.
        overall = (
            consistency * 0.50 +
            quality * 0.40 +
            diversity * 0.10
        )
        
        return {
            "source_diversity": diversity,
            "source_quality": quality,
            "data_consistency": consistency,
            "overall_confidence": overall,
        }

    @classmethod
    def calculate_overall_confidence(
        cls,
        sources: List[Dict],
        final_answer: str,
        query: str,
        embeddings: List[List[float]] = None,
        validation_issues: List[Dict] = None,
        expected_aspects: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate overall research confidence.
        
        Returns dict with:
        - source_diversity: 0-1
        - source_quality: 0-1
        - data_consistency: 0-1
        - answer_completeness: 0-1
        - no_hallucination: 0-1
        - overall_confidence: weighted average (0-1)
        """
        embeddings = embeddings or []
        validation_issues = validation_issues or []
        
        diversity = cls.score_source_diversity(sources)
        quality = cls.score_source_quality(sources)
        consistency = cls.score_data_consistency(embeddings, sources) if embeddings else 0.5
        completeness = cls.score_answer_completeness(final_answer, query, expected_aspects)
        hallucination = cls.score_no_hallucination(final_answer, sources, validation_issues)
        
        # Weighted calculation
        overall = (
            diversity * 0.15 +
            quality * 0.25 +
            consistency * 0.20 +
            completeness * 0.25 +
            hallucination * 0.15
        )
        
        return {
            "source_diversity": diversity,
            "source_quality": quality,
            "data_consistency": consistency,
            "answer_completeness": completeness,
            "no_hallucination": hallucination,
            "overall_confidence": overall,
        }

    @staticmethod
    def should_continue_research(
        overall_confidence: float,
        min_confidence: float = 0.7,
        iterations: int = 0,
        max_iterations: int = 5,
        has_follow_ups: bool = False,
        has_contradictions: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if research should continue based on confidence and pending tasks.
        """
        if iterations >= max_iterations:
            return False, f"Reached maximum iterations ({max_iterations})"
        
        # If we have contradictions or identified gaps (follow-ups), we SHOULD continue
        # unless confidence is already exceptionally high.
        if has_contradictions and overall_confidence < 0.9:
            if not has_follow_ups:
                return False, (
                    "Contradictions flagged but no pending search queries; "
                    "stopping to avoid a non-progress loop."
                )
            return True, f"Contradictions detected | Confidence: {overall_confidence:.2f}"
            
        if has_follow_ups and overall_confidence < min_confidence:
            return True, f"Follow-up queries pending | Confidence: {overall_confidence:.2f}"

        if overall_confidence >= min_confidence:
            return False, f"Sufficient confidence reached ({overall_confidence:.2f})"

        # No pending searches: continuing would loop web_search with 0 queries (non-progress).
        if not has_follow_ups:
            return False, (
                f"Confidence below threshold ({overall_confidence:.2f}) but no pending "
                "search queries; stopping to avoid an infinite low-confidence loop."
            )

        return True, f"Confidence too low ({overall_confidence:.2f}), continuing research"
