"""Data validation and quality assurance module."""
from __future__ import annotations

from typing import List, Dict, Tuple
import re

from utils.logger import get_logger
from database.grounding_verifier import GroundingVerifier, AnswerVerification

logger = get_logger(__name__)


class DataValidator:
    """Validate data for completeness, consistency, and hallucinations."""

    def __init__(self, embedding_service=None):
        """Initialize validator with optional embedding service for grounding."""
        self.grounding_verifier = GroundingVerifier(embedding_service=embedding_service)
        self.embedding_service = embedding_service

    def validate_completeness(
        self,
        final_answer: str,
        sources: List[Dict],
        query: str
    ) -> Dict:
        """
        Check if answer is complete (not missing key information).
        
        Returns dict with:
        - passed: bool
        - score: 0-1
        - issues: List[str]
        """
        issues = []
        
        # Check for incomplete indicators
        incomplete_phrases = [
            "unable to find",
            "no information",
            "not available",
            "unknown",
            "unclear",
            "inconclusive",
        ]
        
        answer_lower = final_answer.lower()
        incomplete_count = sum(
            1 for phrase in incomplete_phrases
            if phrase in answer_lower
        )
        
        if incomplete_count > 2:
            issues.append(f"Multiple incompleteness indicators found ({incomplete_count})")
        
        # Check minimum answer length
        answer_words = len(final_answer.split())
        if answer_words < 100:
            issues.append(f"Answer too short ({answer_words} words, expected >100)")
        
        # Check source coverage
        if len(sources) < 3:
            issues.append(f"Insufficient source diversity ({len(sources)} sources, expected >=3)")
        
        # Calculate score
        score = max(0.0, 1.0 - (len(issues) * 0.2))
        
        return {
            "passed": len(issues) == 0,
            "score": score,
            "issues": issues,
            "validation_type": "completeness"
        }

    def validate_consistency(
        self,
        sources: List[Dict],
        final_answer: str
    ) -> Dict:
        """
        Check for consistency contradictions between sources and answer.
        
        Returns dict with:
        - passed: bool
        - score: 0-1
        - conflicts: List[str]
        """
        conflicts = []
        
        # Extract key claims from answer
        # Simple pattern: look for "X is Y" statements
        claim_patterns = [
            r"([A-Z][a-z\s]+) is ([^.]+)\.",
            r"([A-Z][a-z\s]+) was ([^.]+)\.",
            r"([A-Z][a-z\s]+) are ([^.]+)\.",
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, final_answer)
            claims.extend(matches)
        
        if not claims:
            return {
                "passed": True,
                "score": 0.8,
                "conflicts": [],
                "validation_type": "consistency"
            }
        
        # Check if claims contradict content in sources
        source_contents = " ".join([s.get("content", "") for s in sources])
        
        for subject, claim in claims:
            # Look for contradictory statements
            contradiction_patterns = [
                f"{subject} is not {claim}",
                f"{subject} was not {claim}",
                f"{subject} are not {claim}",
            ]
            
            for contra_pattern in contradiction_patterns:
                if contra_pattern.lower() in source_contents.lower():
                    conflicts.append(f"Potential contradiction: {subject} {claim}")
        
        # Check for common contradictory topics
        if "increasing" in final_answer.lower() and "decreasing" in final_answer.lower():
            conflicts.append("Answer contains both 'increasing' and 'decreasing' - potential contradiction")
        
        score = max(0.0, 1.0 - (len(conflicts) * 0.2))
        
        return {
            "passed": len(conflicts) == 0,
            "score": score,
            "conflicts": conflicts,
            "validation_type": "consistency"
        }

    def validate_factual_claims(
        self,
        final_answer: str,
        sources: List[Dict]
    ) -> Dict:
        """
        Validate that claims in answer are backed by sources.
        
        Returns dict with:
        - passed: bool
        - score: 0-1
        - unsupported_claims: List[str]
        """
        unsupported_claims = []
        
        # Extract numbers/dates that look like factual claims
        numbers = re.findall(r"\b(\d+(?:\.\d+)?(?:\s+(?:million|billion|trillion|thousand|%|degrees?))?)\b", final_answer)
        dates = re.findall(r"\b(\d{1,2}/\d{1,2}/\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b", final_answer)
        
        source_text = " ".join([s.get("content", "") for s in sources])
        
        # Check if numbers/dates appear in sources
        for number in numbers[:5]:  # Check first 5 numbers
            number_str = str(number).split()[0]  # Get base number
            if number_str not in source_text:
                unsupported_claims.append(f"Number '{number}' not found in any source")
        
        for date in dates[:5]:  # Check first 5 dates
            if date not in source_text:
                unsupported_claims.append(f"Date '{date}' not found in any source")
        
        # Check for proper citations
        import urllib.parse
        citation_pattern = r"\(https?://[^\)]+\)"
        citations = re.findall(citation_pattern, final_answer)
        
        # Expect at least 1 citation per 5 sentences
        sentences = len(final_answer.split("."))
        expected_citations = max(1, sentences // 5)
        
        if len(citations) < expected_citations:
            unsupported_claims.append(
                f"Insufficient citations ({len(citations)}, expected >={expected_citations})"
            )
        
        score = max(0.0, 1.0 - (len(unsupported_claims) * 0.15))
        
        return {
            "passed": len(unsupported_claims) == 0,
            "score": score,
            "unsupported_claims": unsupported_claims,
            "validation_type": "factual_claims"
        }

    def validate_grounding(
        self,
        final_answer: str,
        sources: List[Dict],
        query: str,
        threshold: float = 0.80
    ) -> Dict:
        """
        Verify that answer claims are grounded in sources.
        
        This is the STRONGEST hallucination check.
        
        Returns dict with:
        - passed: bool (grounding_score >= threshold)
        - grounding_score: float (0-1)
        - hallucination_risk: str (low/medium/high)
        - ungrounded_claims: List[str]
        - recommendation: str
        - verification_results: Full verification details
        """
        try:
            verification = self.grounding_verifier.verify_answer(
                answer=final_answer,
                sources=sources,
                query=query
            )
            
            # Convert to dict format
            result = {
                "passed": verification.grounding_score >= threshold,
                "grounding_score": verification.grounding_score,
                "hallucination_risk": verification.hallucination_risk,
                "ungrounded_claims": verification.ungrounded_claims,
                "recommendation": verification.recommendation,
                "total_claims_checked": verification.total_claims,
                "grounded_claims": verification.grounded_claims,
                "ungrounded_count": verification.hallucinated_claims,
                "validation_type": "grounding",
            }
            
            logger.info(f"Grounding validation: {verification.grounding_score:.1%} grounded")
            
            return result
            
        except Exception as e:
            logger.error(f"Grounding verification failed: {e}")
            # Return conservative result on error
            return {
                "passed": False,
                "grounding_score": 0.5,
                "hallucination_risk": "high",
                "ungrounded_claims": ["Verification error"],
                "recommendation": "Manual review required",
                "total_claims_checked": 0,
                "grounded_claims": 0,
                "ungrounded_count": 0,
                "validation_type": "grounding",
                "error": str(e)
            }

    def detect_hallucination_markers(
        self,
        final_answer: str,
        sources: List[Dict]
    ) -> Dict:
        """
        Detect markers that suggest hallucination in the answer.
        
        Looks for:
        - Non-existent sources or quotes
        - Overly specific details not in sources
        - Contradictions with facts
        """
        hallucination_markers = []
        
        # Check for quotes that might not be sourced
        quote_pattern = r'"([^"]{20,})"'
        quotes = re.findall(quote_pattern, final_answer)
        
        source_text = " ".join([s.get("content", "") for s in sources])
        
        for quote in quotes[:5]:  # Check first 5 quotes
            if quote not in source_text:
                hallucination_markers.append(f"Quote '{quote[:50]}...' not found in sources")
        
        # Check for overly specific claims without sources
        specific_patterns = [
            r"According to ([A-Z][a-z\s]+),",  # Attribution
            r"([A-Z][a-z\s]+) stated that",
            r"([A-Z][a-z\s]+) confirmed",
        ]
        
        for pattern in specific_patterns:
            matches = re.findall(pattern, final_answer)
            for match in matches[:3]:
                if match not in source_text:
                    hallucination_markers.append(f"Source '{match}' not mentioned in any document")
        
        # Check for impossible combinations
        if "future predictions" in final_answer.lower() and "2050" in final_answer:
            hallucination_markers.append("Contains specific predictions about future with uncertain sourcing")
        
        score = max(0.0, 1.0 - (len(hallucination_markers) * 0.25))
        flagged = len(hallucination_markers) > 0
        
        return {
            "passed": not flagged,
            "score": score,
            "markers": hallucination_markers,
            "hallucination_flagged": flagged,
            "validation_type": "hallucination"
        }

    def validate_all(
        self,
        final_answer: str,
        sources: List[Dict],
        query: str
    ) -> Dict:
        """
        Run all validation checks INCLUDING grounding verification.
        
        Returns:
        - results: List of validation results
        - all_passed: bool
        - overall_quality_score: 0-1
        - critical_issues: List (grounding failures are critical)
        """
        results = [
            self.validate_completeness(final_answer, sources, query),
            self.validate_consistency(sources, final_answer),
            self.validate_factual_claims(final_answer, sources),
            self.detect_hallucination_markers(final_answer, sources),
            self.validate_grounding(final_answer, sources, query),  # MOST IMPORTANT
        ]
        
        all_passed = all(r.get("passed", False) for r in results)
        
        # Calculate overall quality score (grounding is weighted heavily)
        scores = []
        for r in results:
            score = r.get("score") or r.get("grounding_score", 0.0)
            if r.get("validation_type") == "grounding":
                # Weight grounding 2x more important
                scores.append(score * 2)
            else:
                scores.append(score)
        
        overall_quality = sum(scores) / sum([1 if r.get("validation_type") != "grounding" else 2 for r in results]) if scores else 0.0
        overall_quality = min(1.0, overall_quality)
        
        # Identify critical issues
        critical_issues = []
        grounding_result = results[-1]  # Last one is grounding
        
        if not grounding_result.get("passed"):
            critical_issues.append(f"🔴 HALLUCINATION RISK: {grounding_result.get('hallucination_risk')}")
            critical_issues.append(f"   {grounding_result.get('recommendation')}")
        
        return {
            "results": results,
            "all_passed": all_passed,
            "overall_quality_score": overall_quality,
            "issues_found": len([r for r in results if not r.get("passed", False)]),
            "critical_issues": critical_issues,
            "grounding_validation": grounding_result,
            "safe_to_use": all_passed and grounding_result.get("passed", False)
        }
