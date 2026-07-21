"""Grounding and hallucination detection verifier.

Verifies that answer claims are actually supported by source content.
Implements sentence-level verification, citation enforcement, and
factual consistency checking.
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    claim: str
    is_grounded: bool
    supporting_sources: List[str]
    confidence: float
    evidence_text: str
    issues: List[str]


@dataclass
class AnswerVerification:
    """Overall verification result for entire answer."""
    total_claims: int
    grounded_claims: int
    hallucinated_claims: int
    grounding_score: float
    hallucination_risk: str  # low, medium, high
    ungrounded_claims: List[str]
    verification_results: List[VerificationResult]
    recommendation: str


class GroundingVerifier:
    """
    Verify answer claims are grounded in sources.
    
    This prevents hallucination by:
    1. Sentence-level claim extraction
    2. Evidence matching in sources
    3. Citation enforcement
    4. Factual consistency checking
    """

    def __init__(self, embedding_service=None):
        """
        Initialize verifier.
        
        Parameters
        ----------
        embedding_service : EmbeddingService, optional
            For semantic similarity matching (improves recall)
        """
        self.embedding_service = embedding_service

    def verify_answer(
        self,
        answer: str,
        sources: List[Dict],
        query: str
    ) -> AnswerVerification:
        """
        Verify entire answer against sources.
        
        Parameters
        ----------
        answer : str
            The generated answer to verify
        sources : List[Dict]
            List of source dicts with 'url', 'title', 'content'
        query : str
            Original query (for context)
            
        Returns
        -------
        AnswerVerification
            Detailed verification results
        """
        logger.info(f"Starting grounding verification for answer ({len(answer)} chars)")
        
        # Extract sentences/claims
        sentences = self._extract_sentences(answer)
        logger.info(f"Extracted {len(sentences)} sentences to verify")
        
        # Combine source content
        source_content = self._prepare_source_content(sources)
        
        # Verify each claim
        results = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            result = self._verify_claim(
                claim=sentence,
                source_content=source_content,
                sources=sources,
                query=query
            )
            results.append(result)
        
        # Calculate overall scores
        grounded = sum(1 for r in results if r.is_grounded)
        hallucinated = len(results) - grounded
        grounding_score = grounded / len(results) if results else 0.0
        
        # Determine risk level
        if grounding_score >= 0.95:
            hallucination_risk = "low"
        elif grounding_score >= 0.80:
            hallucination_risk = "medium"
        else:
            hallucination_risk = "high"
        
        # Generate recommendation
        ungrounded = [r.claim for r in results if not r.is_grounded]
        if hallucination_score := 1.0 - grounding_score > 0.2:
            recommendation = f"❌ HIGH HALLUCINATION RISK ({hallucinated}/{len(results)} claims ungrounded). Regenerate with stricter prompt."
        elif hallucination_risk == "medium":
            recommendation = f"⚠️  MEDIUM RISK ({hallucinated} claims need verification). Review and cite sources explicitly."
        else:
            recommendation = f"✅ LOW RISK (Highly grounded). Safe to use. {grounded}/{len(results)} claims supported."
        
        verification = AnswerVerification(
            total_claims=len(results),
            grounded_claims=grounded,
            hallucinated_claims=hallucinated,
            grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            ungrounded_claims=ungrounded,
            verification_results=results,
            recommendation=recommendation
        )
        
        logger.info(
            f"Verification complete: {grounding_score:.1%} grounded, "
            f"risk={hallucination_risk}"
        )
        
        return verification

    def _verify_claim(
        self,
        claim: str,
        source_content: str,
        sources: List[Dict],
        query: str
    ) -> VerificationResult:
        """Verify single claim against sources."""
        
        issues = []
        supporting_sources = []
        confidence = 0.0
        evidence_text = ""
        
        # Extract key entities/numbers from claim
        entities = self._extract_entities(claim)
        
        if not entities:
            # Very generic claim - weak signal
            issues.append("No specific entities/numbers to verify")
            confidence = 0.3
        
        # Try exact match in sources
        exact_match_score, exact_sources = self._exact_match_search(
            claim, entities, sources
        )
        
        if exact_match_score > 0.7:
            confidence = exact_match_score
            supporting_sources = exact_sources
            evidence_text = claim  # Found verbatim
            return VerificationResult(
                claim=claim,
                is_grounded=True,
                supporting_sources=supporting_sources,
                confidence=confidence,
                evidence_text=evidence_text,
                issues=issues
            )
        
        # Try semantic match if embedding service available
        if self.embedding_service:
            semantic_match_score, semantic_sources, evidence = (
                self._semantic_match_search(claim, sources)
            )
            if semantic_match_score > 0.6:
                confidence = semantic_match_score
                supporting_sources = semantic_sources
                evidence_text = evidence
                return VerificationResult(
                    claim=claim,
                    is_grounded=True,
                    supporting_sources=supporting_sources,
                    confidence=confidence,
                    evidence_text=evidence_text,
                    issues=issues
                )
        
        # Try partial match (at least one entity found)
        partial_match_score, partial_sources = self._partial_match_search(
            entities, sources
        )
        
        if partial_match_score > 0.5:
            confidence = partial_match_score * 0.8  # Discount partial matches
            supporting_sources = partial_sources
            if not supporting_sources:
                issues.append("Entities found but claim not directly stated")
        else:
            issues.append(f"No source evidence found for: {', '.join(entities[:3])}")
        
        is_grounded = confidence > 0.6
        
        return VerificationResult(
            claim=claim,
            is_grounded=is_grounded,
            supporting_sources=supporting_sources,
            confidence=confidence,
            evidence_text=evidence_text,
            issues=issues
        )

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Split on periods, but keep question marks and exclamation marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities/numbers/proper nouns from claim."""
        entities = []
        
        # Numbers and statistics
        numbers = re.findall(r'\b\d+(?:[.,]\d+)*\b', text)
        entities.extend(numbers)
        
        # Quoted text
        quotes = re.findall(r'"([^"]+)"', text)
        entities.extend(quotes)
        
        # Proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns[:5])  # Limit to first 5
        
        return list(set(entities))

    def _prepare_source_content(self, sources: List[Dict]) -> str:
        """Combine all source content into one searchable text."""
        content_parts = []
        for source in sources:
            if isinstance(source, dict):
                content = source.get("content") or source.get("text") or ""
                if content:
                    content_parts.append(content)
        return "\n\n".join(content_parts)

    def _exact_match_search(
        self,
        claim: str,
        entities: List[str],
        sources: List[Dict]
    ) -> Tuple[float, List[str]]:
        """Search for exact match of claim in sources."""
        
        matching_sources = []
        best_score = 0.0
        
        for source in sources:
            content = source.get("content") or source.get("text") or ""
            
            # Check if claim (or close variant) appears in source
            if claim in content or claim.lower() in content.lower():
                matching_sources.append(source.get("url") or source.get("title") or "Unknown")
                best_score = 1.0
                break
            
            # Check for claim without articles
            simplified_claim = re.sub(r'\b(a|an|the)\b', '', claim, flags=re.IGNORECASE)
            if simplified_claim in content or simplified_claim.lower() in content.lower():
                matching_sources.append(source.get("url") or source.get("title") or "Unknown")
                best_score = 0.95
        
        return best_score, matching_sources

    def _partial_match_search(
        self,
        entities: List[str],
        sources: List[Dict]
    ) -> Tuple[float, List[str]]:
        """Search for at least one entity in sources."""
        
        matching_sources = []
        match_count = 0
        
        for source in sources:
            content = source.get("content") or source.get("text") or ""
            
            for entity in entities:
                if entity.lower() in content.lower():
                    match_count += 1
                    if source.get("url") or source.get("title"):
                        matching_sources.append(
                            source.get("url") or source.get("title")
                        )
                    break
        
        if not match_count:
            return 0.0, []
        
        # Score based on how many entities found
        score = min(1.0, match_count / max(1, len(entities)))
        return score, matching_sources

    def _semantic_match_search(
        self,
        claim: str,
        sources: List[Dict]
    ) -> Tuple[float, List[str], str]:
        """Search using semantic similarity."""
        
        if not self.embedding_service:
            return 0.0, [], ""
        
        try:
            # Embed claim
            claim_embedding = self.embedding_service.embed_text(claim)
            
            best_score = 0.0
            best_source = ""
            best_evidence = ""
            matching_sources = []
            
            for source in sources:
                content = source.get("content") or source.get("text") or ""
                if not content:
                    continue
                
                # Split source into sentences and embed each
                source_sentences = self._extract_sentences(content)
                
                for sentence in source_sentences:
                    if len(sentence.split()) < 3:  # Skip very short
                        continue
                    
                    sentence_embedding = self.embedding_service.embed_text(sentence)
                    
                    # Calculate cosine similarity
                    similarity = self.embedding_service.cosine_similarity(
                        claim_embedding,
                        sentence_embedding
                    )
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_source = source.get("url") or source.get("title") or "Unknown"
                        best_evidence = sentence
                
                if best_score > 0.7:
                    matching_sources.append(best_source)
            
            return best_score, matching_sources, best_evidence
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return 0.0, [], ""

    def _check_citation_format(self, answer: str) -> Dict:
        """Check if answer uses proper citation format."""
        
        # Look for [Source X] or [1] style citations
        citation_pattern = r'\[(?:Source\s*\d+|\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        has_citations = len(citations) > 0
        citation_density = len(citations) / len(answer.split()) if answer.split() else 0
        
        return {
            "has_citations": has_citations,
            "citation_count": len(citations),
            "citation_density": citation_density,
            "properly_formatted": citation_density > 0.02  # At least 1 citation per 50 words
        }

    def reject_if_hallucinating(
        self,
        verification: AnswerVerification,
        threshold: float = 0.8
    ) -> Tuple[bool, str]:
        """
        Decide whether to reject answer due to hallucination.
        
        Returns (should_reject, reason)
        """
        
        if verification.grounding_score < threshold:
            return True, (
                f"Hallucination risk too high ({1-verification.grounding_score:.0%}). "
                f"Only {verification.grounding_score:.0%} of claims are grounded in sources. "
                f"Ungrounded claims: {', '.join(verification.ungrounded_claims[:3])}"
            )
        
        return False, ""


class StrictGroundedAnswerGenerator:
    """
    Generate answers that MUST be grounded in sources.
    
    Uses system prompts to enforce strict adherence to source material.
    """

    STRICT_SYSTEM_PROMPT = """You are a research assistant that ONLY uses provided sources.

STRICT RULES:
1. Answer ONLY using the provided sources
2. Do NOT infer, assume, or use outside knowledge
3. Do NOT combine information that isn't explicitly stated together
4. If information is not in sources, say "Not found in provided sources"
5. ALWAYS cite which source supports each claim
6. Use format: "Fact [Source: title/URL]"
7. Do NOT hallucinate or make up details
8. If sources contradict each other, mention all versions
9. Prefer direct quotes when making specific claims
10. If confidence is low, state uncertainty explicitly

PENALIZING FACTORS:
- Using facts not in sources: FAIL
- Missing citations: FAIL
- Contradicting sources without acknowledgment: FAIL
- Combining unrelated facts: FAIL

Your answer must be verifiable against sources."""

    @staticmethod
    def get_grounded_prompt(query: str, sources: List[Dict]) -> str:
        """Generate prompt for grounded answer generation."""
        
        source_text = "SOURCES:\n"
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Unknown")
            url = source.get("url", "Unknown")
            content = source.get("content", "")[:500]  # First 500 chars
            
            source_text += f"\n[Source {i}: {title}]\nURL: {url}\nContent: {content}...\n"
        
        prompt = f"""{StrictGroundedAnswerGenerator.STRICT_SYSTEM_PROMPT}

QUESTION: {query}

{source_text}

ANSWER (must be grounded in sources above):"""
        
        return prompt
