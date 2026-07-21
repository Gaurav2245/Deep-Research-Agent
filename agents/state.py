
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.base import SearchResponse


@dataclass
class ResearchState:
   

    query: str
    search_queries: List[str] = field(default_factory=list)
    search_responses: List[SearchResponse] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    iteration: int = 0
    final_answer: Optional[str] = None
    error: Optional[str] = None
    
    # Enhanced v2.0 fields
    scored_sources: List[dict] = field(default_factory=list)
    query_embedding: Optional[List[float]] = None
    source_embeddings: List[List[float]] = field(default_factory=list)
    previous_embeddings: List[List[float]] = field(default_factory=list)
    context_embedding: Optional[List[float]] = None
    validation_results: dict = field(default_factory=dict)
    confidence_score: float = 0.0
    confidence_breakdown: dict = field(default_factory=dict)
    data_quality_score: float = 0.0
    hallucination_flagged: bool = False
    should_continue_research: bool = False
    confidence_reason: str = ""
    research_id: Optional[str] = None
    chat_history: List[dict] = field(default_factory=list)
    
    # v2.1 State-Aware Tracking
    attempted_queries: List[str] = field(default_factory=list)
    failed_domains: List[str] = field(default_factory=list)
    information_gain: float = 1.0
    processed_urls: List[str] = field(default_factory=list)
    has_new_data: bool = True
    # Loop safety: detect repeated non-progress (same queries + no data + no gain)
    research_progress_fingerprint: str = ""
    research_stagnation_repeats: int = 0
    # v2.2 Conversational Grounding
    extracted_entities: List[str] = field(default_factory=list)
    conversation_summary: Optional[str] = None

    # v2.3 Conversational Cognition (NEW)
    understood_intent: str = ""  # The reconstructed, unambiguous intent
    query_reasoning: str = ""  # LLM reasoning for the reconstruction
    entities_resolved: dict = field(default_factory=dict)  # Mapping of pronouns/references to entities
    is_follow_up: bool = False  # Explicit flag for conversational continuity
    active_topic: str = ""  # The primary subject under discussion

    # v2.5 Conversational scope (constraint grounding for retrieval)
    scoped_entities: List[str] = field(default_factory=list)  # Exact entities this turn refers to
    scope_context: str = ""  # Tournament, season, league, domain (e.g. "IPL 2026")

    # v2.4 Structured conversational knowledge (entity → attributes), from prior answers
    conversational_knowledge: Dict[str, Dict[str, Any]] = field(default_factory=dict)

