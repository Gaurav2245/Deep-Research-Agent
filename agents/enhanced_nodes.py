"""
Enhanced node implementations with database integration.

These nodes extend the original agents/nodes.py with:
- Source scoring and filtering
- Vector embeddings
- Data validation
- Confidence scoring
- Database persistence
"""
from __future__ import annotations

import hashlib

import json
from typing import Callable, List
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from agents.state import ResearchState
from config import AgentConfig, ResearchDepth, get_agent_config
from database import SessionLocal, Research, Source
from database.source_scorer import SourceScorer, SourceFilter
from database.embedding_service import EmbeddingService, get_embedding_service
from database.confidence_scorer import ConfidenceScorer
from database.data_validator import DataValidator
from tools.base import BaseSearchTool, SearchResponse
from utils.logger import get_logger

logger = get_logger(__name__)

NodeFn = Callable[[ResearchState], ResearchState]


def _json_safe_validation_rows(results: list | None) -> list:
    """Ensure validation issue rows are JSON-serializable for Postgres JSON columns."""
    out: list = []
    for row in results or []:
        if not isinstance(row, dict):
            continue
        clean: dict = {}
        for k, v in row.items():
            if k == "verification_details":
                continue
            try:
                json.dumps(v)
                clean[k] = v
            except TypeError:
                clean[k] = str(v)[:4000]
        out.append(clean)
    return out


# Helper Functions

def _parse_json_list(text: str) -> List[str]:
    """Safely parse a JSON array from an LLM response."""
    text = text.strip()
    for fence in ("```json", "```"):
        text = text.removeprefix(fence).removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON list from LLM output: %r", text)
    return []


def _format_context(responses: List[SearchResponse]) -> str:
    """Flatten search results into a single readable context string."""
    chunks: List[str] = []
    for resp in responses:
        for r in resp.results:
            chunks.append(
                f"### {r.title}\nURL: {r.url}\n\n{r.content}\n"
            )
        if resp.answer:
            chunks.append(f"### Direct answer from search\n{resp.answer}\n")
    return "\n---\n".join(chunks)


# Enhanced Nodes

def make_source_scorer_node(config: AgentConfig | None = None) -> NodeFn:
    """
    Nested multi-stage scoring:
    1. Vector similarity (FAISS)
    2. Recency/Freshness
    3. Domain Authority/Reliability
    """
    cfg = config or get_agent_config()
    embedding_service = get_embedding_service()

    def source_scorer_node(state: ResearchState) -> ResearchState:
        if not state.has_new_data and state.scored_sources:
            logger.info("[SourceScorer] Skipping - no new data found")
            return state

        logger.info("[SourceScorer] Starting nested multi-stage scoring")
        
        all_results = []
        for response in state.search_responses:
            for result in response.results:
                if result.url not in state.processed_urls:
                    all_results.append(result)
        
        if not all_results:
            logger.info("[SourceScorer] No new results to score")
            return state

        # Stage 1: Vector Similarity (FAISS)
        logger.info("[SourceScorer] Stage 1: Vector Similarity Filtering for %d new results", len(all_results))
        try:
            query_embedding = embedding_service.embed_text(state.query)
            state.query_embedding = query_embedding
            
            contents = [r.content for r in all_results]
            content_embeddings = embedding_service.embed_texts(contents)
            
            # Use FAISS to find top 20 most similar
            similar_indices = embedding_service.find_most_similar_faiss(
                query_embedding, content_embeddings, top_k=20
            )
            
            stage1_results = []
            for idx, sim in similar_indices:
                res = all_results[idx]
                res_dict = {
                    "title": res.title,
                    "url": res.url,
                    "content": res.content,
                    "relevance_score": getattr(res, 'score', 0.5),
                    "vector_similarity": sim,
                    "published_date": getattr(res, 'published_date', None)
                }
                stage1_results.append(res_dict)
                
            logger.info("[SourceScorer] Stage 1 complete: %d sources selected", len(stage1_results))
        except Exception as e:
            logger.error("[SourceScorer] Stage 1 failed: %s. Using all results.", e)
            stage1_results = [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                    "relevance_score": getattr(r, 'score', 0.5),
                    "vector_similarity": 0.5,
                    "published_date": getattr(r, 'published_date', None)
                }
                for r in all_results
            ]

        # Stage 2: Recency/Freshness
        logger.info("[SourceScorer] Stage 2: Recency Scoring")
        for res in stage1_results:
            # Prefer explicit metadata date if available
            date_str = res.get("published_date")
            source_type = "metadata" if date_str else "none"
            
            # Fallback to context-aware regex if metadata is missing
            if not date_str:
                import re
                snippet = res["content"][:2000]
                
                # 1. Look for full dates like "May 15, 2026" or "15 May 2026" or "2026-05-15"
                # This is much more precise than just a 4-digit year.
                full_date_match = re.search(
                    r"(?:published|updated|posted|on|dated)\s*(?::|on)?\s*"
                    r"(\b(?:\d{1,2}[-/th\s]+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/th\s]+\d{1,2}(?:[-/th\s]+\d{2,4})?\b|"
                    r"\b\d{4}-\d{1,2}-\d{1,2}\b|"
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)", 
                    snippet, re.IGNORECASE
                )
                
                if full_date_match:
                    date_str = full_date_match.group(1)
                    source_type = "full_date_regex"
                else:
                    # 2. Fallback to year-only with context
                    year_match = re.search(r"(?:published|updated|date|posted|on|modified)\s*(?::|on)?\s*\b(202[0-9])\b", snippet, re.IGNORECASE)
                    if year_match:
                        date_str = year_match.group(1)
                        source_type = "year_regex"
            
            logger.info(f"[SourceScorer] Date detection for {res['url'][:30]}...: found {date_str} via {source_type}")
            
            freshness = SourceScorer.score_content_freshness(date_str)
            res["content_freshness"] = freshness
            res["content_date"] = date_str
        
        # Sort by freshness + similarity
        stage1_results.sort(key=lambda x: (x["content_freshness"] * 0.4 + x["vector_similarity"] * 0.6), reverse=True)
        stage2_results = stage1_results[:15]
        logger.info("[SourceScorer] Stage 2 complete")

        # Stage 3: Domain Authority & Final Scoring
        logger.info("[SourceScorer] Stage 3: Reliability & Final Selection")
        final_scored = []
        for res in stage2_results:
            scores = SourceScorer.calculate_source_score(
                url=res["url"],
                raw_search_score=res["relevance_score"],
                content=res["content"],
                query=state.query,
                content_date=res.get("content_date"),
                title=res.get("title")
            )
            
            res.update(scores)
            res["overall_score"] = (res["overall_score"] + res["vector_similarity"]) / 2
            final_scored.append(res)
            
        # Merge with existing scored sources and deduplicate by normalized URL
        from utils.domain_filter import normalize_url
        existing_sources = getattr(state, 'scored_sources', [])
        seen_norms = {normalize_url(s["url"]) for s in final_scored}
        for s in existing_sources:
            norm = normalize_url(s["url"])
            if norm not in seen_norms:
                final_scored.append(s)
                seen_norms.add(norm)

        diverse = SourceFilter.select_best_sources(final_scored, count=15, min_score=0.4)
        
        logger.info(
            "[SourceScorer] Final selection: %d sources (added %d new)",
            len(diverse), len([s for s in diverse if s["url"] not in [ex["url"] for ex in existing_sources]])
        )
        
        state.scored_sources = diverse
        # Update processed_urls
        for r in all_results:
            if r.url not in state.processed_urls:
                state.processed_urls.append(r.url)

        return state

    return source_scorer_node


def make_embedder_node(config: AgentConfig | None = None) -> NodeFn:
    """
    Generate vector embeddings for query and sources for semantic search.
    """
    cfg = config or get_agent_config()
    embedding_service = get_embedding_service()

    def embedder_node(state: ResearchState) -> ResearchState:
        if not state.has_new_data and state.source_embeddings:
            logger.info("[Embedder] Skipping - no new data to embed")
            return state
            
        logger.info("[Embedder] Generating embeddings for current best sources")
        
        try:
            # Embed the query if not already present
            if state.query_embedding is None:
                state.query_embedding = embedding_service.embed_text(state.query)
            
            # Embed current scored source contents (batch)
            source_contents = [s.get("content", "") for s in getattr(state, 'scored_sources', [])]
            if source_contents:
                source_embeddings = embedding_service.embed_texts(source_contents)
                state.source_embeddings = source_embeddings
                logger.info("[Embedder] Updated embeddings for %d sources", len(source_embeddings))
            
            # Embed aggregated context for reference
            combined_context = "\n\n".join(state.context)
            if combined_context:
                state.context_embedding = embedding_service.embed_text(combined_context[:8000]) # Cap for efficiency
                
        except Exception as e:
            logger.error("[Embedder] Failed to generate embeddings: %s", e)
        
        return state

    return embedder_node


def make_data_validator_node(config: AgentConfig | None = None) -> NodeFn:
    """
    Validate data for completeness, consistency, and hallucinations.
    """
    cfg = config or get_agent_config()
    embedding_service = get_embedding_service()

    def data_validator_node(state: ResearchState) -> ResearchState:
        if not state.final_answer:
            logger.info("[DataValidator] Skipped (no answer yet)")
            return state
        
        logger.info("[DataValidator] Running validation checks")
        
        # Prepare source data
        sources = [
            {
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "content": s.get("content", ""),
            }
            for s in getattr(state, 'scored_sources', [])
        ]
        
        # Run all validations
        validator = DataValidator(embedding_service=embedding_service)
        validation_results = validator.validate_all(
            final_answer=state.final_answer,
            sources=sources,
            query=state.query,
        )
        
        # Store validation results
        state.validation_results = validation_results
        state.data_quality_score = validation_results["overall_quality_score"]
        
        # Flag if hallucination detected
        for result in validation_results["results"]:
            if result.get("validation_type") == "hallucination" and not result.get("passed"):
                state.hallucination_flagged = True
                logger.warning("[DataValidator] Hallucination markers detected!")
        
        logger.info(
            "[DataValidator] Quality score: %.2f (issues: %d)",
            state.data_quality_score,
            validation_results["issues_found"]
        )
        
        return state

    return data_validator_node


def make_confidence_scorer_node(config: AgentConfig | None = None) -> NodeFn:
    """
    Calculate overall confidence and determine if research is complete.
    """
    cfg = config or get_agent_config()
    min_confidence = 0.7  # 70% confidence threshold

    def confidence_scorer_node(state: ResearchState) -> ResearchState:
        logger.info("[ConfidenceScorer] Calculating research confidence based on context")
        
        sources = [
            {
                "url": s.get("url", ""),
                "overall_score": s.get("overall_score", 0.0),
                "content": s.get("content", ""),
            }
            for s in getattr(state, 'scored_sources', [])
        ]
        
        current_embeddings = getattr(state, 'source_embeddings', [])
        previous_embeddings = getattr(state, 'previous_embeddings', [])
        
        # Calculate Information Gain
        info_gain = ConfidenceScorer.calculate_information_gain(
            current_embeddings, 
            previous_embeddings
        )
        state.information_gain = info_gain
        
        # Calculate confidence based on context only
        confidence_data = ConfidenceScorer.calculate_context_confidence(
            sources=sources,
            embeddings=current_embeddings,
        )
        
        state.confidence_score = confidence_data["overall_confidence"]
        state.confidence_breakdown = confidence_data
        
        # Detect if we have new follow-up queries or contradictions
        has_follow_ups = len(state.search_queries) > 0
        has_contradictions = any(
            r.get("validation_type") == "contradiction" and not r.get("passed")
            for r in state.validation_results.get("results", [])
        )

        # Force stop if Consistency is exceptionally high (Absolute Agreement)
        consistency = confidence_data.get("data_consistency", 0.0)
        absolute_agreement = consistency > 0.95

        should_continue, reason = ConfidenceScorer.should_continue_research(
            confidence_data["overall_confidence"],
            min_confidence=min_confidence,
            iterations=state.iteration,
            max_iterations=cfg.max_search_iterations,
            has_follow_ups=has_follow_ups if not absolute_agreement else False,
            has_contradictions=has_contradictions
        )

        ck = getattr(state, "conversational_knowledge", None) or {}
        if (
            should_continue
            and not has_follow_ups
            and ck
            and getattr(state, "is_follow_up", False)
        ):
            should_continue = False
            reason = (
                "Structured conversational knowledge available and no pending searches; "
                "skipping further web research for this follow-up."
            )

        # Early termination only when there is nothing queued for the next web_search round.
        # (If follow_up produced new queries, we must not stop just because the last round was dry.)
        if should_continue and not has_follow_ups:
            if not state.has_new_data and state.iteration >= 1:
                should_continue = False
                reason = "No new data in last retrieval and no pending search queries."
            elif (info_gain is not None and info_gain <= 0.01) and state.iteration >= 1:
                should_continue = False
                reason = f"Negligible information gain ({info_gain:.4f}) with no pending queries."

        # Stagnation: same snapshot twice in a row with nothing to run → non-progress loop
        fp_src = "|".join(
            [
                ",".join(sorted(state.search_queries or [])),
                str(bool(state.has_new_data)),
                f"{float(info_gain):.6f}",
                str(len(state.scored_sources or [])),
                str(state.iteration),
                str(len(state.attempted_queries or [])),
            ]
        )
        fp = hashlib.sha256(fp_src.encode("utf-8")).hexdigest()[:32]
        prev_fp = getattr(state, "research_progress_fingerprint", "") or ""
        if fp == prev_fp and not has_follow_ups:
            state.research_stagnation_repeats = getattr(state, "research_stagnation_repeats", 0) + 1
        else:
            state.research_stagnation_repeats = 0
        state.research_progress_fingerprint = fp
        if should_continue and not has_follow_ups and state.research_stagnation_repeats >= 1:
            should_continue = False
            reason = "Stagnation: repeated non-progress state (no pending queries)."

        state.should_continue_research = should_continue
        state.confidence_reason = reason
        
        # Store current embeddings for next round comparison
        state.previous_embeddings = current_embeddings
        
        logger.info(
            "[ConfidenceScorer] Confidence: %.2f | Gain: %.2f | New Data: %s | Continue: %s",
            state.confidence_score,
            info_gain,
            state.has_new_data,
            should_continue
        )
        
        return state

    return confidence_scorer_node


def make_database_persistence_node(db_session: Session | None = None) -> NodeFn:
    """
    Save research results to PostgreSQL.
    """
    def database_persistence_node(state: ResearchState) -> ResearchState:
        db = db_session or SessionLocal()
        
        try:
            logger.info("[DbPersistence] Saving research to database")
            
            # Find or create research record
            research = None
            if getattr(state, 'research_id', None):
                try:
                    import uuid
                    r_id = uuid.UUID(state.research_id) if isinstance(state.research_id, str) else state.research_id
                    research = db.query(Research).get(r_id)
                except Exception as e:
                    logger.warning(f"[DbPersistence] Invalid research_id format: {e}")

            if not research:
                research = Research(
                    query=state.query,
                    final_answer=state.final_answer,
                    confidence_score=getattr(state, 'confidence_score', 0.0),
                    data_quality_score=getattr(state, 'data_quality_score', 0.0),
                    validation_issues=_json_safe_validation_rows(
                        state.validation_results.get("results", [])
                    ),
                    research_complete=True,
                    total_iterations=state.iteration,
                    follow_up_questions=state.follow_up_questions,
                    hallucination_flagged=getattr(state, 'hallucination_flagged', False),
                    chat_history=getattr(state, 'chat_history', []),
                    # v2.3 Conversational Cognition
                    understood_intent=getattr(state, 'understood_intent', state.query),
                    query_reasoning=getattr(state, 'query_reasoning', ""),
                    active_topic=getattr(state, 'active_topic', ""),
                    is_follow_up=getattr(state, 'is_follow_up', False),
                    entities_resolved=getattr(state, 'entities_resolved', {}),
                    conversational_knowledge=getattr(state, 'conversational_knowledge', {}) or {},
                )
                db.add(research)
                db.flush()
            else:
                research.final_answer = state.final_answer
                research.confidence_score = getattr(state, 'confidence_score', 0.0)
                research.data_quality_score = getattr(state, 'data_quality_score', 0.0)
                research.validation_issues = _json_safe_validation_rows(
                    state.validation_results.get("results", [])
                )
                research.research_complete = True
                research.total_iterations = state.iteration
                research.follow_up_questions = state.follow_up_questions
                research.hallucination_flagged = getattr(state, 'hallucination_flagged', False)
                research.chat_history = getattr(state, 'chat_history', [])
                # v2.3 Conversational Cognition
                research.understood_intent = getattr(state, 'understood_intent', state.query)
                research.query_reasoning = getattr(state, 'query_reasoning', "")
                research.active_topic = getattr(state, 'active_topic', "")
                research.is_follow_up = getattr(state, 'is_follow_up', False)
                research.entities_resolved = getattr(state, 'entities_resolved', {})
                research.conversational_knowledge = getattr(state, 'conversational_knowledge', {}) or {}

            # Save detailed validation records
            from database.models import DataValidation
            for val in state.validation_results.get("results", []):
                validation_rec = DataValidation(
                    research_id=research.id,
                    validation_type=val.get("validation_type", "unknown"),
                    passed=val.get("passed", False),
                    confidence=val.get("confidence", 0.0),
                    issue_description=val.get("reason", ""),
                    severity="warning" if not val.get("passed") else "info"
                )
                db.add(validation_rec)

            # Generate embedding for query
            try:
                embedding_service = get_embedding_service()
                research.embedding = embedding_service.embed_text(state.query)
            except Exception as e:
                logger.warning("[DbPersistence] Failed to embed query: %s", e)
            
            # Save sources
            from database.models import Source, SourceScore
            
            # Get existing source URLs for this research to prevent duplicates
            existing_source_urls = {s.url for s in research.sources} if research.id else set()
            
            for scored_source in getattr(state, 'scored_sources', []):
                url = scored_source.get("url", "")
                if not url or url in existing_source_urls:
                    continue
                    
                source = Source(
                    research_id=research.id,
                    title=scored_source.get("title", ""),
                    url=url,
                    content=scored_source.get("content", ""),
                    source_score=scored_source.get("overall_score", 0.0),
                    relevance_score=scored_source.get("relevance", 0.0),
                    authority_score=scored_source.get("authority", 0.0),
                    recency_score=scored_source.get("freshness", 0.0),
                    content_quality=scored_source.get("content_quality", 0.0),
                    scraped_successfully=True,
                    discovered_at=datetime.utcnow(),
                )
                db.add(source)
                db.flush() # Get source ID
                existing_source_urls.add(url)

                # Save detailed source scores
                source_score_rec = SourceScore(
                    source_id=source.id,
                    domain_authority=scored_source.get("authority", 0.0),
                    content_freshness=scored_source.get("freshness", 0.0),
                    topical_relevance=scored_source.get("relevance", 0.0),
                    factual_consistency=scored_source.get("consistency", 0.5),
                    citation_quality=scored_source.get("citation_quality", 0.0),
                )
                db.add(source_score_rec)
                
                # Try to embed source content
                try:
                    embedding_service = get_embedding_service()
                    source.content_embedding = embedding_service.embed_text(
                        scored_source.get("content", "")
                    )
                except Exception as e:
                    logger.debug("[DbPersistence] Failed to embed source: %s", e)
            
            db.commit()
            state.research_id = research.id
            logger.info("[DbPersistence] Research saved: %s", research.id)
            
        except Exception as e:
            logger.error("[DbPersistence] Failed to save: %s", e)
            db.rollback()
        finally:
            if not db_session:
                db.close()
        
        return state

    return database_persistence_node


# State Extensions

def extend_research_state():
    
    pass
