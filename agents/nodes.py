
from __future__ import annotations

import json
from typing import Callable, List

from langchain_core.language_models import BaseChatModel

from agents.prompts import (
    entity_extractor_prompt,
    follow_up_prompt,
    query_planner_prompt,
    relational_knowledge_prompt,
    synthesiser_prompt,
)
from agents.conversational_knowledge import (
    assimilate_placeholder_entities_from_text,
    build_entity_constrained_search_queries,
    build_synthesis_context_from_memory,
    conversational_memory_covers_entities,
    format_knowledge_for_prompt,
    infer_requested_attributes_from_intent,
    last_prior_assistant_content,
    merge_entity_facts,
    parse_relational_extraction_response,
    query_strings_respect_entity_scope,
    query_strings_respect_scope_context,
)
from agents.state import ResearchState
from config import AgentConfig, ResearchDepth, get_agent_config
from tools.base import BaseSearchTool, SearchResponse
from utils.logger import get_logger

logger = get_logger(__name__)

# Type alias

NodeFn = Callable[[ResearchState], ResearchState]


# Helpers

def _parse_json_list(text: str) -> List[str]:
    """Safely parse a JSON array from an LLM response."""
    text = text.strip()
    # Strip accidental markdown fences
    for fence in ("```json", "```"):
        text = text.removeprefix(fence).removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON list from LLM output: %r", text)
    return []


def _sanitize_extracted_entities(entities: List[str]) -> List[str]:
    """Drop meta-cognitive / prose fragments mistaken for entity names."""
    out: List[str] = []
    for raw in entities:
        s = (raw or "").strip()
        if len(s) < 2 or len(s) > 120:
            continue
        low = s.lower()
        if low.startswith(
            (
                "the user",
                "user is",
                "user's",
                "there is no",
                "there are no",
                "no prior",
                "no players",
                "no mention",
                "referenced in prior",
                "prior context",
                "structured knowledge",
                "tracked entit",
                "this request",
                "requesting",
            )
        ):
            continue
        if "no players" in low or "no batsman" in low or "not referenced" in low:
            continue
        out.append(s)
    return out


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


def _format_chat_history(history: List[dict]) -> str:
    """Format chat history list into a readable string."""
    if not history:
        return "No previous conversation."
    
    formatted = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content") or "[No content]"
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


# Node factories

def make_entity_extractor_node(llm: BaseChatModel) -> NodeFn:
    """
    Extracts key entities and resolves references from history.
    """
    chain = entity_extractor_prompt | llm

    def entity_extractor_node(state: ResearchState) -> ResearchState:
        # Never feed understood_intent as the sole "question" — it is meta prose and
        # contaminates extraction ("The user is requesting...").
        literal_q = (state.query or "").strip()
        intent_ctx = (getattr(state, "understood_intent", None) or "").strip()
        logger.info("[EntityExtractor] Literal user question: %r", literal_q)

        history_str = _format_chat_history(state.chat_history)
        summary = state.conversation_summary or "No summary available."
        ck = getattr(state, "conversational_knowledge", None) or {}
        ck_str = format_knowledge_for_prompt(ck)
        prior = last_prior_assistant_content(state.chat_history) or "(none)"
        if len(prior) > 12000:
            prior = prior[:12000] + "\n…(truncated)"

        response = chain.invoke(
            {
                "literal_user_question": literal_q,
                "reconstructed_intent": intent_ctx or "(none)",
                "chat_history": history_str,
                "conversation_summary": summary,
                "conversational_knowledge": ck_str,
                "prior_assistant_excerpt": prior,
            }
        )
        entities = _sanitize_extracted_entities(_parse_json_list(response.content))

        # Merge with existing entities and deduplicate
        existing = state.extracted_entities or []
        combined = list(dict.fromkeys(existing + entities))
        for _ref, name in (getattr(state, "entities_resolved", None) or {}).items():
            if isinstance(name, str) and name.strip() and name not in combined:
                combined.append(name.strip())
        scoped = getattr(state, "scoped_entities", None) or []
        combined = list(dict.fromkeys(list(scoped) + combined))
        if not combined and ck:
            combined = list(ck.keys())

        state.extracted_entities = combined

        logger.info("[EntityExtractor] Tracked Entities: %s", state.extracted_entities)
        return state

    return entity_extractor_node


def make_query_planner_node(
    llm: BaseChatModel,
    config: AgentConfig | None = None,
) -> NodeFn:
    """
    Generates an initial set of search queries from the user question.
    """
    cfg = config or get_agent_config()
    num_queries = 3 if cfg.research_depth == ResearchDepth.SHALLOW else 5
    chain = query_planner_prompt | llm

    def query_planner_node(state: ResearchState) -> ResearchState:
        # Use understood_intent if available (v2.3 Conversational Cognition)
        target_query = getattr(state, 'understood_intent', state.query)
        logger.info("[QueryPlanner] Evaluating context and planning queries for: %r", target_query)
        
        history_str = _format_chat_history(state.chat_history)
        entities_str = ", ".join(state.extracted_entities) if state.extracted_entities else "None"
        summary = state.conversation_summary or "No summary available."
        ck = getattr(state, "conversational_knowledge", None) or {}
        ck_str = format_knowledge_for_prompt(ck)

        effective_scoped = list(
            dict.fromkeys(
                (getattr(state, "scoped_entities", None) or [])
                + (state.extracted_entities or [])
            )
        )
        if not effective_scoped and ck:
            effective_scoped = list(ck.keys())
        scope_ctx = (getattr(state, "scope_context", None) or "").strip()

        req_attrs = infer_requested_attributes_from_intent(target_query)
        if conversational_memory_covers_entities(ck, effective_scoped, req_attrs):
            logger.info(
                "[QueryPlanner] Memory covers %d scoped entities for attrs %s; skipping web search.",
                len(effective_scoped),
                req_attrs,
            )
            state.search_queries = []
            return state

        scoped_for_prompt = ", ".join(effective_scoped) if effective_scoped else "(none)"
        scope_for_prompt = scope_ctx if scope_ctx else "(none)"

        response = chain.invoke(
            {
                "question": target_query, 
                "num_queries": num_queries,
                "chat_history": history_str,
                "entities": entities_str,
                "conversation_summary": summary,
                "conversational_knowledge": ck_str,
                "scoped_entities": scoped_for_prompt,
                "scope_context": scope_for_prompt,
            }
        )
        
        # Parse the JSON response
        queries = _parse_json_list(response.content)

        # Check if the planner explicitly returned an empty list (meaning context is sufficient)
        # Note: _parse_json_list returns [] on parse error too, but we check if LLM response content 
        # actually looks like an empty list vs a failure.
        content = response.content.strip()
        if content == "[]" or content == "```json\n[]\n```":
            logger.info("[QueryPlanner] Existing context is SUFFICIENT. Skipping new searches.")
            state.search_queries = []
            return state

        if not queries:
            # If it's a new query (no history) and we got no queries, fall back to raw query
            if not state.chat_history:
                logger.warning("[QueryPlanner] LLM returned no queries for new session; falling back to raw query")
                queries = [state.query]
            else:
                # If there is history, maybe it meant to return [], but let's be safe
                logger.info("[QueryPlanner] No new queries generated. Proceeding with existing context.")
                queries = []

        if (
            not queries
            and effective_scoped
            and state.chat_history
            and infer_requested_attributes_from_intent(target_query)
        ):
            logger.info("[QueryPlanner] Empty planner output; using entity-constrained fallback.")
            queries = build_entity_constrained_search_queries(
                effective_scoped, scope_ctx, target_query, num_queries
            )

        if effective_scoped and queries:
            if not query_strings_respect_entity_scope(
                queries, effective_scoped
            ) or not query_strings_respect_scope_context(queries, scope_ctx):
                logger.warning(
                    "[QueryPlanner] Unconstrained or scope-drifting queries; replacing with entity-scoped queries."
                )
                queries = build_entity_constrained_search_queries(
                    effective_scoped, scope_ctx, target_query, num_queries
                )

        logger.info("[QueryPlanner] Planned %d queries: %s", len(queries), queries)
        state.search_queries = queries
        # Initialize attempted queries
        state.attempted_queries.extend(queries)
        return state

    return query_planner_node



def make_web_search_node(
    search_tool: BaseSearchTool,
    config: AgentConfig | None = None,
) -> NodeFn:
    """
    Executes all planned queries and appends results to state.
    """
    cfg = config or get_agent_config()
    search_depth = "advanced" if cfg.research_depth == ResearchDepth.DEEP else "basic"

    def web_search_node(state: ResearchState) -> ResearchState:
        logger.info(
            "[WebSearch] Running %d queries via %s",
            len(state.search_queries),
            search_tool.provider_name(),
        )

        if not state.search_queries:
            state.search_responses = []
            state.has_new_data = False
            return state

        responses: List[SearchResponse] = []
        all_follow_ups: List[str] = []
        new_unique_results_found = False
        
        # Initialize processed_urls if not exists
        if not hasattr(state, 'processed_urls'):
            state.processed_urls = []

        for q in state.search_queries:
            try:
                resp = search_tool.search(q, search_depth=search_depth)
                # Filter out redundant results (Information Gain check)
                unique_results = []
                for res in resp.results:
                    # Simple heuristic: if URL already in context or processed, skip
                    if res.url in state.processed_urls or any(res.url in c for c in state.context):
                        logger.debug("[WebSearch] Skipping redundant URL: %s", res.url)
                        continue
                    
                    unique_results.append(res)
                    new_unique_results_found = True
                
                resp.results = unique_results
                if unique_results or resp.answer:
                    responses.append(resp)
                
                all_follow_ups.extend(resp.follow_up_questions)
                logger.debug("[WebSearch] query=%r → %d unique results", q, len(resp.results))

            except Exception as exc:
                logger.error("[WebSearch] Failed for query %r: %s", q, exc)

        state.search_responses = responses # Only store current round's new responses for downstream nodes
        if responses:
            state.context.append(_format_context(responses))
        
        state.has_new_data = new_unique_results_found
        state.follow_up_questions.extend(all_follow_ups)
        state.iteration += 1
        return state

    return web_search_node


def make_follow_up_node(
    llm: BaseChatModel,
    config: AgentConfig | None = None,
) -> NodeFn:
    """
    Evaluates the gathered context and decides if more searches are needed.
    """
    cfg = config or get_agent_config()
    max_follow_ups = 3 if cfg.research_depth == ResearchDepth.DEEP else 0
    chain = follow_up_prompt | llm

    def follow_up_node(state: ResearchState) -> ResearchState:
        if not cfg.enable_follow_up_searches or cfg.research_depth == ResearchDepth.SHALLOW:
            logger.info("[FollowUp] Skipped (depth=shallow or disabled)")
            state.search_queries = []
            return state

        combined_context = "\n\n".join(state.context)
        history_str = _format_chat_history(state.chat_history)
        entities_str = ", ".join(state.extracted_entities) if state.extracted_entities else "None"
        summary = state.conversation_summary or "No summary available."
        ck_str = format_knowledge_for_prompt(getattr(state, "conversational_knowledge", None))

        # Format Search Ledger and Failed Paths
        ledger = "\n".join([f"- {q}" for q in state.attempted_queries]) or "None"
        failed = "\n".join([f"- {d}" for d in state.failed_domains]) or "None"
        
        logger.info("[FollowUp] Evaluating whether more searches are needed | Ledger: %d queries", len(state.attempted_queries))

        response = chain.invoke(
            {
                "question": state.query,
                "context": combined_context[:8000],  # Increased context window slightly
                "max_follow_ups": max_follow_ups,
                "chat_history": history_str,
                "attempted_queries": ledger,
                "failed_domains": failed,
                "entities": entities_str,
                "conversation_summary": summary,
                "conversational_knowledge": ck_str,
            }
        )
        new_queries = _parse_json_list(response.content)
        
        # Filter out any queries the LLM might have repeated despite instructions
        new_queries = [q for q in new_queries if q not in state.attempted_queries]
        
        logger.info("[FollowUp] %d additional queries identified", len(new_queries))
        state.search_queries = new_queries
        state.attempted_queries.extend(new_queries)
        return state

    return follow_up_node


def make_relational_knowledge_extractor_node(llm: BaseChatModel) -> NodeFn:
    """
    After synthesis, extract entity→attribute facts from the answer and merge into
    conversational_knowledge for follow-up turns (memory-first reasoning).
    """
    chain = relational_knowledge_prompt | llm

    def relational_knowledge_node(state: ResearchState) -> ResearchState:
        answer = (state.final_answer or "").strip()
        if not answer:
            return state
        prior = getattr(state, "conversational_knowledge", None) or {}
        try:
            response = chain.invoke(
                {
                    "prior_knowledge": format_knowledge_for_prompt(prior),
                    "assistant_answer": answer[:24000],
                }
            )
            delta = parse_relational_extraction_response(response.content)
            merged = merge_entity_facts(prior, delta)
            heur = assimilate_placeholder_entities_from_text(answer[:24000])
            state.conversational_knowledge = merge_entity_facts(merged, heur)
            logger.info(
                "[RelationalMemory] conversational_knowledge entities=%d (LLM delta keys=%d, heuristic=%d)",
                len(state.conversational_knowledge),
                len(delta),
                len(heur),
            )
        except Exception as exc:
            logger.warning("[RelationalMemory] Extraction failed: %s", exc)
            heur = assimilate_placeholder_entities_from_text(answer[:24000])
            state.conversational_knowledge = merge_entity_facts(prior, heur)
            logger.info(
                "[RelationalMemory] Fallback heuristic entities=%d",
                len(heur),
            )
        return state

    return relational_knowledge_node


def make_synthesiser_node(llm: BaseChatModel) -> NodeFn:
    """
    Combines all gathered context into a final, cited answer.
    Optimized for reasoning quality via semantic deduplication and evidence scoring.
    """
    chain = synthesiser_prompt | llm
    from database.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()

    def synthesiser_node(state: ResearchState) -> ResearchState:
        logger.info("[Synthesiser] Building optimized Evidence Ledger")
        
        # 1. Gather evidence and apply Entity Boost
        scored_sources = getattr(state, 'scored_sources', [])
        evidence_with_scores = []
        
        # Track entities for boosting
        entities = state.extracted_entities or []
        
        for s in scored_sources:
            chunk = f"### {s.get('title', 'Source')}\nURL: {s.get('url')}\nRelevance Score: {s.get('overall_score', 0):.2f}\n\n{s.get('content', '')}"
            # Base score from the source scorer
            base_score = s.get('overall_score', 0.5)
            evidence_with_scores.append({"content": chunk, "score": base_score})
            
        # Add direct answers from search responses
        for resp in state.search_responses:
            if resp.answer:
                chunk = f"### Direct Search Answer\n{resp.answer}"
                evidence_with_scores.append({"content": chunk, "score": 0.8}) # High base score for direct answers
                
        if not evidence_with_scores:
            logger.warning("[Synthesiser] No primary evidence found; falling back to raw context")
            # If no scored sources, we use raw context chunks but they won't have scores
            evidence_chunks = state.context
        else:
            # Apply Entity Boost
            raw_chunks = [e["content"] for e in evidence_with_scores]
            boosts = embedding_service.rank_by_entities(raw_chunks, entities)
            
            for i, (idx, boost) in enumerate(boosts):
                evidence_with_scores[i]["score"] += boost
                
            # Sort by boosted score
            evidence_with_scores.sort(key=lambda x: x["score"], reverse=True)
            evidence_chunks = [e["content"] for e in evidence_with_scores]

        # 2. Semantic Deduplication
        logger.info("[Synthesiser] Deduplicating %d chunks", len(evidence_chunks))
        try:
            # We need embeddings for all chunks for deduplication
            # If we don't have them (e.g. they weren't all in scored_sources), generate them
            chunk_embeddings = embedding_service.embed_texts(evidence_chunks)
            unique_chunks = embedding_service.deduplicate_chunks(evidence_chunks, chunk_embeddings, threshold=0.85)
            logger.info("[Synthesiser] Deduplication reduced chunks from %d to %d", len(evidence_chunks), len(unique_chunks))
            evidence_chunks = unique_chunks
        except Exception as e:
            logger.error("[Synthesiser] Deduplication failed: %s", e)

        # 4. Context Assembly & Capping (approx 25k chars)
        MAX_CONTEXT_CHARS = 25000
        combined_context = ""
        for chunk in evidence_chunks:
            if len(combined_context) + len(chunk) > MAX_CONTEXT_CHARS:
                logger.info("[Synthesiser] Context capped at %d characters", len(combined_context))
                break
            combined_context += chunk + "\n\n---\n\n"

        if not combined_context.strip():
            mem_ctx = build_synthesis_context_from_memory(
                getattr(state, "conversational_knowledge", None),
                state.chat_history,
            )
            if mem_ctx:
                combined_context = mem_ctx
                logger.info("[Synthesiser] Using internal conversational memory (no web evidence assembled)")

        history_str = _format_chat_history(state.chat_history)
        entities_str = ", ".join(state.extracted_entities) if state.extracted_entities else "None"
        summary = state.conversation_summary or "No summary available."
        
        # Use understood_intent if available (v2.3 Conversational Cognition)
        target_query = getattr(state, 'understood_intent', state.query)
        
        logger.info(
            "[Synthesiser] Generating final answer (final context length=%d chars, chunks=%d)",
            len(combined_context),
            len(evidence_chunks)
        )

        if not combined_context.strip():
            logger.warning("[Synthesiser] Combined context is empty!")
            state.final_answer = "I'm sorry, but I couldn't find any relevant information to answer your question."
            return state

        response = chain.invoke(
            {
                "question": target_query, 
                "context": combined_context,
                "chat_history": history_str,
                "entities": entities_str,
                "conversation_summary": summary
            }
        )
        
        # Extract summary from response if present
        import re
        content = response.content
        summary_match = re.search(r"\[SUMMARY:\s*(.*?)\]", content, re.DOTALL | re.IGNORECASE)
        if summary_match:
            new_summary = summary_match.group(1).strip()
            state.conversation_summary = new_summary
            # Clean up the response content by removing the summary tag
            content = re.sub(r"\[SUMMARY:.*?\]", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        
        state.final_answer = content
        logger.info("[Synthesiser] Answer generated (%d chars)", len(state.final_answer))
        return state

    return synthesiser_node
