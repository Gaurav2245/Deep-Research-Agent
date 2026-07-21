
from __future__ import annotations
import json
from typing import Callable
from langchain_core.language_models import BaseChatModel
from agents.state import ResearchState
from agents.prompts import query_understanding_prompt
from agents.conversational_knowledge import (
    bootstrap_knowledge_from_prior_assistant,
    format_knowledge_for_prompt,
    last_prior_assistant_content,
)
from utils.logger import get_logger

logger = get_logger(__name__)

NodeFn = Callable[[ResearchState], ResearchState]


def _vague_plural_reference(raw: str, intent: str) -> bool:
    """User refers to the whole prior set (e.g. table) without naming each entity."""
    blob = f"{raw} {intent}".lower()
    needles = (
        "above mentioned",
        "above-mentioned",
        "mentioned player",
        "those player",
        "same player",
        "listed player",
        "them ",
        " they ",
        "their ",
        "earlier",
        "previous answer",
        "from the table",
        "all of them",
    )
    return any(n in blob for n in needles)

def _format_chat_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    formatted = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content") or "[No content]"
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

def make_query_understanding_node(llm: BaseChatModel) -> NodeFn:
    """
    Performs semantic reconstruction of the user's query.
    """
    chain = query_understanding_prompt | llm

    def query_understanding_node(state: ResearchState) -> ResearchState:
        logger.info("[QueryUnderstanding] Reconstructing intent for: %r", state.query)

        # Machine-usable continuity: seed entity keys from last assistant answer text
        # before LLM (relational extraction may have failed or been empty).
        _ck_before = len(getattr(state, "conversational_knowledge", None) or {})
        state.conversational_knowledge = bootstrap_knowledge_from_prior_assistant(
            state.chat_history,
            getattr(state, "conversational_knowledge", None),
        )
        _ck_after = len(state.conversational_knowledge or {})
        if _ck_after > _ck_before:
            logger.info(
                "[QueryUnderstanding] Bootstrapped %d new entity keys from prior assistant text (CK total=%d)",
                _ck_after - _ck_before,
                _ck_after,
            )

        history_str = _format_chat_history(state.chat_history)
        entities_str = ", ".join(state.extracted_entities) if state.extracted_entities else "None"
        summary = state.conversation_summary or "No summary available."
        ck_str = format_knowledge_for_prompt(getattr(state, "conversational_knowledge", None))
        prior_excerpt = last_prior_assistant_content(state.chat_history) or "(none)"
        if len(prior_excerpt) > 12000:
            prior_excerpt = prior_excerpt[:12000] + "\n…(truncated)"

        try:
            response = chain.invoke(
                {
                    "question": state.query,
                    "chat_history": history_str,
                    "conversation_summary": summary,
                    "entities": entities_str,
                    "conversational_knowledge": ck_str,
                    "prior_assistant_excerpt": prior_excerpt,
                }
            )
            
            content = response.content.strip()
            # Remove markdown fences if present
            for fence in ("```json", "```"):
                content = content.removeprefix(fence).removesuffix("```").strip()
            
            data = json.loads(content)
            
            state.understood_intent = data.get("understood_intent", state.query)
            state.query_reasoning = data.get("reasoning", "")
            state.entities_resolved = data.get("resolved_entities", {})
            state.active_topic = data.get("active_topic", "")
            state.is_follow_up = data.get("is_follow_up", False)

            scoped = data.get("scoped_entities", [])
            if isinstance(scoped, list):
                state.scoped_entities = [
                    str(x).strip() for x in scoped if isinstance(x, str) and str(x).strip()
                ]
            else:
                state.scoped_entities = []
            state.scope_context = (
                str(data.get("scope_context", "") or "").strip()
            )

            ck = getattr(state, "conversational_knowledge", None) or {}
            if (
                not state.scoped_entities
                and ck
                and state.is_follow_up
                and _vague_plural_reference(state.query, state.understood_intent)
            ):
                state.scoped_entities = list(ck.keys())
            
            logger.info("[QueryUnderstanding] Understood Intent: %r", state.understood_intent)
            if state.is_follow_up:
                logger.info("[QueryUnderstanding] Reasoning: %s", state.query_reasoning)
                
        except Exception as e:
            logger.error("[QueryUnderstanding] Failed to reconstruct intent: %s", e)
            # Fallback to raw query
            state.understood_intent = state.query
            ck = getattr(state, "conversational_knowledge", None) or {}
            if (
                not getattr(state, "scoped_entities", None)
                and ck
                and getattr(state, "is_follow_up", False)
                and _vague_plural_reference(state.query, state.understood_intent)
            ):
                state.scoped_entities = list(ck.keys())

        return state

    return query_understanding_node
