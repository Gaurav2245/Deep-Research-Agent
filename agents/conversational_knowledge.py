"""
Structured conversational knowledge: entity → attribute → value.
Used for follow-ups that should reuse prior synthesized answers without web search.
"""
from __future__ import annotations

import json
import re
from typing import Any


def format_knowledge_for_prompt(knowledge: dict[str, dict[str, Any]] | None) -> str:
    """Compact, stable rendering for LLM prompts."""
    if not knowledge:
        return "(none — no structured facts stored yet from prior answers in this session.)"
    try:
        return json.dumps(knowledge, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(knowledge)


def merge_entity_facts(
    existing: dict[str, dict[str, Any]],
    delta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Deep-merge per-entity attribute maps (later values override)."""
    out: dict[str, dict[str, Any]] = {k: dict(v) for k, v in existing.items()}
    for entity, attrs in delta.items():
        if entity not in out:
            out[entity] = {}
        out[entity].update(attrs)
    return out


def prior_context_available_for_answer(
    *,
    scored_sources: list | None,
    context: list | None,
    conversational_knowledge: dict[str, Any] | None,
    chat_history: list[dict] | None,
) -> bool:
    """
    True if we can answer without a new web search round: prior sources, prior
    context blobs, structured memory, or an earlier assistant message in history.
    """
    if scored_sources:
        return True
    if context:
        return True
    if conversational_knowledge:
        return True
    h = chat_history or []
    for msg in h[:-1]:
        if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
            return True
    return False


def last_prior_assistant_content(chat_history: list[dict]) -> str:
    """
    Last assistant message before the current user message.
    Streamlit appends the current user turn before invoking the graph, so the
    final message in history is typically the current query.
    """
    if not chat_history or len(chat_history) < 2:
        return ""
    for msg in reversed(chat_history[:-1]):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    return ""


def build_synthesis_context_from_memory(
    conversational_knowledge: dict[str, dict[str, Any]] | None,
    chat_history: list[dict],
) -> str:
    """
    When web/scored evidence is empty, still give the synthesiser usable
    internal context from structured memory + the last assistant answer.
    """
    parts: list[str] = []
    k = conversational_knowledge or {}
    if k:
        parts.append(
            "### Internal conversational knowledge (from prior turns)\n"
            "Use these entity→attribute facts as authoritative for this session "
            "when the user asks follow-ups about the same subjects.\n"
        )
        parts.append(format_knowledge_for_prompt(k))

    prior_answer = last_prior_assistant_content(chat_history)
    if prior_answer:
        parts.append(
            "\n### Previous assistant answer in this conversation (verbatim excerpt)\n"
            + prior_answer[:12000]
        )

    return "\n".join(parts).strip()


def parse_relational_extraction_response(text: str) -> dict[str, dict[str, Any]]:
    """Parse LLM JSON output for relational extraction."""
    text = text.strip()
    for fence in ("```json", "```"):
        text = text.removeprefix(fence).removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    facts = data.get("entity_facts")
    if not isinstance(facts, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entity, attrs in facts.items():
        if not isinstance(entity, str) or not isinstance(attrs, dict):
            continue
        clean_attrs: dict[str, Any] = {}
        for ak, av in attrs.items():
            if isinstance(ak, str):
                clean_attrs[ak] = av
        # Keep entity keys even with empty attrs (cross-turn persistence / partial LLM output).
        out[entity] = clean_attrs
    return out


# --- Assistant output assimilation (no LLM): seed entity keys from prior answers ---

_SKIP_ASSIMILATION_PHRASES = frozenset(
    {
        "summary",
        "key findings",
        "introduction",
        "conclusion",
        "primary data sources",
        "references",
        "sources",
        "note",
        "disclaimer",
        "answer",
        "overview",
    }
)

_TABLE_HEADER_TOKENS = frozenset(
    {
        "player",
        "batsman",
        "batter",
        "bowler",
        "team",
        "runs",
        "sr",
        "strike",
        "rate",
        "average",
        "avg",
        "wickets",
        "economy",
        "eco",
        "match",
        "opponent",
        "year",
        "ipl",
        "rank",
        "position",
        "versus",
        "vs",
        "venue",
        "date",
        "balls",
        "overs",
        "points",
        "metric",
        "value",
        "name",
    }
)


def _looks_like_person_or_team_name(s: str) -> bool:
    if not s or len(s) > 100:
        return False
    if s.lower() in _SKIP_ASSIMILATION_PHRASES:
        return False
    if not re.match(r"^[A-Za-z0-9\s\.'-]+$", s):
        return False
    words = s.split()
    if len(words) >= 2:
        return True
    if len(words) == 1 and len(words[0]) >= 4 and words[0][0].isupper() and words[0].isalpha():
        return True
    return False


def assimilate_placeholder_entities_from_text(
    text: str, max_entities: int = 48
) -> dict[str, dict[str, Any]]:
    """
    Deterministic extraction of likely people/team names from the prior assistant
    message (markdown tables, bold). Merged into conversational_knowledge as
    entity -> {{}} so follow-ups bind even when relational LLM extraction failed.
    """
    facts: dict[str, dict[str, Any]] = {}
    if not text or not text.strip():
        return facts

    for m in re.finditer(r"\*\*([^*]{2,100})\*\*", text):
        inner = m.group(1).strip()
        if inner.lower() in _SKIP_ASSIMILATION_PHRASES:
            continue
        if _looks_like_person_or_team_name(inner):
            facts.setdefault(inner, {})
            if len(facts) >= max_entities:
                return facts

    # "Name | stat" or "Name - 123" or "Name: 45" without markdown bold
    _name_stat_line = re.compile(
        r"^[\s•\-\*]*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]+){1,4})\s*(?:\||[-–:])\s*[\d\.\s]",
        re.MULTILINE,
    )
    for m in _name_stat_line.finditer(text):
        name = m.group(1).strip()
        if _looks_like_person_or_team_name(name):
            facts.setdefault(name, {})
            if len(facts) >= max_entities:
                return facts

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if re.search(r":-+|-+:", line.replace(" ", "")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        first = cells[0]
        low = first.lower()
        if low in _TABLE_HEADER_TOKENS or not first or first.startswith("---"):
            continue
        if len(first) < 3:
            continue
        if _looks_like_person_or_team_name(first):
            facts.setdefault(first, {})
        if len(facts) >= max_entities:
            break

    return facts


def bootstrap_knowledge_from_prior_assistant(
    chat_history: list[dict] | None,
    existing: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Merge heuristic entities from last assistant message into knowledge."""
    prior = last_prior_assistant_content(chat_history or [])
    if not prior:
        return dict(existing or {})
    delta = assimilate_placeholder_entities_from_text(prior)
    if not delta:
        return dict(existing or {})
    return merge_entity_facts(existing or {}, delta)


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def entity_has_semantic_attribute(facts: dict[str, Any], canonical: str) -> bool:
    """Match stored keys to canonical attrs (e.g. strike_rate vs 'Batting SR')."""
    if canonical in facts:
        return True
    cn = _norm_key(canonical)
    for fk in facts:
        if not isinstance(fk, str):
            continue
        fn = _norm_key(fk)
        if fn == cn:
            return True
        if canonical == "strike_rate" and "strike" in fn and "rate" in fn:
            return True
        if canonical == "runs" and fn in ("runs", "total_runs", "run"):
            return True
        if canonical == "wickets" and "wicket" in fn:
            return True
        if canonical == "average" and ("average" in fn or "avg" in fn):
            return True
    return False


def infer_requested_attributes_from_intent(intent: str) -> set[str]:
    """Lightweight attribute hints from reconstructed intent (not exhaustive)."""
    if not intent:
        return set()
    t = intent.lower()
    out: set[str] = set()
    if "strike rate" in t or "strike-rate" in t or "batting strike" in t:
        out.add("strike_rate")
    if re.search(r"\bruns\b", t) and "strike" not in t:
        out.add("runs")
    if "wicket" in t:
        out.add("wickets")
    if "average" in t or re.search(r"\bavg\b", t):
        out.add("average")
    return out


def conversational_memory_covers_entities(
    knowledge: dict[str, dict[str, Any]] | None,
    entities: list[str],
    required_attributes: set[str],
) -> bool:
    """
    True if every entity has values for every required semantic attribute.
    If required_attributes is empty, returns False (unknown what to check).
    """
    if not knowledge or not entities or not required_attributes:
        return False
    for ent in entities:
        facts = knowledge.get(ent)
        if not isinstance(facts, dict) or not facts:
            return False
        for attr in required_attributes:
            if not entity_has_semantic_attribute(facts, attr):
                return False
    return True


def query_strings_respect_entity_scope(
    queries: list[str],
    scoped_entities: list[str],
) -> bool:
    if not scoped_entities:
        return True
    for q in queries:
        ql = q.lower()
        if not any(e.lower() in ql for e in scoped_entities):
            return False
    return True


def query_strings_respect_scope_context(queries: list[str], scope_context: str) -> bool:
    if not scope_context or not scope_context.strip():
        return True
    token = scope_context.strip().lower()
    for q in queries:
        if token not in q.lower():
            return False
    return True


def build_entity_constrained_search_queries(
    entities: list[str],
    scope_context: str,
    intent: str,
    max_queries: int,
) -> list[str]:
    """
    Deterministic queries when the LLM drifts to generic phrasing.
    Keeps at least one entity name + optional scope + metric words from intent.
    """
    if not entities or max_queries <= 0:
        return []
    scope = (scope_context or "").strip()
    intent_l = (intent or "").lower()
    metric_bits: list[str] = []
    if "strike rate" in intent_l or "strike-rate" in intent_l:
        metric_bits.append("strike rate")
    if "runs" in intent_l:
        metric_bits.append("runs")
    if "wicket" in intent_l:
        metric_bits.append("wickets")
    if "average" in intent_l:
        metric_bits.append("batting average")
    if not metric_bits:
        metric_bits.append("statistics")

    metric_phrase = " ".join(dict.fromkeys(metric_bits))
    scope_l = scope.lower()
    sport_anchor = "cricket" if any(
        x in intent_l for x in ("ipl", "cricket", "batsman", "batting", "t20", "odi", "test match")
    ) or any(x in scope_l for x in ("ipl", "psl", "bbl", "wpl", "cricket", "t20")) else ""

    queries: list[str] = []
    if len(entities) <= 6:
        names_join = ", ".join(entities)
        parts = [metric_phrase, names_join]
        if scope:
            parts.insert(0, scope)
        if sport_anchor:
            parts.append(sport_anchor)
        queries.append(" ".join(parts))

    for ent in entities:
        if len(queries) >= max_queries:
            break
        parts = [ent]
        if scope:
            parts.append(scope)
        parts.append(metric_phrase)
        if sport_anchor:
            parts.append(sport_anchor)
        queries.append(" ".join(parts))

    return queries[:max_queries]
