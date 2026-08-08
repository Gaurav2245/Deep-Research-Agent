"""Helpers shared between agents/nodes.py and agents/enhanced_nodes.py."""
from __future__ import annotations

import json
from typing import List

from tools.base import SearchResponse
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_list(text: str) -> List[str]:
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


def format_context(responses: List[SearchResponse]) -> str:
    """Flatten search results into a single readable context string."""
    chunks: List[str] = []
    for resp in responses:
        for r in resp.results:
            chunks.append(f"### {r.title}\nURL: {r.url}\n\n{r.content}\n")
        if resp.answer:
            chunks.append(f"### Direct answer from search\n{resp.answer}\n")
    return "\n---\n".join(chunks)
