
from __future__ import annotations

from typing import Any, Dict, List

from tavily import TavilyClient

from config import TavilyConfig, get_tavily_config
from tools.base import BaseSearchTool, SearchResponse, SearchResult
from utils.logger import get_logger
from utils.domain_filter import is_excluded_domain

logger = get_logger(__name__)


class TavilySearchTool(BaseSearchTool):
    """
    Wraps the Tavily search API.

    Config is injected at construction time — no global state.
    """

    def __init__(self, config: TavilyConfig | None = None) -> None:
        self._cfg = config or get_tavily_config()
        self._client = TavilyClient(api_key=self._cfg.api_key)
        logger.info("TavilySearchTool initialised (max_results=%d)", self._cfg.max_results)

    # ── BaseSearchTool interface ──────────────────────────────────────────────

    def provider_name(self) -> str:
        return "Tavily"

    def search(self, query: str, **kwargs) -> SearchResponse:
        max_results: int = kwargs.pop("max_results", self._cfg.max_results)
        search_depth: str = kwargs.pop("search_depth", "advanced")

        logger.debug("Tavily search | query=%r depth=%s", query, search_depth)

        try:
            raw: Dict[str, Any] = self._client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,
                include_published_date=True,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            raise

        results: List[SearchResult] = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
                raw_content=r.get("raw_content"),
                published_date=r.get("published_date"),
            )
            for r in raw.get("results", [])
            if not is_excluded_domain(r.get("url", ""))  # Filter excluded domains
        ]

        excluded_count = len(raw.get("results", [])) - len(results)
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} results from excluded domains (Reddit, Wikipedia, blogs)")

        return SearchResponse(
            query=query,
            results=results,
            answer=raw.get("answer"),
            follow_up_questions=raw.get("follow_up_questions") or [],
        )
