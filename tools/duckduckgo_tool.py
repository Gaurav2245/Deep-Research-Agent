
from __future__ import annotations

from typing import List

from duckduckgo_search import DDGS

from config import DuckDuckGoConfig, get_duckduckgo_config
from tools.base import BaseSearchTool, SearchResponse, SearchResult
from utils.logger import get_logger
from utils.domain_filter import is_excluded_domain

logger = get_logger(__name__)


class DuckDuckGoSearchTool(BaseSearchTool):
    """
    Wraps the DuckDuckGo search engine (no API key needed).

    Config is injected at construction time.
    """

    def __init__(self, config: DuckDuckGoConfig | None = None) -> None:
        self._cfg = config or get_duckduckgo_config()
        logger.info(
            "DuckDuckGoSearchTool initialised (max_results=%d)", self._cfg.max_results
        )

    # ── BaseSearchTool interface ──────────────────────────────────────────────

    def provider_name(self) -> str:
        return "DuckDuckGo"

    def search(self, query: str, **kwargs) -> SearchResponse:
        max_results: int = kwargs.pop("max_results", self._cfg.max_results)

        logger.debug("DuckDuckGo search | query=%r max_results=%d", query, max_results)

        try:
            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(query, max_results=max_results)
                )
        except Exception as exc:
            logger.error("DuckDuckGo search failed: %s", exc)
            raise

        results: List[SearchResult] = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                content=r.get("body", ""),
                score=0.0,
            )
            for r in raw_results
            if not is_excluded_domain(r.get("href", ""))  # Filter excluded domains
        ]

        excluded_count = len(raw_results) - len(results)
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} results from excluded domains (Reddit, Wikipedia, blogs)")

        return SearchResponse(
            query=query,
            results=results,
        )
