"""
tools/nse_router_tool.py
Wraps a default search tool with automatic routing to NSETool for queries
that clearly ask for live Indian stock-market data (index snapshots,
gainers/losers, most-active). Falls back to the default tool for everything
else, and if NSE itself fails for any reason — so live-data routing can
never make a research run less reliable than it already was.
"""
from __future__ import annotations

from typing import Optional

from tools.base import BaseSearchTool, SearchResponse
from tools.query_router import classify_nse_query
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoRoutingSearchTool(BaseSearchTool):
    """Routes Indian stock-market queries to NSETool; everything else goes to the default tool."""

    def __init__(self, default_tool: BaseSearchTool, nse_tool: Optional[BaseSearchTool] = None):
        self.default_tool = default_tool
        self._nse_tool = nse_tool  # lazily constructed: avoids Playwright cost when never used

    def provider_name(self) -> str:
        return f"AutoRouting({self.default_tool.provider_name()} + NSE)"

    def _get_nse_tool(self) -> BaseSearchTool:
        if self._nse_tool is None:
            from tools.nse_tool import NSETool
            self._nse_tool = NSETool()
        return self._nse_tool

    def search(self, query: str, **kwargs) -> SearchResponse:
        canonical_queries = classify_nse_query(query)

        if canonical_queries:
            try:
                nse = self._get_nse_tool()
                responses = [nse.search(cq) for cq in canonical_queries]
                merged_results = [r for resp in responses for r in resp.results]
                answers = [resp.answer for resp in responses if resp.answer]
                logger.info("[AutoRouting] Routed %r to NSE (%s)", query, canonical_queries)
                return SearchResponse(
                    query=query,
                    results=merged_results,
                    answer="\n\n".join(answers) if answers else None,
                )
            except Exception as exc:
                logger.warning(
                    "[AutoRouting] NSE routing failed for %r (%s); falling back to %s",
                    query, exc, self.default_tool.provider_name(),
                )

        return self.default_tool.search(query, **kwargs)
