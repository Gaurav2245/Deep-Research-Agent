"""
tools/query_router.py
Classifies free-text search queries as Indian stock-market data requests,
mapping them to the canonical query strings NSETool understands.

This exists because the query-planner LLM generates natural-language search
strings (e.g. "Nifty 50 top gainers today NSE"), but NSETool only recognises
exact canonical phrases ("gainers", "nifty 50", ...). Classification is
deliberately conservative: it only fires for queries anchored to a
recognizable Indian market token (nifty/sensex/nse/bse/banknifty), so it
doesn't misroute unrelated queries that happen to contain words like
"losers" (e.g. "biggest losers of World War 2").

"quote SYMBOL" and "option chain SYMBOL" are intentionally NOT auto-routed:
LLM-generated queries reference companies by name ("Reliance Industries"),
not ticker symbols ("RELIANCE"), and there's no reliable name->ticker
mapping here. Those remain reachable only via an explicit SEARCH_PROVIDER=nse
query using the exact canonical phrase.
"""
from __future__ import annotations

import re
from typing import List

_MARKET_ANCHORS = ("nifty", "sensex", "nse", "bse", "banknifty", "bank nifty")

# Maps a recognized substring to the exact canonical index name NSETool's
# INDEX_MAP expects. Order matters: longer/more specific phrases first.
_INDEX_NAMES = [
    ("nifty bank", "banknifty"),
    ("bank nifty", "banknifty"),
    ("banknifty", "banknifty"),
    ("nifty it", "nifty it"),
    ("nifty auto", "nifty auto"),
    ("nifty pharma", "nifty pharma"),
    ("nifty 50", "nifty 50"),
    ("nifty50", "nifty 50"),
    ("sensex", "sensex"),
    ("nifty", "nifty 50"),
]


def _has_market_anchor(text: str) -> bool:
    return any(anchor in text for anchor in _MARKET_ANCHORS)


def classify_nse_query(query: str) -> List[str]:
    """
    Map a free-text search query to zero or more canonical NSE query
    strings that NSETool.search() understands (e.g. "gainers", "nifty 50").

    Returns an empty list if the query isn't clearly an Indian stock-market
    data request — callers should fall back to normal web search in that case.
    """
    text = (query or "").strip().lower()
    if not text or not _has_market_anchor(text):
        return []

    canonical: List[str] = []

    if re.search(r"\bgainers?\b", text):
        canonical.append("gainers")
    if re.search(r"\bloo?sers?\b", text):
        canonical.append("losers")
    if "most active" in text or "most traded" in text:
        canonical.append("most active")
    if "market status" in text or (
        "market" in text and any(kw in text for kw in ("open", "closed", "close"))
    ):
        canonical.append("market status")

    if canonical:
        return canonical

    # No action keyword matched — check for a bare index-snapshot query.
    for phrase, resolved in _INDEX_NAMES:
        if phrase in text:
            return [resolved]

    return []
