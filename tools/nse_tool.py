"""
tools/nse_tool.py
Dedicated adapter for NSE India (nseindia.com).

NSE blocks plain HTTP requests.  This tool:
  1. Opens the NSE homepage with Playwright to capture session cookies.
  2. Re-uses those cookies to call NSE's internal JSON REST APIs.
  3. Returns clean, structured data (indices, quotes, option chains, etc.)
     as a normalised SearchResponse so the agent graph needs no changes.

Supported queries (case-insensitive):
  - "nifty 50"             → index snapshot
  - "banknifty"            → index snapshot
  - "quote RELIANCE"       → equity quote
  - "option chain NIFTY"   → option chain (CE + PE)
  - "gainers"              → top gainers
  - "losers"               → top losers
  - "most active"          → most actively traded
  - Any NSE API URL        → raw JSON fetch
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from tools.base import BaseSearchTool, SearchResponse, SearchResult
from utils.logger import get_logger

logger = get_logger(__name__)

# ── NSE API endpoint map ──────────────────────────────────────────────────────

NSE_BASE = "https://www.nseindia.com"

INDEX_MAP: Dict[str, str] = {
    "nifty 50":    "NIFTY 50",
    "nifty50":     "NIFTY 50",
    "nifty":       "NIFTY 50",
    "banknifty":   "NIFTY BANK",
    "nifty bank":  "NIFTY BANK",
    "nifty it":    "NIFTY IT",
    "nifty auto":  "NIFTY AUTO",
    "nifty pharma":"NIFTY PHARMA",
    "sensex":      "SENSEX",          # BSE — note: NSE API won't have this
}

ENDPOINTS: Dict[str, str] = {
    "index":        "/api/equity-stockIndices?index={index}",
    "quote":        "/api/quote-equity?symbol={symbol}",
    "option_chain": "/api/option-chain-indices?symbol={symbol}",
    "gainers":      "/api/live-analysis-variations?index=gainers",
    "losers":       "/api/live-analysis-variations?index=loosers",
    "most_active":  "/api/live-analysis-variations?index=mostactive",
    "market_status":"/api/marketStatus",
    "holidays":     "/api/holiday-master?type=trading",
}


@dataclass
class NSEConfig:
    timeout: int = 20
    max_retries: int = 3
    retry_delay: float = 2.0
    headless: bool = True


class NSETool(BaseSearchTool):
    """
    Live market data from NSE India.

    The tool maintains a requests.Session pre-seeded with cookies
    obtained by visiting the NSE homepage via Playwright.
    Cookies are refreshed automatically when they expire.
    """

    def __init__(self, config: NSEConfig | None = None) -> None:
        self._cfg = config or NSEConfig()
        self._session: Optional[requests.Session] = None
        self._cookie_ts: float = 0.0
        self._cookie_ttl: float = 300.0   # refresh every 5 minutes
        logger.info("NSETool initialised")

    # ── BaseSearchTool interface ──────────────────────────────────────────────

    def provider_name(self) -> str:
        return "NSE India"

    def search(self, query: str, **kwargs) -> SearchResponse:
        """
        Dispatch the query to the appropriate NSE endpoint.

        Supported query patterns
        ------------------------
        "nifty 50"           → index snapshot
        "quote TCS"          → equity quote for TCS
        "option chain NIFTY" → NIFTY option chain
        "gainers"            → top gainers
        "losers"             → top losers
        "most active"        → most active stocks
        "market status"      → market open/close status
        """
        q = query.strip().lower()
        logger.info("[NSETool] Query: %r", query)

        self._ensure_session()

        # ── Route query ───────────────────────────────────────────────────
        if q in ("gainers", "top gainers"):
            return self._fetch_variations("gainers", query)

        if q in ("losers", "top losers"):
            return self._fetch_variations("losers", query)

        if q in ("most active", "most active stocks"):
            return self._fetch_variations("most_active", query)

        if q in ("market status", "market open", "market close"):
            return self._fetch_market_status(query)

        if q.startswith("option chain"):
            symbol = query.split(maxsplit=2)[-1].upper().strip()
            return self._fetch_option_chain(symbol, query)

        if q.startswith("quote "):
            symbol = query[6:].upper().strip()
            return self._fetch_quote(symbol, query)

        # Default: try as index name
        index_key = q
        index_name = INDEX_MAP.get(index_key, query.upper())
        return self._fetch_index(index_name, query)

    # ── Session management ────────────────────────────────────────────────────

    def _ensure_session(self) -> None:
        """Refresh cookies if session is stale."""
        if self._session and (time.time() - self._cookie_ts) < self._cookie_ttl:
            return
        logger.info("[NSETool] Refreshing NSE session cookies via Playwright")
        self._session = self._build_session()
        self._cookie_ts = time.time()

    def _build_session(self) -> requests.Session:
        """Visit NSE homepage with Playwright to capture anti-bot cookies."""
        cookies: Dict[str, str] = {}

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=self._cfg.headless)
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                )
                page = context.new_page()
                page.goto(NSE_BASE, timeout=30_000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)   # let JS set cookies

                for c in context.cookies():
                    cookies[c["name"]] = c["value"]

                browser.close()
            logger.info("[NSETool] Captured %d cookies", len(cookies))

        except Exception as exc:
            logger.warning(
                "[NSETool] Playwright cookie capture failed (%s); "
                "falling back to headerless session — may hit 401s",
                exc,
            )

        session = requests.Session()
        session.cookies.update(cookies)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": NSE_BASE + "/",
                "Connection": "keep-alive",
            }
        )
        return session

    # ── API fetchers ──────────────────────────────────────────────────────────

    def _get(self, path: str) -> Dict[str, Any]:
        url = NSE_BASE + path
        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._session.get(url, timeout=self._cfg.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("[NSETool] Attempt %d failed for %s: %s", attempt, url, exc)
                if attempt < self._cfg.max_retries:
                    # Cookie may be stale — refresh and retry
                    self._session = self._build_session()
                    self._cookie_ts = time.time()
                    time.sleep(self._cfg.retry_delay)
        raise RuntimeError(f"NSE API unavailable after {self._cfg.max_retries} attempts: {url}")

    def _fetch_index(self, index_name: str, original_query: str) -> SearchResponse:
        path = ENDPOINTS["index"].format(index=quote(index_name))
        data = self._get(path)

        records = data.get("data", [])
        table_md = _records_to_markdown(
            records,
            columns=["symbol", "lastPrice", "change", "pChange", "totalTradedVolume"],
            headers=["Symbol", "Last Price", "Change", "% Change", "Volume"],
        )

        summary = data.get("metadata", {})
        advance = data.get("advance", {})

        content = (
            f"## {index_name} — Live Snapshot\n\n"
            f"**Last:** {summary.get('last', 'N/A')}  "
            f"**Change:** {summary.get('change', 'N/A')} "
            f"({summary.get('percentChange', 'N/A')}%)\n"
            f"**Advances:** {advance.get('advances', 'N/A')}  "
            f"**Declines:** {advance.get('declines', 'N/A')}  "
            f"**Unchanged:** {advance.get('unchanged', 'N/A')}\n\n"
            f"{table_md}"
        )

        return _make_response(original_query, f"{index_name} Index", NSE_BASE, content)

    def _fetch_quote(self, symbol: str, original_query: str) -> SearchResponse:
        path = ENDPOINTS["quote"].format(symbol=quote(symbol))
        data = self._get(path)

        info = data.get("info", {})
        price = data.get("priceInfo", {})
        meta = data.get("metadata", {})

        content = (
            f"## {symbol} — Equity Quote\n\n"
            f"**Company:** {info.get('companyName', symbol)}\n"
            f"**Industry:** {info.get('industry', 'N/A')}\n"
            f"**Series:** {meta.get('series', 'N/A')}\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Last Price | {price.get('lastPrice', 'N/A')} |\n"
            f"| Open | {price.get('open', 'N/A')} |\n"
            f"| High | {price.get('intraDayHighLow', {}).get('max', 'N/A')} |\n"
            f"| Low | {price.get('intraDayHighLow', {}).get('min', 'N/A')} |\n"
            f"| Prev Close | {price.get('previousClose', 'N/A')} |\n"
            f"| Change | {price.get('change', 'N/A')} ({price.get('pChange', 'N/A')}%) |\n"
            f"| 52W High | {price.get('weekHighLow', {}).get('max', 'N/A')} |\n"
            f"| 52W Low | {price.get('weekHighLow', {}).get('min', 'N/A')} |\n"
            f"| VWAP | {price.get('vwap', 'N/A')} |\n"
        )

        return _make_response(original_query, f"{symbol} Quote", NSE_BASE, content)

    def _fetch_option_chain(self, symbol: str, original_query: str) -> SearchResponse:
        path = ENDPOINTS["option_chain"].format(symbol=quote(symbol))
        data = self._get(path)

        filtered = data.get("filtered", {})
        records = filtered.get("data", [])
        expiry = data.get("records", {}).get("expiryDates", [])
        spot = data.get("records", {}).get("underlyingValue", "N/A")

        rows: List[str] = [
            "| Strike | CE OI | CE Chg OI | CE LTP | PE LTP | PE Chg OI | PE OI |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in records[:30]:   # top 30 strikes around ATM
            ce = r.get("CE", {})
            pe = r.get("PE", {})
            rows.append(
                f"| {r.get('strikePrice', '')} "
                f"| {ce.get('openInterest', '')} "
                f"| {ce.get('changeinOpenInterest', '')} "
                f"| {ce.get('lastPrice', '')} "
                f"| {pe.get('lastPrice', '')} "
                f"| {pe.get('changeinOpenInterest', '')} "
                f"| {pe.get('openInterest', '')} |"
            )

        content = (
            f"## {symbol} Option Chain\n\n"
            f"**Spot Price:** {spot}\n"
            f"**Expiry Dates:** {', '.join(expiry[:6])}\n\n"
            + "\n".join(rows)
        )

        return _make_response(original_query, f"{symbol} Option Chain", NSE_BASE, content)

    def _fetch_variations(self, kind: str, original_query: str) -> SearchResponse:
        path = ENDPOINTS[kind]
        data = self._get(path)

        records = (
            data.get("NIFTY", {}).get("data", [])
            or data.get("data", [])
        )
        table_md = _records_to_markdown(
            records,
            columns=["symbol", "lastPrice", "change", "pChange", "totalTradedVolume"],
            headers=["Symbol", "Last Price", "Change", "% Change", "Volume"],
        )
        label = {"gainers": "Top Gainers", "losers": "Top Losers", "most_active": "Most Active"}[kind]
        content = f"## NSE {label}\n\n{table_md}"
        return _make_response(original_query, label, NSE_BASE, content)

    def _fetch_market_status(self, original_query: str) -> SearchResponse:
        data = self._get(ENDPOINTS["market_status"])
        markets = data.get("marketState", [])
        rows = ["| Market | Status | Message |", "|---|---|---|"]
        for m in markets:
            rows.append(
                f"| {m.get('market', '')} | {m.get('marketStatus', '')} | {m.get('marketStatusMessage', '')} |"
            )
        content = "## NSE Market Status\n\n" + "\n".join(rows)
        return _make_response(original_query, "Market Status", NSE_BASE, content)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _records_to_markdown(
    records: List[Dict],
    columns: List[str],
    headers: List[str],
) -> str:
    if not records:
        return "_No data available._"
    header_row = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for r in records:
        cells = [str(r.get(c, "")) for c in columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_row, sep] + rows)


def _make_response(query: str, title: str, url: str, content: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        results=[SearchResult(title=title, url=url, content=content, score=1.0)],
    )
