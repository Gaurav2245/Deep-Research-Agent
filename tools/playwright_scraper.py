"""
tools/playwright_scraper.py
Renders JavaScript-heavy pages using Playwright (headless Chromium),
extracts all text + HTML tables, and returns a normalised SearchResponse.

This is NOT a search engine — it fetches a specific URL you already know.
Use it as a companion to search tools when you need full page content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

from tools.base import BaseSearchTool, SearchResponse, SearchResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScraperConfig:
    headless: bool = True
    timeout_ms: int = 30_000          # page load timeout
    wait_for_selector: Optional[str] = None   # CSS selector to wait for
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    extra_headers: dict = field(default_factory=dict)
    slow_mo_ms: int = 0


class PlaywrightScraperTool(BaseSearchTool):
    """
    Fetches a URL with a real browser, waits for JS rendering,
    and extracts visible text + HTML tables.

    Usage as a search tool:
        scraper = PlaywrightScraperTool()
        response = scraper.search("https://example.com/data-page")
    """

    def __init__(self, config: ScraperConfig | None = None) -> None:
        self._cfg = config or ScraperConfig()
        logger.info("PlaywrightScraperTool initialised (headless=%s)", self._cfg.headless)

    # ── BaseSearchTool interface ──────────────────────────────────────────────

    def provider_name(self) -> str:
        return "PlaywrightScraper"

    def search(self, query: str, **kwargs) -> SearchResponse:
        """
        `query` is treated as a URL when it starts with http/https.
        Falls back to a Google search URL otherwise.
        """
        url = query if query.startswith("http") else f"https://www.google.com/search?q={query}"
        return self.fetch_url(url, **kwargs)

    # ── Core scraping method ─────────────────────────────────────────────────

    def fetch_url(
        self,
        url: str,
        wait_selector: str | None = None,
        extract_tables: bool = True,
        scroll: bool = True,
    ) -> SearchResponse:
        """
        Render `url` with Playwright and return extracted content.

        Parameters
        ----------
        url:            Target URL.
        wait_selector:  CSS selector to wait for before extraction.
        extract_tables: Whether to parse HTML tables into markdown.
        scroll:         Scroll to bottom to trigger lazy-loaded content.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. Run:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        logger.info("[PlaywrightScraper] Fetching %s", url)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=self._cfg.headless,
                slow_mo=self._cfg.slow_mo_ms,
            )
            context = browser.new_context(
                user_agent=self._cfg.user_agent,
                extra_http_headers=self._cfg.extra_headers,
                viewport={"width": 1280, "height": 900},
            )
            page = context.new_page()

            try:
                page.goto(url, timeout=self._cfg.timeout_ms, wait_until="domcontentloaded")

                selector = wait_selector or self._cfg.wait_for_selector
                if selector:
                    page.wait_for_selector(selector, timeout=self._cfg.timeout_ms)
                else:
                    page.wait_for_load_state("networkidle", timeout=15_000)

                if scroll:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1500)

                title = page.title()
                body_text = page.inner_text("body") if self._can_select(page, "body") else ""
                html_content = page.content()

            finally:
                browser.close()

        tables_md: List[str] = []
        if extract_tables:
            tables_md = _extract_tables_from_html(html_content)

        content_parts = [_clean_text(body_text)]
        if tables_md:
            content_parts.append("\n\n### Extracted Tables\n" + "\n\n".join(tables_md))

        result = SearchResult(
            title=title,
            url=url,
            content="\n\n".join(content_parts),
            score=1.0,
        )

        logger.info(
            "[PlaywrightScraper] Done | title=%r tables=%d", title, len(tables_md)
        )
        return SearchResponse(query=url, results=[result])

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _can_select(page, selector: str) -> bool:
        try:
            return page.query_selector(selector) is not None
        except Exception:
            return False


# ── HTML table extractor ──────────────────────────────────────────────────────

def _extract_tables_from_html(html: str) -> List[str]:
    """Parse all <table> elements and convert to markdown."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; table extraction skipped")
        return []

    soup = BeautifulSoup(html, "html.parser")
    tables_md: List[str] = []

    for idx, table in enumerate(soup.find_all("table"), start=1):
        md = _table_to_markdown(table)
        if md:
            tables_md.append(f"**Table {idx}**\n{md}")

    return tables_md


def _table_to_markdown(table) -> str:
    """Convert a BeautifulSoup <table> element to a markdown table string."""
    rows = table.find_all("tr")
    if not rows:
        return ""

    md_rows: List[List[str]] = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        md_rows.append([_clean_cell(c.get_text()) for c in cells])

    if not md_rows:
        return ""

    # Normalise column count
    max_cols = max(len(r) for r in md_rows)
    md_rows = [r + [""] * (max_cols - len(r)) for r in md_rows]

    header = "| " + " | ".join(md_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in md_rows[1:])

    return "\n".join(filter(None, [header, separator, body]))


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().replace("|", "\\|")


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
