
from __future__ import annotations

import re
from typing import Callable, List
from urllib.parse import urlparse

from agents.state import ResearchState
from tools.playwright_scraper import PlaywrightScraperTool, ScraperConfig
from utils.logger import get_logger
from utils.table_extractor import html_tables_to_markdown
from utils.domain_filter import should_skip_scraping

logger = get_logger(__name__)

NodeFn = Callable[[ResearchState], ResearchState]

JS_HEAVY_DOMAINS = {
    "nseindia.com",
    "bseindia.com",
    "moneycontrol.com",
    "screener.in",
    "tradingview.com",
    "finance.yahoo.com",
    "investing.com",
    "economictimes.indiatimes.com",
    "reuters.com",
    "bloomberg.com",
    "nature.com",
    "sciencedirect.com",
    "hbr.org",
}

# Only scrape URLs matching these patterns (avoid scraping every result)
SCRAPE_URL_PATTERNS = [
    r"nseindia\.com",
    r"bseindia\.com",
    r"moneycontrol\.com",
    r"screener\.in",
    r"finance\.yahoo\.com",
    r"reuters\.com/article",
    r"reuters\.com/business",
    r"bloomberg\.com/news",
    r"nature\.com/articles",
    r"sciencedirect\.com/science/article",
    r"hbr\.org/20[0-9]{2}/",
    r"\.gov/",
    r"\.edu/",
]


def _should_scrape(url: str) -> bool:
    # Skip excluded domains entirely
    if should_skip_scraping(url):
        return False
    
    domain = urlparse(url).netloc.lower().replace("www.", "")
    if domain in JS_HEAVY_DOMAINS:
        return True
    return any(re.search(pat, url) for pat in SCRAPE_URL_PATTERNS)


def make_scraper_node(
    scraper: PlaywrightScraperTool | None = None,
    max_urls: int = 3,
) -> NodeFn:
    """
    Factory for the scraper node.

    Parameters
    ----------
    scraper:    Pre-built PlaywrightScraperTool. Created with defaults if None.
    max_urls:   Maximum URLs to deep-scrape per pass (keeps runtime bounded).
    """
    _scraper = scraper or PlaywrightScraperTool()

    def scraper_node(state: ResearchState) -> ResearchState:
        # Initialize scraped_urls in state if it doesn't exist
        if not hasattr(state, 'processed_urls'):
             state.processed_urls = []
        
        # We'll use a local set of already scraped URLs for this session if not persisted in state
        # Actually, processed_urls from state is better.
        
        # Collect all URLs from previous search responses
        candidate_urls: List[str] = []
        for resp in state.search_responses:
            for result in resp.results:
                if result.url and _should_scrape(result.url) and result.url not in state.processed_urls:
                    candidate_urls.append(result.url)

        # Deduplicate while preserving order
        seen = set()
        unique_urls = [u for u in candidate_urls if not (u in seen or seen.add(u))]
        urls_to_scrape = unique_urls[:max_urls]

        if not urls_to_scrape:
            logger.info("[ScraperNode] No new JS-heavy URLs found; skipping")
            return state

        logger.info("[ScraperNode] Deep-scraping %d new URL(s): %s", len(urls_to_scrape), urls_to_scrape)

        enriched_chunks: List[str] = []
        for url in urls_to_scrape:
            try:
                response = _scraper.fetch_url(url, extract_tables=True, scroll=True)
                # Mark as processed immediately so we don't try again even if it fails partially
                if url not in state.processed_urls:
                    state.processed_urls.append(url)

                for r in response.results:
                    if r.content.strip():
                        enriched_chunks.append(
                            f"### Deep-scraped: {r.title}\nURL: {url}\n\n{r.content}"
                        )
                    else:
                        # Content was empty - likely a scrape failure or paywall
                        domain = urlparse(url).netloc.lower().replace("www.", "")
                        if domain not in state.failed_domains:
                            state.failed_domains.append(domain)
                            logger.info("[ScraperNode] Tracking failed domain: %s", domain)
            except Exception as exc:
                logger.warning("[ScraperNode] Failed to scrape %s: %s", url, exc)
                domain = urlparse(url).netloc.lower().replace("www.", "")
                if domain not in state.failed_domains:
                    state.failed_domains.append(domain)
                    logger.info("[ScraperNode] Tracking failed domain: %s", domain)

        if enriched_chunks:
            state.context.append(
                "## Deep-Scraped Page Content\n\n"
                + "\n\n---\n\n".join(enriched_chunks)
            )
            state.has_new_data = True

        return state

    return scraper_node
