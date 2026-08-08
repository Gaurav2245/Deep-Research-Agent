from agents.scraper_node import make_scraper_node
from agents.state import ResearchState
from tools.base import SearchResponse, SearchResult


class FakeScraper:
    """Deterministic stand-in for PlaywrightScraperTool.fetch_url."""

    def __init__(self, outcomes: dict):
        self._outcomes = outcomes  # url -> SearchResponse or Exception

    def fetch_url(self, url: str, extract_tables: bool = True, scroll: bool = True) -> SearchResponse:
        outcome = self._outcomes[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_state_with_results(urls: list[str]) -> ResearchState:
    state = ResearchState(query="root question")
    state.search_responses = [
        SearchResponse(
            query="q",
            results=[SearchResult(title="T", url=url, content="c") for url in urls],
        )
    ]
    return state


def test_scraper_node_skips_when_no_scrapeable_urls():
    state = make_state_with_results(["https://example.com/plain-article"])
    node = make_scraper_node(scraper=FakeScraper({}), max_urls=3)
    result = node(state)
    assert result.processed_urls == []


def test_scraper_node_marks_success_as_processed():
    url = "https://nseindia.com/report"
    outcome = SearchResponse(query=url, results=[SearchResult(title="R", url=url, content="Deep content")])
    node = make_scraper_node(scraper=FakeScraper({url: outcome}), max_urls=3)
    state = make_state_with_results([url])

    result = node(state)

    assert url in result.processed_urls
    assert result.has_new_data is True
    assert any("Deep content" in c for c in result.context)


def test_scraper_node_does_not_mark_failed_url_as_processed():
    url = "https://nseindia.com/report"
    node = make_scraper_node(scraper=FakeScraper({url: RuntimeError("boom")}), max_urls=3)
    state = make_state_with_results([url])

    result = node(state)

    assert url not in result.processed_urls
    assert "nseindia.com" in result.failed_domains


def test_scraper_node_scrapes_multiple_urls_independently():
    url_ok = "https://nseindia.com/ok"
    url_fail = "https://bseindia.com/fail"
    outcomes = {
        url_ok: SearchResponse(query=url_ok, results=[SearchResult(title="OK", url=url_ok, content="Good data")]),
        url_fail: RuntimeError("timeout"),
    }
    node = make_scraper_node(scraper=FakeScraper(outcomes), max_urls=3)
    state = make_state_with_results([url_ok, url_fail])

    result = node(state)

    assert url_ok in result.processed_urls
    assert url_fail not in result.processed_urls
    assert "bseindia.com" in result.failed_domains
