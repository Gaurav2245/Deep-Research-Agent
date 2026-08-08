from agents.nodes import make_web_search_node
from agents.state import ResearchState
from tools.base import BaseSearchTool, SearchResponse, SearchResult


class FakeSearchTool(BaseSearchTool):
    """Deterministic search tool for testing: maps query -> SearchResponse or exception."""

    def __init__(self, responses: dict):
        self._responses = responses

    def provider_name(self) -> str:
        return "Fake"

    def search(self, query: str, **kwargs) -> SearchResponse:
        result = self._responses[query]
        if isinstance(result, Exception):
            raise result
        return result


def make_state(queries, **overrides) -> ResearchState:
    state = ResearchState(query="root question")
    state.search_queries = queries
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_web_search_node_no_queries_short_circuits():
    node = make_web_search_node(FakeSearchTool({}))
    state = make_state([])
    result = node(state)
    assert result.search_responses == []
    assert result.has_new_data is False


def test_web_search_node_aggregates_results_from_all_queries():
    tool = FakeSearchTool(
        {
            "q1": SearchResponse(query="q1", results=[SearchResult(title="A", url="https://a.com", content="a")]),
            "q2": SearchResponse(query="q2", results=[SearchResult(title="B", url="https://b.com", content="b")]),
        }
    )
    node = make_web_search_node(tool)
    state = make_state(["q1", "q2"])
    result = node(state)

    urls = {r.url for resp in result.search_responses for r in resp.results}
    assert urls == {"https://a.com", "https://b.com"}
    assert result.has_new_data is True
    assert result.iteration == 1


def test_web_search_node_filters_already_processed_urls():
    tool = FakeSearchTool(
        {
            "q1": SearchResponse(query="q1", results=[SearchResult(title="A", url="https://a.com", content="a")]),
        }
    )
    node = make_web_search_node(tool)
    state = make_state(["q1"], processed_urls=["https://a.com"])
    result = node(state)

    assert result.search_responses == []
    assert result.has_new_data is False


def test_web_search_node_survives_individual_query_failures():
    tool = FakeSearchTool(
        {
            "good": SearchResponse(query="good", results=[SearchResult(title="A", url="https://a.com", content="a")]),
            "bad": RuntimeError("provider exploded"),
        }
    )
    node = make_web_search_node(tool)
    state = make_state(["good", "bad"])
    result = node(state)

    urls = {r.url for resp in result.search_responses for r in resp.results}
    assert urls == {"https://a.com"}
