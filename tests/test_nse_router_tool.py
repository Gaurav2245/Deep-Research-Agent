from tools.base import BaseSearchTool, SearchResponse, SearchResult
from tools.nse_router_tool import AutoRoutingSearchTool


class FakeTool(BaseSearchTool):
    def __init__(self, name: str, responses: dict | None = None, error: Exception | None = None):
        self._name = name
        self._responses = responses or {}
        self._error = error
        self.calls = []

    def provider_name(self) -> str:
        return self._name

    def search(self, query: str, **kwargs) -> SearchResponse:
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return self._responses.get(
            query,
            SearchResponse(query=query, results=[SearchResult(title="T", url="https://x.com", content="c")]),
        )


def test_non_market_query_uses_default_tool():
    default = FakeTool("Default")
    nse = FakeTool("NSE")
    router = AutoRoutingSearchTool(default_tool=default, nse_tool=nse)

    router.search("RBI monetary policy decisions")

    assert default.calls == ["RBI monetary policy decisions"]
    assert nse.calls == []


def test_market_query_routes_to_nse():
    default = FakeTool("Default")
    nse = FakeTool("NSE", responses={
        "gainers": SearchResponse(query="gainers", results=[SearchResult(title="Gainer", url="https://nse.com/g", content="")]),
    })
    router = AutoRoutingSearchTool(default_tool=default, nse_tool=nse)

    result = router.search("Nifty 50 top gainers today NSE")

    assert nse.calls == ["gainers"]
    assert default.calls == []
    assert [r.url for r in result.results] == ["https://nse.com/g"]


def test_combined_gainers_and_losers_merges_results():
    default = FakeTool("Default")
    nse = FakeTool("NSE", responses={
        "gainers": SearchResponse(query="gainers", results=[SearchResult(title="G", url="https://nse.com/g", content="")], answer="gainers answer"),
        "losers": SearchResponse(query="losers", results=[SearchResult(title="L", url="https://nse.com/l", content="")], answer="losers answer"),
    })
    router = AutoRoutingSearchTool(default_tool=default, nse_tool=nse)

    result = router.search("nifty 50 top 10 gainers and losers as of today")

    assert set(nse.calls) == {"gainers", "losers"}
    urls = {r.url for r in result.results}
    assert urls == {"https://nse.com/g", "https://nse.com/l"}
    assert "gainers answer" in result.answer
    assert "losers answer" in result.answer


def test_falls_back_to_default_tool_when_nse_fails():
    default = FakeTool("Default")
    nse = FakeTool("NSE", error=RuntimeError("NSE cookies expired"))
    router = AutoRoutingSearchTool(default_tool=default, nse_tool=nse)

    router.search("sensex today")

    assert nse.calls == ["sensex"]
    assert default.calls == ["sensex today"]


def test_provider_name_reflects_composition():
    router = AutoRoutingSearchTool(default_tool=FakeTool("Tavily"), nse_tool=FakeTool("NSE"))
    assert router.provider_name() == "AutoRouting(Tavily + NSE)"
