import pytest

from tools.base import BaseSearchTool, SearchResponse, SearchResult
from tools.fallback_tool import FallbackSearchTool


class FakeTool(BaseSearchTool):
    def __init__(self, name: str, response: SearchResponse | None = None, error: Exception | None = None):
        self._name = name
        self._response = response
        self._error = error

    def provider_name(self) -> str:
        return self._name

    def search(self, query: str, **kwargs) -> SearchResponse:
        if self._error is not None:
            raise self._error
        return self._response


def test_uses_primary_when_it_returns_results():
    primary = FakeTool("Primary", response=SearchResponse(query="q", results=[SearchResult(title="T", url="https://x.com", content="c")]))
    tool = FallbackSearchTool(primary=primary)
    tool.fallback = FakeTool("DuckDuckGo", error=RuntimeError("should not be called"))

    result = tool.search("q")
    assert result.results[0].url == "https://x.com"


def test_falls_back_when_primary_raises():
    primary = FakeTool("Primary", error=RuntimeError("primary down"))
    tool = FallbackSearchTool(primary=primary)
    tool.fallback = FakeTool("DuckDuckGo", response=SearchResponse(query="q", results=[SearchResult(title="F", url="https://ddg.com", content="c")]))

    result = tool.search("q")
    assert result.results[0].url == "https://ddg.com"


def test_falls_back_when_primary_returns_empty_results():
    primary = FakeTool("Primary", response=SearchResponse(query="q", results=[]))
    tool = FallbackSearchTool(primary=primary)
    tool.fallback = FakeTool("DuckDuckGo", response=SearchResponse(query="q", results=[SearchResult(title="F", url="https://ddg.com", content="c")]))

    result = tool.search("q")
    assert result.results[0].url == "https://ddg.com"


def test_raises_when_both_primary_and_fallback_fail():
    primary = FakeTool("Primary", error=RuntimeError("primary down"))
    tool = FallbackSearchTool(primary=primary)
    tool.fallback = FakeTool("DuckDuckGo", error=RuntimeError("fallback down too"))

    with pytest.raises(RuntimeError, match="fallback down too"):
        tool.search("q")
