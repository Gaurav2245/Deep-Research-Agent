from agents._shared import format_context, parse_json_list
from tools.base import SearchResponse, SearchResult


def test_parse_json_list_valid_array():
    assert parse_json_list('["a", "b", "c"]') == ["a", "b", "c"]


def test_parse_json_list_strips_markdown_fences():
    text = '```json\n["x", "y"]\n```'
    assert parse_json_list(text) == ["x", "y"]


def test_parse_json_list_drops_blank_entries():
    assert parse_json_list('["a", "", "  ", "b"]') == ["a", "b"]


def test_parse_json_list_invalid_json_returns_empty():
    assert parse_json_list("not json at all") == []


def test_parse_json_list_non_list_json_returns_empty():
    assert parse_json_list('{"a": 1}') == []


def test_format_context_joins_results_and_answer():
    response = SearchResponse(
        query="q",
        results=[SearchResult(title="Title", url="https://x.com", content="Body text")],
        answer="Direct answer here",
    )
    formatted = format_context([response])
    assert "Title" in formatted
    assert "https://x.com" in formatted
    assert "Body text" in formatted
    assert "Direct answer here" in formatted


def test_format_context_empty_list():
    assert format_context([]) == ""
