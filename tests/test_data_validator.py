import pytest

from database.data_validator import DataValidator


@pytest.fixture
def validator():
    return DataValidator()


def test_validate_completeness_flags_short_answer(validator):
    result = validator.validate_completeness(
        final_answer="Too short.", sources=[{"content": "x"}] * 3, query="What happened?"
    )
    assert result["passed"] is False
    assert any("too short" in issue.lower() for issue in result["issues"])


def test_validate_completeness_flags_insufficient_sources(validator):
    long_answer = " ".join(["word"] * 150)
    result = validator.validate_completeness(final_answer=long_answer, sources=[], query="q")
    assert result["passed"] is False
    assert any("source" in issue.lower() for issue in result["issues"])


def test_validate_completeness_passes_with_substantial_answer_and_sources(validator):
    long_answer = " ".join(["word"] * 150)
    sources = [{"content": "x"}] * 3
    result = validator.validate_completeness(final_answer=long_answer, sources=sources, query="q")
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_validate_consistency_no_claims_passes(validator):
    result = validator.validate_consistency(sources=[], final_answer="Nothing to see here")
    assert result["passed"] is True


def test_validate_consistency_flags_increasing_and_decreasing(validator):
    result = validator.validate_consistency(
        sources=[], final_answer="Prices are increasing while demand is decreasing rapidly."
    )
    assert result["passed"] is False
    assert result["conflicts"]


def test_detect_hallucination_markers_flags_unsourced_quote(validator):
    answer = 'The report stated "this is a very specific and unverifiable quotation right here".'
    result = validator.detect_hallucination_markers(final_answer=answer, sources=[{"content": "unrelated"}])
    assert result["hallucination_flagged"] is True


def test_detect_hallucination_markers_no_issues_with_sourced_content(validator):
    answer = "The market closed higher today."
    sources = [{"content": "The market closed higher today."}]
    result = validator.detect_hallucination_markers(final_answer=answer, sources=sources)
    assert result["passed"] is True
    assert result["hallucination_flagged"] is False
