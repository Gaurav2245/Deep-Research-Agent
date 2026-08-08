import pytest

from database.confidence_scorer import ConfidenceScorer


def test_score_source_diversity_empty_is_zero():
    assert ConfidenceScorer.score_source_diversity([]) == 0.0


def test_score_source_diversity_all_same_domain_is_low():
    sources = [{"url": "https://a.com/1"}, {"url": "https://a.com/2"}, {"url": "https://a.com/3"}]
    diversity = ConfidenceScorer.score_source_diversity(sources)
    assert diversity == pytest.approx(1 / 3)


def test_score_source_diversity_all_unique_domains_is_full():
    sources = [{"url": "https://a.com"}, {"url": "https://b.com"}, {"url": "https://c.com"}]
    assert ConfidenceScorer.score_source_diversity(sources) == 1.0


def test_score_source_quality_empty_is_zero():
    assert ConfidenceScorer.score_source_quality([]) == 0.0


def test_score_source_quality_averages_overall_scores():
    sources = [{"overall_score": 0.5}, {"overall_score": 1.0}]
    assert ConfidenceScorer.score_source_quality(sources) == 0.75


def test_score_data_consistency_defaults_with_fewer_than_two_embeddings():
    assert ConfidenceScorer.score_data_consistency([]) == 0.5
    assert ConfidenceScorer.score_data_consistency([[0.1, 0.2]]) == 0.5


def test_calculate_information_gain_defaults_when_missing_embeddings():
    assert ConfidenceScorer.calculate_information_gain([], []) == 1.0
    assert ConfidenceScorer.calculate_information_gain([[0.1]], []) == 1.0


def test_score_answer_completeness_empty_answer_is_zero():
    assert ConfidenceScorer.score_answer_completeness("", "query") == 0.0


def test_score_answer_completeness_short_answer_is_low():
    assert ConfidenceScorer.score_answer_completeness("Too short.", "query") == 0.3


def test_score_no_hallucination_no_sources_or_answer_is_low():
    assert ConfidenceScorer.score_no_hallucination("", []) == 0.3
    assert ConfidenceScorer.score_no_hallucination("answer", []) == 0.3


def test_score_no_hallucination_deducts_for_flagged_issues():
    base = ConfidenceScorer.score_no_hallucination("answer", [{"content": "x"}], [])
    with_issue = ConfidenceScorer.score_no_hallucination(
        "answer", [{"content": "x"}], [{"validation_type": "hallucination"}]
    )
    assert with_issue < base


def test_should_continue_research_stops_at_max_iterations():
    should_continue, reason = ConfidenceScorer.should_continue_research(
        overall_confidence=0.1, iterations=5, max_iterations=5
    )
    assert should_continue is False
    assert "maximum iterations" in reason.lower()


def test_should_continue_research_stops_when_confidence_sufficient():
    should_continue, _ = ConfidenceScorer.should_continue_research(
        overall_confidence=0.9, min_confidence=0.7, iterations=1
    )
    assert should_continue is False


def test_should_continue_research_continues_with_pending_follow_ups_and_low_confidence():
    should_continue, _ = ConfidenceScorer.should_continue_research(
        overall_confidence=0.3, min_confidence=0.7, iterations=1, has_follow_ups=True
    )
    assert should_continue is True


def test_should_continue_research_stops_low_confidence_no_pending_queries():
    should_continue, reason = ConfidenceScorer.should_continue_research(
        overall_confidence=0.3, min_confidence=0.7, iterations=1, has_follow_ups=False
    )
    assert should_continue is False
    assert "no pending" in reason.lower()
