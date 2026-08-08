from types import SimpleNamespace

from utils.pdf_generator import generate_pdf_from_state


def make_state_like(**overrides) -> SimpleNamespace:
    defaults = dict(
        query="What is the capital of France?",
        understood_intent="What is the capital of France?",
        final_answer="Paris is the capital of France.\n\n- Population is about 2.1 million\n- Known for the Eiffel Tower",
        confidence_score=0.85,
        data_quality_score=0.9,
        hallucination_flagged=False,
        iteration=2,
        validation_results={"results": [{"validation_type": "completeness", "passed": True}]},
        scored_sources=[{"url": "https://en.wikipedia.org/wiki/Paris", "title": "Paris", "overall_score": 0.8}],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_generate_pdf_from_state_produces_valid_pdf():
    pdf_bytes = generate_pdf_from_state(make_state_like())
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500


def test_generate_pdf_from_state_handles_missing_optional_fields():
    # Mirrors what the /research/{id}/pdf API route builds for a bare-bones research row:
    # no sources, no validations, empty understood_intent.
    state_like = make_state_like(scored_sources=[], validation_results={"results": []}, understood_intent="")
    pdf_bytes = generate_pdf_from_state(state_like)
    assert pdf_bytes[:4] == b"%PDF"
