"""
Regression tests for api/routes/validation.py: these endpoints used to crash
with TypeError ("missing 1 required positional argument: 'self'") because
DataValidator's methods were called as unbound class methods. Confirms the
fix holds at the full HTTP layer, not just the unit level.
"""
import uuid

from database import Research, Source


def make_research(final_answer: str | None = "Paris is the capital of France. " * 30) -> Research:
    research = Research(
        id=uuid.uuid4(),
        query="What is the capital of France?",
        final_answer=final_answer,
    )
    research.sources = [
        Source(
            id=uuid.uuid4(),
            url="https://en.wikipedia.org/wiki/Paris",
            title="Paris",
            content="Paris is the capital of France.",
        )
        for _ in range(3)
    ]
    return research


def test_quality_endpoint_no_longer_crashes(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/quality")

    assert r.status_code == 200
    body = r.json()
    assert "overall_quality_score" in body
    assert isinstance(body["validation_results"], list)


def test_quality_endpoint_404_when_research_missing(client, fake_session):
    fake_session.data[Research] = []
    r = client.get(f"/api/v1/research/{uuid.uuid4()}/quality")
    assert r.status_code == 404


def test_quality_endpoint_400_when_not_complete(client, fake_session):
    research = make_research(final_answer=None)
    fake_session.data[Research] = [research]
    r = client.get(f"/api/v1/research/{research.id}/quality")
    assert r.status_code == 400


def test_validate_completeness_endpoint(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.post(f"/api/v1/validate/completeness?research_id={research.id}")

    assert r.status_code == 200
    assert r.json()["validation_type"] == "completeness"


def test_validate_consistency_endpoint(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.post(f"/api/v1/validate/consistency?research_id={research.id}")

    assert r.status_code == 200
    assert r.json()["validation_type"] == "consistency"


def test_validate_hallucination_endpoint(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.post(f"/api/v1/validate/hallucination?research_id={research.id}")

    assert r.status_code == 200
    assert r.json()["validation_type"] == "hallucination"


def test_validate_factual_claims_endpoint(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.post(f"/api/v1/validate/factual-claims?research_id={research.id}")

    assert r.status_code == 200
    assert r.json()["validation_type"] == "factual_claims"
