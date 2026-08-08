import uuid
from datetime import datetime

from database import Research, Source


def make_research(**overrides) -> Research:
    defaults = dict(
        id=uuid.uuid4(),
        query="What is the capital of France?",
        final_answer="Paris is the capital of France.",
        confidence_score=0.8,
        data_quality_score=0.9,
        research_complete=True,
        total_iterations=2,
        hallucination_flagged=False,
        follow_up_questions=[],
        understood_intent="What is the capital of France?",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    defaults.update(overrides)
    research = Research(**defaults)
    research.sources = []
    research.validations = []
    return research


def test_start_research_returns_immediately_and_schedules_background_task(client, fake_session, monkeypatch):
    calls = []

    def fake_run_research(query, research_id=None, max_iterations=None):
        calls.append((query, research_id, max_iterations))

    monkeypatch.setattr("main.run_research", fake_run_research)

    r = client.post("/api/v1/research", json={"query": "What is the capital of France?", "depth": "quick"})

    assert r.status_code == 200
    body = r.json()
    assert body["research_complete"] is False
    assert body["final_answer"] is None
    # Background task ran synchronously under TestClient and passed depth through correctly.
    assert len(calls) == 1
    query, research_id, max_iterations = calls[0]
    assert query == "What is the capital of France?"
    assert max_iterations == 1  # "quick" -> 1 iteration


def test_get_research_status_found(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}")

    assert r.status_code == 200
    assert r.json()["final_answer"] == "Paris is the capital of France."


def test_get_research_status_not_found(client, fake_session):
    fake_session.data[Research] = []
    r = client.get(f"/api/v1/research/{uuid.uuid4()}")
    assert r.status_code == 404


def test_get_research_detail(client, fake_session):
    research = make_research()
    research.sources = [
        Source(
            id=uuid.uuid4(), url="https://en.wikipedia.org/wiki/Paris", title="Paris",
            content="Paris is the capital.", source_score=0.9, relevance_score=0.9,
            authority_score=0.9, recency_score=0.9, content_quality=0.9,
            is_primary_source=True, is_verified=False,
        )
    ]
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/detail")

    assert r.status_code == 200
    body = r.json()
    assert len(body["sources"]) == 1
    assert body["sources"][0]["url"].startswith("https://en.wikipedia.org")


def test_get_research_pdf_returns_pdf_bytes(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/pdf")

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"


def test_get_research_pdf_400_when_incomplete(client, fake_session):
    research = make_research(final_answer=None)
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/pdf")
    assert r.status_code == 400


def test_get_confidence_score(client, fake_session):
    research = make_research()
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/confidence")

    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["overall_confidence"] <= 1.0


def test_list_research_sessions(client, fake_session):
    fake_session.data[Research] = [make_research(), make_research()]

    r = client.get("/api/v1/research")

    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["sessions"]) == 2
