import uuid
from datetime import datetime

from database import Research, Source


def make_source(**overrides) -> Source:
    defaults = dict(
        id=uuid.uuid4(),
        url="https://www.moneycontrol.com/article",
        title="Test Source",
        content="Some content here.",
        source_score=0.7,
        relevance_score=0.7,
        authority_score=1.0,
        recency_score=0.5,
        content_quality=0.6,
        is_primary_source=False,
        is_verified=False,
        discovered_at=datetime.utcnow(),
    )
    defaults.update(overrides)
    return Source(**defaults)


def make_research_with_sources(sources) -> Research:
    research = Research(
        id=uuid.uuid4(),
        query="test",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    research.sources = sources
    return research


def test_get_research_sources(client, fake_session):
    research = make_research_with_sources([make_source(), make_source(source_score=0.2)])
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/sources")

    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_research_sources_filters_by_score(client, fake_session):
    research = make_research_with_sources([make_source(source_score=0.9), make_source(source_score=0.1)])
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/sources", params={"filter_by_score": 0.5})

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["source_score"] == 0.9


def test_get_research_sources_404_when_missing(client, fake_session):
    fake_session.data[Research] = []
    r = client.get(f"/api/v1/research/{uuid.uuid4()}/sources")
    assert r.status_code == 404


def test_score_source_is_stateless(client, fake_session):
    r = client.post(
        "/api/v1/score-source",
        json={
            "url": "https://www.reuters.com/markets/article",
            "title": "Some market news",
            "content": "Detailed market analysis content here.",
            "relevance_score": 0.8,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["authority"] == 1.0  # reuters.com is high-authority


def test_get_source_analysis(client, fake_session):
    research = make_research_with_sources([make_source(), make_source()])
    fake_session.data[Research] = [research]

    r = client.get(f"/api/v1/research/{research.id}/sources/analysis")

    assert r.status_code == 200
    body = r.json()
    assert body["total_sources"] == 2


def test_mark_source_verified(client, fake_session):
    source = make_source()
    fake_session.data[Source] = [source]

    r = client.put(f"/api/v1/sources/{source.id}/verify", params={"verified": True})

    assert r.status_code == 200
    assert r.json()["is_verified"] is True
    assert source.is_verified is True
