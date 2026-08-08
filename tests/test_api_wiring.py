"""
HTTP-layer tests for api/main.py: router wiring and auth gating — using the
real app with the DB dependency overridden (no Postgres required) and
TestClient used *without* a context manager so the lifespan's init_db()
never runs (verified separately: TestClient(app) without `with` skips
startup/shutdown events).
"""
import os


def test_health_reachable_without_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_protected_route_reachable_without_api_key_by_default(client):
    r = client.get("/api/v1/research")
    assert r.status_code == 200


def test_protected_routes_require_api_key_when_configured(client):
    os.environ["API_KEY"] = "secret123"

    unauthenticated = client.get("/api/v1/research")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/api/v1/research", headers={"X-API-Key": "secret123"})
    assert authenticated.status_code == 200

    wrong_key = client.get("/api/v1/research", headers={"X-API-Key": "wrong"})
    assert wrong_key.status_code == 401


def test_health_exempt_even_when_api_key_configured(client):
    os.environ["API_KEY"] = "secret123"
    r = client.get("/health")
    assert r.status_code == 200


def test_conversations_route_also_gated_by_api_key(client):
    os.environ["API_KEY"] = "secret123"
    assert client.get("/api/v1/conversations").status_code == 401
    assert client.get("/api/v1/conversations", headers={"X-API-Key": "secret123"}).status_code == 200
