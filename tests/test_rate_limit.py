from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.rate_limit import RateLimitMiddleware


def make_app(requests_per_window: int, window_seconds: int = 60) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )

    @app.get("/thing")
    def get_thing():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    return app


def test_requests_within_limit_succeed():
    client = TestClient(make_app(requests_per_window=3))
    for _ in range(3):
        assert client.get("/thing").status_code == 200


def test_requests_beyond_limit_are_rejected():
    client = TestClient(make_app(requests_per_window=2))
    assert client.get("/thing").status_code == 200
    assert client.get("/thing").status_code == 200
    third = client.get("/thing")
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_health_endpoint_is_exempt_from_rate_limiting():
    client = TestClient(make_app(requests_per_window=1))
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200


def test_zero_limit_disables_rate_limiting():
    client = TestClient(make_app(requests_per_window=0))
    for _ in range(10):
        assert client.get("/thing").status_code == 200
