"""Shared fixtures for API route tests: a fake DB session/query that avoids
needing a real Postgres connection, configurable per-model per test.
"""
from __future__ import annotations

import os
import uuid as _uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from api.main import app
from database import get_db


class FakeQuery:
    """Minimal stand-in for a SQLAlchemy Query.

    Filtering is not actually evaluated (comparisons like `Model.id == x`
    can't be interpreted without a real engine) — tests instead control
    the result set directly via `session.data[Model]`, and typically keep
    exactly zero or one relevant row per test.
    """

    def __init__(self, session: "FakeSession", model):
        self.session = session
        self.model = model

    def filter(self, *args, **kwargs):
        return self

    def filter_by(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def offset(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def _rows(self):
        return self.session.data.get(self.model, [])

    def count(self):
        return len(self._rows())

    def all(self):
        return list(self._rows())

    def first(self):
        rows = self._rows()
        return rows[0] if rows else None

    def get(self, id_):
        for row in self._rows():
            if getattr(row, "id", None) == id_:
                return row
        return None


class FakeSession:
    def __init__(self):
        self.data: dict = {}
        self.added = []
        self.deleted = []
        self.committed = False

    def query(self, model):
        return FakeQuery(self, model)

    def add(self, obj):
        # A real flush would generate a PK and apply column-level `default=`
        # values (e.g. message_count=0, created_at=datetime.utcnow); simulate
        # that generically here since FakeSession never actually flushes to
        # an engine.
        if getattr(obj, "id", None) is None:
            obj.id = _uuid.uuid4()

        for column in sa_inspect(type(obj)).columns:
            if getattr(obj, column.key, None) is not None:
                continue
            default = column.default
            if default is None:
                continue
            if default.is_scalar:
                setattr(obj, column.key, default.arg)
            elif default.is_callable:
                try:
                    setattr(obj, column.key, default.arg(None))
                except TypeError:
                    setattr(obj, column.key, default.arg())

        self.added.append(obj)
        self.data.setdefault(type(obj), [])
        if obj not in self.data[type(obj)]:
            self.data[type(obj)].append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        rows = self.data.get(type(obj), [])
        if obj in rows:
            rows.remove(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def client(fake_session):
    def fake_get_db():
        yield fake_session

    app.dependency_overrides[get_db] = fake_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    os.environ.pop("API_KEY", None)
