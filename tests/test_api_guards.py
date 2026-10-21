"""API guards for the public demo: /query rate limit and /ingest admin token.

No DB and no LLM: the RAG calls are stubbed, so these run anywhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main, rag
from app.config import get_settings


@pytest.fixture
def client(monkeypatch):
    # Keep these stubbed requests out of the developer's real Langfuse project.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    monkeypatch.setattr(rag, "get_client", lambda s, instrument=False: object())
    monkeypatch.setattr(rag, "count_chunks", lambda s: 1)
    monkeypatch.setattr(rag, "answer_with_chunks", lambda s, c, q, k=None: ("ok", []))
    monkeypatch.setattr(rag, "ingest", lambda s, c: 3)
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c
    main.limiter.reset()
    get_settings.cache_clear()


def test_query_is_rate_limited_per_ip(client):
    limit = get_settings().rate_limit_per_minute
    codes = [client.post("/query", json={"question": "hi"}).status_code for _ in range(limit + 1)]
    assert codes[:limit] == [200] * limit
    assert codes[-1] == 429


def test_ingest_open_without_admin_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    assert client.post("/ingest").status_code == 200


def test_ingest_requires_admin_token_when_set(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    get_settings.cache_clear()
    assert client.post("/ingest").status_code == 403
    assert client.post("/ingest", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.post("/ingest", headers={"X-Admin-Token": "s3cret"}).status_code == 200
