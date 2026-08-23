"""Unit tests for retrieval.

These exercise the real `rag.retrieve()` (pgvector `<=>` nearest-neighbor +
top-k) against an isolated table seeded with hand-crafted vectors. Query
embeddings are faked deterministically, so no LLM API is needed. Requires a
running PostgreSQL + pgvector.
"""

from __future__ import annotations

import pytest

from app import rag
from pgvector import Vector

# Deterministic 3-d "embeddings": each topic points along its own axis, so we
# know exactly which rows should rank nearest for a given query vector.
SEED_ROWS = [
    ("pto-policy.md", 0, "Full-time employees get 20 days of PTO.", [1.0, 0.0, 0.0]),
    ("pto-policy.md", 1, "Sick leave is 10 days, separate from PTO.", [0.9, 0.1, 0.0]),
    ("security-policy.md", 0, "Passwords must be 14 characters.", [0.0, 1.0, 0.0]),
    ("pricing.md", 0, "The Growth plan costs 49 dollars.", [0.0, 0.0, 1.0]),
]

# Keyword -> query vector, so a query string maps to a known direction.
_FAKE_VECTORS = {
    "pto": [1.0, 0.0, 0.0],
    "password": [0.0, 1.0, 0.0],
    "price": [0.0, 0.0, 1.0],
}


def _fake_embed(client, deployment, texts):
    out = []
    for t in texts:
        low = t.lower()
        vec = next((v for kw, v in _FAKE_VECTORS.items() if kw in low), [0.0, 0.0, 0.0])
        out.append(vec)
    return out


@pytest.fixture
def seeded_store(settings, db_available, monkeypatch):
    if not db_available:
        pytest.skip("PostgreSQL + pgvector not available")

    # Point the store at an isolated table and fake the query embedding.
    monkeypatch.setattr(rag, "CHUNKS_TABLE", "chunks_test")
    monkeypatch.setattr(rag, "embed_texts", _fake_embed)

    with rag.connect(settings) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks_test")
        conn.execute(
            "CREATE TABLE chunks_test (id serial PRIMARY KEY, doc text, "
            "chunk_index int, content text, embedding vector(3))"
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks_test (doc, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [(doc, idx, content, Vector(emb)) for doc, idx, content, emb in SEED_ROWS],
            )
        conn.commit()
    yield settings
    with rag.connect(settings) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks_test")
        conn.commit()


@pytest.mark.db
def test_retrieve_returns_k_results(seeded_store):
    results = rag.retrieve(seeded_store, client=None, query="pto", k=2)
    assert len(results) == 2


@pytest.mark.db
def test_retrieve_ranks_nearest_first(seeded_store):
    # "pto" query points along axis 0 -> the two pto-policy rows are nearest.
    results = rag.retrieve(seeded_store, client=None, query="pto days", k=3)
    assert results[0].doc == "pto-policy.md"
    assert results[0].chunk_index == 0  # exact-match row is closest
    assert results[1].doc == "pto-policy.md"
    # distances come back sorted ascending (nearest first)
    assert results[0].distance <= results[1].distance <= results[2].distance


@pytest.mark.db
def test_retrieve_finds_correct_topic(seeded_store):
    assert rag.retrieve(seeded_store, None, "password", k=1)[0].doc == "security-policy.md"
    assert rag.retrieve(seeded_store, None, "price", k=1)[0].doc == "pricing.md"


@pytest.mark.db
def test_retrieve_respects_k_larger_than_corpus(seeded_store):
    results = rag.retrieve(seeded_store, None, "pto", k=99)
    assert len(results) == len(SEED_ROWS)
