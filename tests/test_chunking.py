"""Unit tests for chunking — pure functions, no DB or network."""

from __future__ import annotations

import pytest

from app.rag import chunk_text


def test_short_text_is_one_chunk():
    text = "Meridian Pay offers 20 days of PTO."
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert chunks == [text]


def test_empty_text_yields_no_chunks():
    assert chunk_text("", chunk_size=800, overlap=120) == []
    assert chunk_text("   \n  ", chunk_size=800, overlap=120) == []


def test_chunks_respect_size_limit():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_no_word_is_split_across_chunks():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    original_words = text.split()
    for c in chunks:
        for w in c.split():
            assert w in original_words  # whole words only


def test_overlap_carries_context_between_chunks():
    text = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_text(text, chunk_size=40, overlap=15)
    assert len(chunks) >= 2
    # Consecutive chunks should share at least one word (the overlap tail).
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split()), "expected overlapping words"


def test_reassembled_chunks_cover_all_words_in_order():
    text = " ".join(f"t{i}" for i in range(45))
    chunks = chunk_text(text, chunk_size=30, overlap=8)
    # Every original word appears in some chunk, first chunk starts at the start.
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())
    assert chunks[0].startswith("t0")


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_chunk_size_raises(bad):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=bad, overlap=0)


@pytest.mark.parametrize("overlap", [-1, 200, 300])
def test_invalid_overlap_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=200, overlap=overlap)
