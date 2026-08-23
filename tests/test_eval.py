"""RAG eval gate (Phase 4).

Runs the frozen question set through the real pipeline and asserts the aggregate
faithfulness / relevance / retrieval scores clear their thresholds. This is the
test that becomes the CI merge gate in Phase 6.

Needs Azure OpenAI (embeddings + chat) and a populated vector store; skips
cleanly if either is unavailable so the pure unit tests can still run anywhere.
"""

from __future__ import annotations

import pytest

from app import rag
from eval.run_eval import THRESHOLDS, evaluate, load_questions


@pytest.fixture(scope="module")
def report(settings, db_available, azure_configured):
    if not db_available:
        pytest.skip("PostgreSQL + pgvector not available")
    if not azure_configured:
        pytest.skip("Azure OpenAI not configured (.env)")
    client = rag.get_client(settings)
    if rag.count_chunks(settings) == 0:
        rag.ingest(settings, client)
    return evaluate(settings, client)


@pytest.mark.eval
def test_eval_set_is_frozen_and_well_formed():
    questions = load_questions()
    assert len(questions) >= 20
    for q in questions:
        assert {"question", "expected_answer", "source_doc"} <= q.keys()


@pytest.mark.eval
def test_retrieval_hit_rate_clears_threshold(report):
    m = report.metrics
    assert m["retrieval_hit_rate"] >= THRESHOLDS["retrieval_hit_rate"], m


@pytest.mark.eval
def test_faithfulness_clears_threshold(report):
    m = report.metrics
    assert m["faithfulness"] >= THRESHOLDS["faithfulness"], m


@pytest.mark.eval
def test_relevance_clears_threshold(report):
    m = report.metrics
    assert m["relevance"] >= THRESHOLDS["relevance"], m


@pytest.mark.eval
def test_overall_gate_passes(report):
    assert report.passed(), report.metrics
