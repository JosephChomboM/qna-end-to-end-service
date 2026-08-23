"""FastAPI wrapper around the core RAG loop (Phase 3).

Exposes the same pipeline the CLI uses, as HTTP endpoints:

    POST /ingest  -> (re)build the vector store from the corpus
    POST /query   -> {"question": "..."} -> {"answer": ..., "sources": [...]}
    GET  /health  -> liveness check

`/query` returns exactly what `python -m app.rag "<question>"` returns, because
both call the same `rag.answer()` with temperature 0.

Run locally:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import rag
from app.config import get_settings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    k: int | None = Field(default=None, ge=1, le=50, description="Top-k chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    documents: int
    chunks: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build settings + Azure client once per process (fails fast if config is missing).
    settings = get_settings()
    app.state.settings = settings
    app.state.client = rag.get_client(settings)
    yield


app = FastAPI(title="Meridian Pay Document Q&A", version="0.3.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    settings = app.state.settings
    client = app.state.client
    chunks = rag.ingest(settings, client)
    documents = sum(1 for _ in rag.iter_corpus())
    return IngestResponse(documents=documents, chunks=chunks)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    settings = app.state.settings
    client = app.state.client
    # Auto-ingest on first use so a fresh store still answers (same as the CLI).
    if rag.count_chunks(settings) == 0:
        rag.ingest(settings, client)
    answer_text, sources = rag.answer(settings, client, req.question, req.k)
    return QueryResponse(answer=answer_text, sources=sources)
