"""FastAPI wrapper around the core RAG loop (Phase 3).

Exposes the same pipeline the CLI uses, as HTTP endpoints:

    POST /ingest  -> (re)build the vector store from the corpus (X-Admin-Token if set)
    POST /query   -> {"question": "..."} -> {"answer": ..., "sources": [...]}
    GET  /health  -> liveness check

`/query` returns exactly what `python -m app.rag "<question>"` returns, because
both call the same `rag.answer()` with temperature 0.

Run locally:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import rag
from app.config import get_settings
from app.observability import init_langfuse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Rate limits on /query protect the free LLM quota on a public URL: one cap per
# client IP, plus one shared cap for the whole service. Limits are read once at
# import time; counters live in memory, so they reset when the process restarts.
_settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
_PER_MINUTE = f"{_settings.rate_limit_per_minute}/minute"
_PER_DAY = f"{_settings.rate_limit_per_day}/day"


def _global_key(request: Request) -> str:
    return "global"


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Gate /ingest. With ADMIN_TOKEN unset (local dev) it stays open."""
    expected = get_settings().admin_token
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Token")


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
    # Langfuse (Phase 8): init first so the instrumented client shares its config.
    app.state.langfuse = init_langfuse(settings)
    # instrument=True -> the live client emits Langfuse traces when configured.
    app.state.client = rag.get_client(settings, instrument=True)
    yield
    if app.state.langfuse is not None:
        app.state.langfuse.flush()  # send any buffered traces on shutdown


app = FastAPI(title="Meridian Pay Document Q&A", version="0.8.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Allow the static page to call the API from another origin (e.g. opened as a
# file:// page, or served from a different port). Wide-open is fine for a local
# teaching setup; a real deployment would pin allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
def ingest() -> IngestResponse:
    settings = app.state.settings
    client = app.state.client
    chunks = rag.ingest(settings, client)
    documents = sum(1 for _ in rag.iter_corpus())
    return IngestResponse(documents=documents, chunks=chunks)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(_PER_MINUTE)
@limiter.limit(_PER_DAY, key_func=_global_key)
def query(request: Request, req: QueryRequest) -> QueryResponse:
    settings = app.state.settings
    client = app.state.client
    langfuse = app.state.langfuse
    # Auto-ingest on first use so a fresh store still answers (same as the CLI).
    if rag.count_chunks(settings) == 0:
        rag.ingest(settings, client)

    if langfuse is None:
        answer_text, chunks = rag.answer_with_chunks(settings, client, req.question, req.k)
        sources = list(dict.fromkeys(c.doc for c in chunks))
        return QueryResponse(answer=answer_text, sources=sources)

    # Wrap the request in a Langfuse trace (SDK v4 API). The LLM + embedding
    # calls inside answer_with_chunks are auto-traced by the instrumented client
    # (token cost + latency); here we add the request span and retrieval quality.
    with langfuse.start_as_current_observation(
        name="query", as_type="span", input={"question": req.question}
    ):
        answer_text, chunks = rag.answer_with_chunks(settings, client, req.question, req.k)
        sources = list(dict.fromkeys(c.doc for c in chunks))
        distances = [c.distance for c in chunks]
        langfuse.update_current_span(
            output={"answer": answer_text, "sources": sources},
            metadata={
                "top_k": req.k or settings.top_k,
                "num_chunks": len(chunks),
                "retrieved_docs": sources,
                "min_distance": min(distances, default=None),
                "max_distance": max(distances, default=None),
            },
        )
        langfuse.set_current_trace_io(
            input={"question": req.question},
            output={"answer": answer_text},
        )
    return QueryResponse(answer=answer_text, sources=sources)


# Serve the static frontend at "/". Mounted last so the API routes above take
# precedence; GET / returns frontend/index.html (html=True).
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
