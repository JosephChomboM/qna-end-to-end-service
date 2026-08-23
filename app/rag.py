"""Core RAG loop for the Document Q&A Service (Phase 1).

Pipeline: ingest -> chunk -> embed -> store in pgvector -> retrieve top-k ->
call the LLM -> grounded answer with sources.

Runnable as a plain script:

    python -m app.rag ingest          # (re)build the vector store from the corpus
    python -m app.rag "your question" # retrieve + answer (auto-ingests if empty)

Config comes from environment variables (loaded from .env). Phase 2 replaces the
os.getenv calls here with pydantic-settings; for now this stays deliberately thin.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from openai import AzureOpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import Settings, get_settings

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


# --------------------------------------------------------------------------- #
# Chunking (pure function — unit-tested in Phase 4)
# --------------------------------------------------------------------------- #
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, breaking on whitespace so words stay whole.

    Greedy: pack words up to ~chunk_size characters, then start the next chunk
    `overlap` characters back so context spans chunk boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        # +1 for the joining space (except the first word).
        added = len(word) + (1 if current else 0)
        if current and current_len + added > chunk_size:
            chunks.append(" ".join(current))
            # Build the overlap tail from the end of the chunk just emitted.
            tail: list[str] = []
            tail_len = 0
            for w in reversed(current):
                if tail_len + len(w) + (1 if tail else 0) > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + (1 if len(tail) > 1 else 0)
            current = tail
            current_len = len(" ".join(current)) if current else 0
            # add the word that didn't fit
            added = len(word) + (1 if current else 0)
        current.append(word)
        current_len += added
    if current:
        chunks.append(" ".join(current))
    return chunks


def iter_corpus(corpus_dir: Path = CORPUS_DIR):
    """Yield (doc_name, text) for every markdown file in the corpus, sorted."""
    for path in sorted(corpus_dir.glob("*.md")):
        yield path.name, path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def get_client(s: Settings) -> AzureOpenAI:
    missing = [
        name
        for name, val in {
            "AZURE_OPENAI_ENDPOINT": s.azure_endpoint,
            "AZURE_OPENAI_API_KEY": s.azure_api_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": s.chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": s.embedding_deployment,
        }.items()
        if not val
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nSet them in .env (see .env.example)."
        )
    return AzureOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_key=s.azure_api_key,
        api_version=s.azure_api_version,
    )


def embed_texts(client: AzureOpenAI, deployment: str, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching to stay well under request limits."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        resp = client.embeddings.create(model=deployment, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


# --------------------------------------------------------------------------- #
# Vector store (pgvector)
# --------------------------------------------------------------------------- #
def connect(s: Settings) -> psycopg.Connection:
    conn = psycopg.connect(s.database_url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ingest(s: Settings, client: AzureOpenAI) -> int:
    """(Re)build the chunks table from the corpus. Returns number of chunks stored."""
    records: list[tuple[str, int, str]] = []
    for doc_name, text in iter_corpus():
        for idx, chunk in enumerate(chunk_text(text, s.chunk_size, s.chunk_overlap)):
            records.append((doc_name, idx, chunk))

    if not records:
        raise SystemExit(f"No corpus documents found in {CORPUS_DIR}")

    embeddings = embed_texts(client, s.embedding_deployment, [r[2] for r in records])

    with connect(s) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            f"""
            CREATE TABLE chunks (
                id serial PRIMARY KEY,
                doc text NOT NULL,
                chunk_index int NOT NULL,
                content text NOT NULL,
                embedding vector({s.embedding_dim}) NOT NULL
            )
            """
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [
                    (doc, idx, content, Vector(emb))
                    for (doc, idx, content), emb in zip(records, embeddings)
                ],
            )
        conn.commit()
    return len(records)


def count_chunks(s: Settings) -> int:
    try:
        with connect(s) as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0
    except psycopg.errors.UndefinedTable:
        return 0


@dataclass
class Retrieved:
    doc: str
    chunk_index: int
    content: str
    distance: float


def retrieve(s: Settings, client: AzureOpenAI, query: str, k: int) -> list[Retrieved]:
    query_embedding = Vector(embed_texts(client, s.embedding_deployment, [query])[0])
    with connect(s) as conn:
        rows = conn.execute(
            """
            SELECT doc, chunk_index, content, embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        ).fetchall()
    return [Retrieved(doc, idx, content, float(dist)) for doc, idx, content, dist in rows]


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a helpful assistant for Meridian Pay employees and customers. "
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't have that information. "
    "Be concise and do not invent facts."
)


def build_context(chunks: list[Retrieved]) -> str:
    return "\n\n".join(f"[{c.doc}]\n{c.content}" for c in chunks)


def answer(s: Settings, client: AzureOpenAI, question: str, k: int | None = None) -> tuple[str, list[str]]:
    k = k or s.top_k
    chunks = retrieve(s, client, question, k)
    context = build_context(chunks)
    resp = client.chat.completions.create(
        model=s.chat_deployment,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    text = resp.choices[0].message.content or ""
    # De-duplicate source docs while preserving retrieval order.
    sources = list(dict.fromkeys(c.doc for c in chunks))
    return text, sources


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('Usage:\n  python -m app.rag ingest\n  python -m app.rag "your question"')
        return 1

    s = get_settings()
    client = get_client(s)

    if argv[0] == "ingest":
        n = ingest(s, client)
        print(f"Ingested {n} chunks from {sum(1 for _ in iter_corpus())} documents.")
        return 0

    question = " ".join(argv)
    if count_chunks(s) == 0:
        print("Vector store empty — ingesting corpus first...", file=sys.stderr)
        ingest(s, client)

    text, sources = answer(s, client, question)
    print(text)
    print("\nSources: " + ", ".join(sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
