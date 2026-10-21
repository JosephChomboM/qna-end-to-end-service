# Meridian Pay Document Q&A

Ask questions in plain English about a company's internal documents and get a grounded answer with the source files it came from, instead of searching handbooks and policies by hand.

A small end-to-end RAG system, built to show the shape of a modern ML/AI service: retrieval, an API, a frontend, tests with an eval gate, containers, CI/CD, experiment tracking, live observability and Kubernetes. Every layer is deliberately thin.

> The corpus is synthetic (a fictional fintech, ~15 markdown docs in `data/corpus/`), so every fact is known and the eval has unambiguous ground truth.

## How it works

```
 browser ──POST /query──▶ FastAPI ──embed question──▶ Azure OpenAI embeddings
 (frontend/index.html)       │
                             ├──top-k nearest chunks──▶ PostgreSQL + pgvector
                             └──question + chunks────▶ Azure OpenAI chat ──▶ answer + sources
```

All keys, embeddings, vector search and LLM calls stay server-side. The page only sends a question and renders the response.

## Run it locally

```bash
cp .env.example .env          # set your Azure OpenAI endpoint, key and deployment names
docker compose up --build     # API + Postgres/pgvector
open http://localhost:8000/   # first question auto-ingests the corpus
```

Without Docker: `pip install -r requirements.txt`, point `DATABASE_URL` at a Postgres with pgvector, then `python -m app.rag "How many PTO days do employees get?"`.

## The three-way distinction

The same kind of metrics (faithfulness, relevance) show up at three lifecycle stages. They are different tools for different questions:

| Tool | Stage | What it answers |
|------|-------|-----------------|
| RAG eval test (`pytest`) | dev / CI | Does this change still pass on a fixed question set? **Pass/fail gate.** |
| MLflow (`mlflow/experiment.py`) | experiment | Which chunk size / top-k / prompt is better? **Offline comparison of runs.** |
| Langfuse (`app/observability.py`) | production | How is the running service behaving? **Live per-request traces** (latency, token cost, retrieval distance). |

## Evaluation

`eval/questions.jsonl` holds 20 frozen question/answer pairs with their gold source document. `pytest` runs them through the real pipeline and scores:

- **retrieval hit rate**: did the gold document appear in the retrieved chunks (deterministic)
- **faithfulness**: is every claim in the answer supported by the retrieved context (LLM judge)
- **relevance**: does the answer match the reference answer (LLM judge)

Each must score at least 0.8, and the same test is the merge gate in CI. Run it yourself with `python -m eval.run_eval`. The question file is frozen on purpose: changing it would make MLflow runs and the gate incomparable.

## Public demo guards

The deployed demo calls a paid LLM API, so it protects the budget:

- `/query` is rate limited per IP (`RATE_LIMIT_PER_MINUTE`) and globally (`RATE_LIMIT_PER_DAY`).
- `POST /ingest` rebuilds the vector table, so set `ADMIN_TOKEN` in production; it then requires an `X-Admin-Token` header.
- Set a spending limit or budget alert on the Azure OpenAI resource as a last line of defense.

## Layout

```
app/           config.py (pydantic-settings), rag.py (core loop), main.py (FastAPI), observability.py (Langfuse)
frontend/      single static page, no build step
data/corpus/   synthetic markdown knowledge base
eval/          frozen question set + scorer
tests/         chunking, retrieval, API guards, eval gate
mlflow/        offline experiment runs
k8s/           Deployment, Service, Secret template (kind / minikube)
.github/       CI: lint → test + eval gate → build → push image to GHCR
```

## Stack

Python · FastAPI · PostgreSQL + pgvector · Azure OpenAI · pytest · Docker Compose · GitHub Actions · MLflow · Langfuse · Kubernetes
