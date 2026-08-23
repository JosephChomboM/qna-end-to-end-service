# Document Q&A Service — End-to-End Teaching Project

## Purpose

A complete, small RAG system built to **teach the shape of a modern ML/AI
system end-to-end**. The value is not any single tool — it's showing how one problem
(answering questions over a document corpus) pulls in each layer *when the problem
demands it*.

**Audience:** one colleague, technical, model-risk background. Goal is breadth — see
the whole system — not depth in any one layer.

## Non-goals (read this before adding anything)

- This is **not** a production system. Every layer stays **thin**: the minimum that
  demonstrates the concept clearly.
- **No parallel building.** Build one vertical slice that answers a question, then
  operationalize outward. Each phase runs before the next is added.
- No feature creep. If a tool isn't in the phase list below, it doesn't go in.

## The three-way distinction to teach (the intellectual core)

Three things in this project look similar and are constantly confused. The whole point
of the teach is to separate them cleanly:

| Layer            | Phase | Lifecycle stage | What it does                                        |
|------------------|-------|-----------------|-----------------------------------------------------|
| RAG eval test    | 4     | dev / CI        | Pass/fail **gate** on a fixed question set          |
| MLflow           | 7     | experiment      | **Offline comparison** of runs (chunk size, top-k…) |
| Langfuse         | 8     | production      | **Live request** monitoring of the running service  |

Same-ish metrics (faithfulness, relevance), three different lifecycle stages. This is
validation staged across dev, experiment, and production. Script this explanation in
advance — if it blurs, the teach fails.

## Stack

- **Language/API:** Python, FastAPI
- **Vector store:** PostgreSQL + pgvector
- **LLM/embeddings:** provider via API (keys from env only)
- **Config:** pydantic-settings + `.env`
- **Tests:** pytest
- **Container:** Docker + docker-compose
- **CI/CD:** GitHub Actions
- **Experiment tracking:** MLflow
- **Observability:** Langfuse
- **Orchestration:** Kubernetes on `kind` or `minikube` (local)

## Corpus & eval set (Phase 0 — decide nothing later)

**Corpus:** a synthetic internal knowledge base for a fictional company (e.g. a small
fintech). ~12–15 short markdown documents: an employee handbook, a couple of product
specs, a policy doc or two, an FAQ. Written so facts are self-contained and unambiguous.

**Why synthetic:** you own every fact, so eval ground truth is unambiguous and there are
no copyright concerns. This is what makes the eval gate real instead of hand-wavy.

**Eval set:** ~20 question/answer pairs against that corpus, stored as a fixed file
(`eval/questions.jsonl`). Each entry: `question`, `expected_answer`, `source_doc`.
**This file is frozen** once written — if corpus or questions drift, MLflow runs stop
being comparable and the CI gate becomes noise.

---

## Build order (each phase has acceptance criteria — don't advance until met)

### Phase 0 — Repo + corpus + eval set
- Init repo, Python env, `.gitignore` (exclude `.env`, `__pycache__`, mlruns).
- Generate the synthetic corpus into `data/corpus/`.
- Write `eval/questions.jsonl` (~20 pairs).
- **Done when:** corpus and eval file exist and are committed.

### Phase 1 — Core RAG loop (local script, no ceremony)
- ingest → chunk → embed → store in pgvector → retrieve top-k → call LLM → answer.
- Runnable as a plain script, answering one real question end to end.
- **Done when:** `python -m app.rag "<a question>"` returns a grounded answer + sources.

### Phase 2 — Config / secrets
- `pydantic-settings` reads all keys/URLs from `.env`. No secrets in code.
- **Done when:** nothing in the repo contains a hardcoded key; app reads from env.

### Phase 3 — FastAPI
- Wrap the loop in `/ingest` and `/query` endpoints.
- **Done when:** `POST /query` returns the same answer the script did.

### Phase 4 — Testing + RAG eval
- Unit tests: chunking, retrieval.
- **RAG eval test:** run the frozen question set, score faithfulness/relevance,
  assert it clears a threshold. This becomes the CI gate.
- **Done when:** `pytest` passes and the eval test produces a pass/fail result.

### Phase 5 — Docker + docker-compose
- Dockerfile for the API; compose file for API + postgres/pgvector.
- **Done when:** `docker compose up` brings the whole thing up and `/query` works.

### Phase 6 — CI/CD (GitHub Actions)
- On push: lint → test → build image → push to a registry.
- The Phase 4 eval test is a **merge gate**.
- **Done when:** a push runs the pipeline green and publishes an image.

### Phase 7 — MLflow
- Offline experiment tracking: vary chunk size / top-k / prompt across runs, log
  faithfulness + relevance per run against the frozen eval set.
- **Done when:** MLflow UI shows ≥3 comparable runs with metrics.

### Phase 8 — Langfuse
- Instrument the live `/query` path: latency, token cost, retrieval quality per request.
- **Done when:** real queries appear as traces in Langfuse.

### Phase 9 — Kubernetes (stretch / learning goal)
- Deploy the Phase 6 image to `kind`/`minikube`: one Deployment, one Service, secrets
  as k8s secrets. Keep it minimal so pods/deployments/services stay visible.
- **Done when:** the service answers a query from inside the cluster.

---

## Suggested repo structure

```
.
├── app/
│   ├── config.py          # pydantic-settings
│   ├── rag.py             # core loop: ingest/chunk/embed/retrieve/answer
│   └── main.py            # FastAPI endpoints
├── data/corpus/           # synthetic markdown docs
├── eval/
│   ├── questions.jsonl    # frozen eval set
│   └── run_eval.py        # scores against the eval set
├── tests/
├── mlflow/                # experiment scripts
├── k8s/                   # deployment.yaml, service.yaml, secrets
├── .github/workflows/     # ci.yml
├── docker-compose.yml
├── Dockerfile
├── .env.example           # committed; .env is git-ignored
└── PLAN.md
```

## How to work through this in Claude Code

- Execute **one phase per session**. Paste the phase's acceptance criteria and don't
  move on until they're met.
- After Phase 3–4, distill the repo conventions (commands, structure, style) into a
  lean `CLAUDE.md` so future sessions have standing context.
- When teaching: walk the coworker down the phase list in order. The order *is* the
  lesson — it shows why each tool exists.