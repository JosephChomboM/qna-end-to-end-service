"""Phase 7 — MLflow offline experiment tracking.

The EXPERIMENT lifecycle stage in the three-way distinction:

    RAG eval (Phase 4)  -> pass/fail gate on a fixed question set
    MLflow   (Phase 7)  -> offline comparison of runs                [this file]
    Langfuse (Phase 8)  -> live request monitoring

It varies chunk size / top-k / prompt and logs faithfulness + relevance for each
configuration, all against the SAME frozen eval set (that's what makes the runs
comparable). Reuses `eval.run_eval.evaluate()` — identical scoring to the CI gate,
different purpose: here we compare configs offline instead of gating a merge.

Run (from the project root, BY PATH — see note below):
    python mlflow/experiment.py
View:
    mlflow ui              # then open http://localhost:5000

NOTE: this directory is named `mlflow/` to match the project layout, which would
shadow the installed `mlflow` library if imported as a package. So run this file
by path (as above), NOT `python -m mlflow.experiment`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow  # the library — safe here: this script's dir has no nested `mlflow`

ROOT = Path(__file__).resolve().parent.parent
# Append (not insert) so `app`/`eval` resolve without shadowing the mlflow library.
sys.path.append(str(ROOT))

from app import rag  # noqa: E402
from app.config import get_settings  # noqa: E402
from eval.run_eval import evaluate  # noqa: E402

DEFAULT_PROMPT = rag.SYSTEM_PROMPT
STRICT_PROMPT = (
    "You are a precise assistant for Meridian Pay. Answer ONLY from the context, "
    "in one short sentence. If the answer is not in the context, reply exactly: "
    "\"I don't have that information.\" Never guess."
)

# Each dict is one experiment run. We vary all three knobs the phase calls for:
# chunk size, top-k, and prompt.
CONFIGS = [
    {"name": "baseline", "chunk_size": 800, "chunk_overlap": 120, "top_k": 4,
     "prompt": DEFAULT_PROMPT, "prompt_name": "default"},
    {"name": "small_chunks", "chunk_size": 400, "chunk_overlap": 80, "top_k": 4,
     "prompt": DEFAULT_PROMPT, "prompt_name": "default"},
    {"name": "topk8_strict", "chunk_size": 800, "chunk_overlap": 120, "top_k": 8,
     "prompt": STRICT_PROMPT, "prompt_name": "strict"},
]


def main() -> None:
    base = get_settings()
    client = rag.get_client(base)

    # MLflow 3 requires a database backend (the plain-file store is retired).
    # A local SQLite file keeps it zero-infra for this teaching setup.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("rag-eval-phase7")

    for cfg in CONFIGS:
        s = base.model_copy(
            update={
                "chunk_size": cfg["chunk_size"],
                "chunk_overlap": cfg["chunk_overlap"],
                "top_k": cfg["top_k"],
            }
        )
        print(
            f"\n=== run: {cfg['name']} "
            f"(chunk {cfg['chunk_size']}/{cfg['chunk_overlap']}, "
            f"top_k {cfg['top_k']}, prompt {cfg['prompt_name']}) ==="
        )

        # Rebuild the vector store for this chunking, and swap in the prompt variant.
        rag.ingest(s, client)
        rag.SYSTEM_PROMPT = cfg["prompt"]
        try:
            report = evaluate(s, client, k=cfg["top_k"])
        finally:
            rag.SYSTEM_PROMPT = DEFAULT_PROMPT  # always restore

        m = report.metrics
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.log_params(
                {
                    "chunk_size": cfg["chunk_size"],
                    "chunk_overlap": cfg["chunk_overlap"],
                    "top_k": cfg["top_k"],
                    "prompt": cfg["prompt_name"],
                    "embedding_deployment": base.embedding_deployment,
                    "chat_deployment": base.chat_deployment,
                    "num_questions": len(report.rows),
                }
            )
            mlflow.log_metrics(
                {
                    "faithfulness": m["faithfulness"],
                    "relevance": m["relevance"],
                    "retrieval_hit_rate": m["retrieval_hit_rate"],
                }
            )
        print(f"  metrics: {m}")

    # Leave the live store at the baseline chunking so the running API is unchanged.
    print("\nRestoring baseline ingest...")
    rag.ingest(base, client)
    print("Done. View runs with:  mlflow ui   (then open http://localhost:5000)")


if __name__ == "__main__":
    main()
