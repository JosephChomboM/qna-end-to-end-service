"""RAG evaluation against the frozen question set (Phase 4).

This is the **dev/CI gate** in the three-way distinction:

    RAG eval (Phase 4)  -> pass/fail gate on a fixed question set   [this file]
    MLflow   (Phase 7)  -> offline comparison of runs
    Langfuse (Phase 8)  -> live request monitoring

For each question in `eval/questions.jsonl` it runs the real RAG pipeline, then
scores three things:

  - retrieval_hit_rate : did the gold `source_doc` appear in the retrieved chunks?
                         (deterministic — tests the retriever)
  - faithfulness       : are the answer's claims supported by the retrieved
                         context? (LLM-as-judge — catches hallucination)
  - relevance          : does the answer correctly address the question, matching
                         the expected answer? (LLM-as-judge — catches wrong answers)

The same metrics recur in Phase 7 (MLflow) and Phase 8 (Langfuse); the point of
the teach is that they live at different lifecycle stages. Phase 7 imports
`evaluate()` to compare runs; here it is a pass/fail gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import AzureOpenAI

from app import rag
from app.config import Settings, get_settings

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"

# Pass/fail thresholds for the gate. The corpus is synthetic and unambiguous, so
# a correct pipeline scores near 1.0; 0.8 is a real gate that isn't flaky.
THRESHOLDS = {"retrieval_hit_rate": 0.8, "faithfulness": 0.8, "relevance": 0.8}

JUDGE_SYSTEM = (
    "You are a strict evaluator of a retrieval-augmented answer. "
    "Respond ONLY with a JSON object and nothing else."
)

JUDGE_TEMPLATE = """\
Question:
{question}

Reference (gold) answer:
{expected}

Retrieved context the model was given:
{context}

Model's answer:
{answer}

Score two things as integers 0 or 1:
- "faithful": 1 if every factual claim in the model's answer is supported by the
  retrieved context, else 0.
- "relevant": 1 if the model's answer correctly addresses the question and is
  consistent with the reference answer, else 0.

Return exactly: {{"faithful": 0 or 1, "relevant": 0 or 1}}
"""


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _judge(client: AzureOpenAI, s: Settings, *, question, expected, context, answer) -> dict:
    resp = client.chat.completions.create(
        model=s.chat_deployment,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_TEMPLATE.format(
                    question=question, expected=expected, context=context, answer=answer
                ),
            },
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate stray text/fences
    data = json.loads(match.group(0)) if match else {}
    return {
        "faithful": int(bool(data.get("faithful", 0))),
        "relevant": int(bool(data.get("relevant", 0))),
    }


@dataclass
class EvalRow:
    question: str
    expected: str
    source_doc: str
    answer: str
    retrieved_docs: list[str]
    hit: int
    faithful: int
    relevant: int


@dataclass
class EvalReport:
    rows: list[EvalRow] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        n = len(self.rows) or 1
        return {
            "retrieval_hit_rate": sum(r.hit for r in self.rows) / n,
            "faithfulness": sum(r.faithful for r in self.rows) / n,
            "relevance": sum(r.relevant for r in self.rows) / n,
        }

    def passed(self, thresholds: dict[str, float] = THRESHOLDS) -> bool:
        m = self.metrics
        return all(m[k] >= t for k, t in thresholds.items())


def evaluate(
    s: Settings,
    client: AzureOpenAI,
    questions: list[dict] | None = None,
    k: int | None = None,
) -> EvalReport:
    questions = questions if questions is not None else load_questions()
    report = EvalReport()
    for q in questions:
        answer_text, chunks = rag.answer_with_chunks(s, client, q["question"], k)
        retrieved_docs = list(dict.fromkeys(c.doc for c in chunks))
        hit = int(q["source_doc"] in retrieved_docs)
        scores = _judge(
            client,
            s,
            question=q["question"],
            expected=q["expected_answer"],
            context=rag.build_context(chunks),
            answer=answer_text,
        )
        report.rows.append(
            EvalRow(
                question=q["question"],
                expected=q["expected_answer"],
                source_doc=q["source_doc"],
                answer=answer_text,
                retrieved_docs=retrieved_docs,
                hit=hit,
                faithful=scores["faithful"],
                relevant=scores["relevant"],
            )
        )
    return report


def main() -> int:
    s = get_settings()
    client = rag.get_client(s)
    if rag.count_chunks(s) == 0:
        print("Vector store empty — ingesting corpus first...")
        rag.ingest(s, client)

    report = evaluate(s, client)
    print(f"\n{'hit':>3} {'faith':>5} {'rel':>3}  question")
    print("-" * 72)
    for r in report.rows:
        flag = "" if (r.hit and r.faithful and r.relevant) else "  <-- check"
        print(f"{r.hit:>3} {r.faithful:>5} {r.relevant:>3}  {r.question[:50]}{flag}")

    print("\n=== metrics ===")
    m = report.metrics
    for name, val in m.items():
        thr = THRESHOLDS[name]
        print(f"  {name:<20} {val:.2f}  (threshold {thr:.2f})  {'PASS' if val >= thr else 'FAIL'}")

    ok = report.passed()
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
