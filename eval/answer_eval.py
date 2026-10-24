"""
answer_eval.py - evaluate the full RAG system (retrieval + generation).

Run from the project root:
    python -m eval.answer_eval --backend ollama --tag llama32-3b
    python -m eval.answer_eval --backend anthropic --tag claude-haiku
    python -m eval.answer_eval --backend ollama --judge-backend anthropic --tag llama32-3b   # stronger judge

Metrics
  Answerable questions
    retrieval_hit    : a gold module was among the retrieved chunks
    citation_hit     : the answer cites at least one gold module
    correctness      : LLM judge compares answer with the gold answer (CORRECT=1, PARTIAL=0.5)
    faithfulness     : LLM judge checks every claim is supported by the retrieved context
    false_refusal    : the system said "not found" although the answer exists
  Unanswerable questions
    correct_refusal  : the system correctly said "not found"

NOTE: LLM judges are imperfect, especially small local ones. Spot-check a few rows of the
output file by hand and say so in your README.

Writes: results/answers_<tag>.jsonl, results/answer_summary_<tag>.md
"""
import argparse
import json
import re
from pathlib import Path

from src.generate import answer
from src.llm import chat
from src.retrieve import Retriever

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("results")

CORRECTNESS_SYSTEM = (
    "You are a strict grader. Compare the SYSTEM ANSWER with the REFERENCE ANSWER for the question. "
    "Reply with exactly one word: CORRECT (all key facts of the reference are present and nothing "
    "contradicts it), PARTIAL (some key facts are present or some are missing), or INCORRECT "
    "(wrong, contradicting, or missing the key facts). Ignore citation markers like [1].")
FAITHFULNESS_SYSTEM = (
    "You are checking faithfulness. Given CONTEXT passages and an ANSWER, reply with exactly one "
    "word: SUPPORTED if every factual claim in the answer is supported by the context, "
    "otherwise UNSUPPORTED. Ignore citation markers like [1].")


def judge(system, user, backend, model, options):
    out = chat(backend, system, user, model, max_tokens=10).upper()
    m = re.search(r"\b(" + "|".join(options) + r")\b", out)
    return m.group(1) if m else "UNPARSEABLE"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-backend", default=None, help="default: same as --backend")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--retrieval", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tag", default=None, help="name used in output file names")
    a = ap.parse_args()
    jb, jm = a.judge_backend or a.backend, a.judge_model or (a.model if not a.judge_backend else None)
    tag = a.tag or f"{a.backend}-{a.retrieval}"

    retriever = Retriever()
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    OUT_DIR.mkdir(exist_ok=True)
    records = []

    for n, q in enumerate(questions, 1):
        print(f"[{n}/{len(questions)}] {q['id']} {q['question'][:70]}")
        res = answer(q["question"], retriever, a.backend, a.model, a.k, a.retrieval)
        gold = {(g["source"], g["module_id"]) for g in q["gold_units"]}
        rec = {"id": q["id"], "type": q["type"], "question": q["question"],
               "gold_answer": q["gold_answer"], "answer": res["answer"], "refused": res["refused"],
               "cited": [f"{s['source']}:{s['module_id']}" for s in res["sources"]]}

        if q["answerable"]:
            rec["retrieval_hit"] = int(any((r["source"], r["module_id"]) in gold for r in res["retrieved"]))
            rec["citation_hit"] = int(any((s["source"], s["module_id"]) in gold for s in res["sources"]))
            if res["refused"]:
                rec.update(correctness="REFUSED", correctness_score=0.0, faithful=None)
            else:
                verdict = judge(CORRECTNESS_SYSTEM,
                                f"Question: {q['question']}\nREFERENCE ANSWER: {q['gold_answer']}\n"
                                f"SYSTEM ANSWER: {res['answer']}", jb, jm, ["CORRECT", "PARTIAL", "INCORRECT"])
                faith = judge(FAITHFULNESS_SYSTEM,
                              f"CONTEXT:\n{res['context']}\n\nANSWER: {res['answer']}", jb, jm,
                              ["SUPPORTED", "UNSUPPORTED"])
                rec.update(correctness=verdict,
                           correctness_score={"CORRECT": 1.0, "PARTIAL": 0.5}.get(verdict, 0.0),
                           faithful=int(faith == "SUPPORTED"))
        else:
            rec["correct_refusal"] = int(res["refused"])
        records.append(rec)

    # ---- summary -----------------------------------------------------------------
    ans = [r for r in records if "retrieval_hit" in r]
    unans = [r for r in records if "correct_refusal" in r]
    not_ref = [r for r in ans if not r["refused"]]
    rows = [
        ("answerable questions", len(ans)),
        ("retrieval_hit", f"{mean(r['retrieval_hit'] for r in ans):.3f}"),
        ("citation_hit", f"{mean(r['citation_hit'] for r in ans):.3f}"),
        ("correctness (CORRECT=1, PARTIAL=.5)", f"{mean(r['correctness_score'] for r in ans):.3f}"),
        ("faithfulness (non-refused answers)", f"{mean(r['faithful'] for r in not_ref):.3f}"),
        ("false_refusal_rate", f"{mean(int(r['refused']) for r in ans):.3f}"),
        ("unanswerable questions", len(unans)),
        ("correct_refusal_rate", f"{mean(r['correct_refusal'] for r in unans):.3f}"),
    ]
    md = (f"## Answer quality - generator: {a.backend} {a.model or ''} | judge: {jb} | "
          f"retrieval: {a.retrieval}, k={a.k}\n\n| metric | value |\n|---|---|\n" +
          "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n")
    (OUT_DIR / f"answer_summary_{tag}.md").write_text(md, encoding="utf-8")
    with (OUT_DIR / f"answers_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n" + md)
    bad = [(r["id"], r.get("correctness")) for r in ans if r["correctness"] != "CORRECT"]
    print("Not fully correct:", bad)
    print("Wrongly answered (should refuse):", [r["id"] for r in unans if not r["correct_refusal"]])


if __name__ == "__main__":
    main()