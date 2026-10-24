"""
retrieval_eval.py - measure how well each retrieval mode finds the right module.

Run from the project root:
    python -m eval.retrieval_eval                    # bm25, dense, hybrid
    python -m eval.retrieval_eval --modes bm25       # one mode only
    python -m eval.retrieval_eval --k 10

Metrics (computed over answerable questions only):
    Hit@k    : 1 if at least one gold module appears in the top-k chunks
    Recall@k : fraction of the gold modules that appear in the top-k chunks
               (matters for questions that need 2+ modules)
    MRR      : mean of 1/rank of the first chunk that belongs to a gold module

A retrieved chunk counts as correct if its (source, module_id) is in gold_units.

Writes: results/retrieval_per_question.csv, results/retrieval_summary.md
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.retrieve import Retriever

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("results")
KS = (1, 3, 5)


def load_questions():
    return [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]


def evaluate(retriever: Retriever, questions, mode: str, max_k: int):
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        gold = {(g["source"], g["module_id"]) for g in q["gold_units"]}
        hits = retriever.search(q["question"], k=max_k, mode=mode)
        ranked = [(h["source"], h["module_id"]) for h in hits]

        first = next((i + 1 for i, u in enumerate(ranked) if u in gold), None)
        row = {"id": q["id"], "type": q["type"], "mode": mode, "first_hit_rank": first or 0,
               "mrr": 1.0 / first if first else 0.0}
        for k in KS:
            top = set(ranked[:k])
            row[f"hit@{k}"] = int(bool(top & gold))
            row[f"recall@{k}"] = len(top & gold) / len(gold)
        rows.append(row)
    return rows


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def summary_table(all_rows, modes):
    cols = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]
    lines = ["| mode | n | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for m in modes:
        rs = [r for r in all_rows if r["mode"] == m]
        lines.append(f"| {m} | {len(rs)} | " + " | ".join(f"{mean(rs, c):.3f}" for c in cols) + " |")
    return "\n".join(lines)


def by_type_table(all_rows, modes, metric="hit@3"):
    types = sorted({r["type"] for r in all_rows})
    lines = [f"| type | n | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for t in types:
        n = len({r["id"] for r in all_rows if r["type"] == t})
        cells = []
        for m in modes:
            rs = [r for r in all_rows if r["type"] == t and r["mode"] == m]
            cells.append(f"{mean(rs, metric):.2f}")
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=10, help="how many chunks to retrieve per question")
    a = ap.parse_args()

    questions = load_questions()
    retriever = Retriever()

    # sanity check: every gold module must exist in the corpus
    corpus = {(c["source"], c["module_id"]) for c in retriever.chunks}
    for q in questions:
        for g in q["gold_units"]:
            if (g["source"], g["module_id"]) not in corpus:
                print(f"[warn] {q['id']}: gold unit {g} not found in chunks.jsonl")

    all_rows = []
    for mode in a.modes:
        print(f"evaluating {mode} ...")
        all_rows += evaluate(retriever, questions, mode, a.k)

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "retrieval_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    md = ("## Retrieval results\n\n" + summary_table(all_rows, a.modes) +
          "\n\n### Hit@3 by question type\n\n" + by_type_table(all_rows, a.modes) + "\n")
    (OUT_DIR / "retrieval_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    # list the questions where each mode fails to find a gold module in the top 5
    print("Failures (no gold module in top-5):")
    for m in a.modes:
        bad = [r["id"] for r in all_rows if r["mode"] == m and r["hit@5"] == 0]
        print(f"  {m:7s}: {bad if bad else 'none'}")


if __name__ == "__main__":
    main()