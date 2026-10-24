"""
generate.py - the full RAG loop: retrieve -> build prompt -> LLM -> cited answer.

Try it from the project root:
    python -m src.generate "How many credits is the Master's Thesis at KIT?" --backend ollama
    python -m src.generate "Wer leitet das Modul Constraint Solving?" --backend anthropic
    python -m src.generate "What is the cost of living in Karlsruhe?" --backend ollama   # should refuse
"""
import argparse
import re

from src.llm import DEFAULT_MODELS, chat
from src.retrieve import Retriever

SYSTEM_PROMPT = """You are ModuleMate, an assistant that answers questions about university \
master's module handbooks.

Rules:
1. Use ONLY the numbered context passages below. Never use outside knowledge and never guess.
2. Answer concisely, in the same language as the question.
3. After every fact, cite the passage number(s) it comes from, like [1] or [2][3].
4. If the passages come from different universities, keep them separate and name the university.
5. If the passages do not contain the answer, reply with exactly: NOT_FOUND
   (nothing else). Missing information is better than a wrong answer."""

NOT_FOUND_MESSAGE = "I couldn't find this in the provided module handbooks."


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] (source: {h['university']}, pages {h['page_start']}-{h['page_end']})\n{h['text']}")
    return "\n\n".join(blocks)


def answer(question: str, retriever: Retriever, backend: str = "ollama", model: str | None = None,
           k: int = 5, mode: str = "dense") -> dict:
    hits = retriever.search(question, k=k, mode=mode)
    context = build_context(hits)
    user = f"Context passages:\n\n{context}\n\nQuestion: {question}"
    raw = chat(backend, SYSTEM_PROMPT, user, model)

    refused = raw.strip().upper().startswith("NOT_FOUND")
    cited_ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw) if 1 <= int(n) <= len(hits)})
    sources = [{"n": n, "university": hits[n - 1]["university"], "source": hits[n - 1]["source"],
                "module_name": hits[n - 1]["module_name"], "module_id": hits[n - 1]["module_id"],
                "pages": f"{hits[n - 1]['page_start']}-{hits[n - 1]['page_end']}"} for n in cited_ids]
    return {
        "question": question,
        "answer": NOT_FOUND_MESSAGE if refused else raw.strip(),
        "raw_answer": raw,
        "refused": refused,
        "sources": [] if refused else sources,
        "retrieved": [{"source": h["source"], "module_id": h["module_id"]} for h in hits],
        "context": context,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "anthropic", "mock"])
    ap.add_argument("--model", default=None, help=f"default: {DEFAULT_MODELS}")
    ap.add_argument("--retrieval", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    res = answer(a.question, Retriever(), a.backend, a.model, a.k, a.retrieval)
    print("\n" + res["answer"])
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['university']} - {s['module_name']} ({s['module_id']}), pages {s['pages']}")


if __name__ == "__main__":
    main()