"""
retrieve.py - search the chunks with three strategies.

    dense  : embedding similarity (understands meaning, works across EN/DE)
    bm25   : keyword matching (great for exact IDs, names, "5 CP")
    hybrid : both, merged with Reciprocal Rank Fusion (RRF)

Try it from the project root:
    python -m src.retrieve "Which modules cover reinforcement learning?"
    python -m src.retrieve "prerequisites for the master thesis" --mode bm25 --k 3
    python -m src.retrieve "Welche Module behandeln Data Mining?" --source tu_braunschweig
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMB_FILE = Path("data/processed/embeddings.npy")
MODEL_NAME = "intfloat/multilingual-e5-base"  # must match index.py
RRF_K = 60  # standard RRF constant


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.emb = np.load(EMB_FILE) if EMB_FILE.exists() else None
        self._model = None  # loaded lazily, only if dense/hybrid is used

    # ---- scoring ---------------------------------------------------------
    def _dense_scores(self, query: str) -> np.ndarray:
        if self.emb is None:
            raise RuntimeError("embeddings.npy not found - run: python -m src.index")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        q = self._model.encode(["query: " + query], normalize_embeddings=True)[0]
        return self.emb @ q  # cosine similarity (vectors are normalised)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(tokenize(query)))

    # ---- public API ------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid", source: str | None = None):
        """Return top-k chunks as dicts with extra keys: score, rank."""
        allowed = np.array([source is None or c["source"] == source for c in self.chunks])

        def ranking(scores):
            scores = np.where(allowed, scores, -np.inf)
            return [i for i in np.argsort(-scores) if np.isfinite(scores[i])]

        if mode == "dense":
            scores = self._dense_scores(query)
            order = ranking(scores)[:k]
            final = {i: float(scores[i]) for i in order}
        elif mode == "bm25":
            scores = self._bm25_scores(query)
            order = ranking(scores)[:k]
            final = {i: float(scores[i]) for i in order}
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for scores in (self._dense_scores(query), self._bm25_scores(query)):
                for rank, i in enumerate(ranking(scores)[:50]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)[:k]
            final = {i: fused[i] for i in order}
        else:
            raise ValueError("mode must be dense | bm25 | hybrid")

        return [{**self.chunks[i], "score": s, "rank": r + 1}
                for r, (i, s) in enumerate(final.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--source", default=None, choices=[None, "kit", "tu_dortmund", "tu_braunschweig"])
    a = ap.parse_args()

    r = Retriever()
    for hit in r.search(a.query, k=a.k, mode=a.mode, source=a.source):
        print(f"\n#{hit['rank']}  score={hit['score']:.4f}  {hit['university']} | "
              f"{hit['module_name']} ({hit['module_id']})  pages {hit['page_start']}-{hit['page_end']}")
        body = hit["text"].split("\n", 1)[1].replace("\n", " ")
        print("   ", body[:260], "...")


if __name__ == "__main__":
    main()