"""
index.py - embed all chunks and save the vectors.

Run from the project root:
    python -m src.index

Reads  : data/processed/chunks.jsonl
Writes : data/processed/embeddings.npy   (row i = chunk i of chunks.jsonl)

First run downloads the embedding model (~1 GB, one time only).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
EMB_FILE = Path("data/processed/embeddings.npy")

# Multilingual (German + English) - needed because one handbook is in German.
# Faster/smaller alternative: "intfloat/multilingual-e5-small"
MODEL_NAME = "intfloat/multilingual-e5-base"


def main():
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks. Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # E5 models expect the prefix "passage: " for documents and "query: " for questions.
    texts = ["passage: " + c["text"] for c in chunks]
    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> dot product == cosine similarity
    )
    np.save(EMB_FILE, emb.astype(np.float32))
    print(f"Saved embeddings {emb.shape} -> {EMB_FILE}")


if __name__ == "__main__":
    main()