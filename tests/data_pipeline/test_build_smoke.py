"""
Smoke test for both vector store builders using 100 docs.
Verifies the full pipeline (load → chunk → embed → normalize → FAISS → save) without crashes.

Usage:
    cd project
    python -m src.data_pipeline.test_build_smoke
"""

import json
import os
import tempfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.data_pipeline.build_vector_store_minilm import (
    load_kaggle, build_chunks, embed_in_batches, KAGGLE_CSV,
    MAX_WORDS as MINILM_MAX_WORDS,
)
from src.data_pipeline.build_vector_store_mpnet import (
    MAX_WORDS as MPNET_MAX_WORDS,
)
from src.data_pipeline.build_vector_store_mpnet import build_chunks as build_chunks_mpnet

SAMPLE_SIZE = 100


def run_smoke(model_name: str, max_words: int, build_chunks_fn, label: str):
    print(f"\n{'='*50}")
    print(f"Smoke test: {label} ({model_name})")
    print(f"{'='*50}")

    docs = load_kaggle(KAGGLE_CSV)[:SAMPLE_SIZE]
    print(f"Loaded {len(docs)} docs")

    page_contents, metadatas = build_chunks_fn(docs)
    print(f"Chunks: {len(page_contents)}")

    model = SentenceTransformer(model_name)
    vectors = embed_in_batches(page_contents, model)
    print(f"Embedded shape: {vectors.shape}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vectors = np.ascontiguousarray(vectors / norms, dtype="float32")
    print("Normalized OK")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        tmp_path = f.name
    faiss.write_index(index, tmp_path)
    os.unlink(tmp_path)
    print("Write + cleanup OK")

    print(f"✓ {label} passed")


if __name__ == "__main__":
    run_smoke("all-MiniLM-L6-v2", MINILM_MAX_WORDS, build_chunks, "MiniLM")
    run_smoke("all-mpnet-base-v2", MPNET_MAX_WORDS, build_chunks_mpnet, "MPNet")
    print("\n✓ Both models passed smoke test — safe to run full build.")
