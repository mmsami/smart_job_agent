"""
Build the FAISS vector index from Kaggle + Arbeitnow job documents.

Loads both sources, maps to JobDocument, applies paragraph-based chunking
with fixed-size fallback (512 tokens, ~50-token overlap), embeds with
all-MiniLM-L6-v2, and saves the index to data/vector_store/.

Usage:
    python src/data_pipeline/build_vector_store.py

Output:
    data/vector_store/faiss.index   — FAISS flat index (normalized, cosine-ready)
    data/vector_store/docstore.json — parallel list of metadata + page_content

Prerequisites:
    data/kaggle_cleaned/postings_cleaned.csv   (from parse_kaggle.py)
    data/arbeitnow/arbeitnow_jobs.json         (from fetch_arbeitnow.py)
"""

import json
import os
import time
from typing import Optional, cast

import faiss
import numpy as np
import pandas as pd
try:
    from src.data_pipeline.schemas import JobDocument
except ImportError:
    from schemas import JobDocument
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_TOKENS = 512  # max tokens per chunk (approx by whitespace words × 1.3)
OVERLAP_TOKENS = 50  # overlap between fixed-size chunks
BATCH_SIZE = 512  # embedding batch size

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_cleaned", "postings_cleaned.csv")
ARBEITNOW_JSON = os.path.join(DATA_DIR, "arbeitnow", "arbeitnow_jobs.json")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(VECTOR_DIR, "docstore.json")


def get_str(value: object) -> Optional[str]:
    return cast(str, value) if value is not None and bool(pd.notna(value)) else None


def get_float(value: object) -> Optional[float]:
    if value is None or not bool(pd.notna(value)):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


# ── Tokenization (approximation — avoids loading a full tokenizer) ─────
def approx_token_count(text: str) -> int:
    """Approximate BPE token count from whitespace-word count."""
    return int(len(text.split()) * 1.3)


def split_into_chunks(text: str) -> list[str]:
    """
    Paragraph-based chunking with fixed-size fallback.

    1. Split on double-newlines.
    2. If paragraph ≤ MAX_TOKENS → keep as one chunk.
    3. If > MAX_TOKENS → slide a 512-token window with OVERLAP_TOKENS overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if approx_token_count(para) <= MAX_TOKENS:
            chunks.append(para)
        else:
            words = para.split()
            step = MAX_TOKENS - OVERLAP_TOKENS
            start = 0
            while start < len(words):
                window = words[start : start + MAX_TOKENS]
                chunks.append(" ".join(window))
                if start + MAX_TOKENS >= len(words):
                    break
                start += step

    return chunks if chunks else [text[:2000]]  # last-resort: truncate


# ── Source loaders ─────────────────────────────────────────────────────
def load_kaggle(path: str) -> list[JobDocument]:
    print(f"Loading Kaggle data from {path}...")
    df = pd.read_csv(path)
    docs: list[JobDocument] = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            doc = JobDocument(
                job_id=str(row["job_id"]),
                title=str(row["title"]),
                company=str(row["company_name"]),
                description=str(row["description"]),
                skill_labels=get_str(row.get("skill_labels")),
                location=get_str(row.get("location")),
                experience_level=get_str(row.get("formatted_experience_level")),
                work_type=get_str(row.get("formatted_work_type")),
                min_salary=get_float(row.get("min_salary")),
                max_salary=get_float(row.get("max_salary")),
                url=get_str(row.get("application_url")),
                source="kaggle",
            )
            docs.append(doc)
        except Exception:
            skipped += 1

    print(f"  Loaded {len(docs):,} Kaggle docs ({skipped:,} skipped)")
    return docs


def load_arbeitnow(path: str) -> list[JobDocument]:
    print(f"Loading Arbeitnow data from {path}...")
    with open(path, encoding="utf-8") as f:
        raw_list = json.load(f)

    docs: list[JobDocument] = []
    skipped = 0
    for raw in raw_list:
        try:
            docs.append(JobDocument(**raw))
        except Exception:
            skipped += 1

    print(f"  Loaded {len(docs):,} Arbeitnow docs ({skipped:,} skipped)")
    return docs


# ── Chunk + embed ──────────────────────────────────────────────────────
def build_chunks(docs: list[JobDocument]) -> tuple[list[str], list[dict]]:
    """
    Returns:
        page_contents — list of strings to embed (prefixed for signal boosting)
        metadatas     — parallel list of metadata dicts
    """
    page_contents: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        prefix = doc.to_page_content_prefix()
        chunks = split_into_chunks(doc.description)
        meta = doc.to_metadata()

        for chunk in chunks:
            # Signal boosting: prefix on every chunk so the embedding "knows"
            # the job title and skills even for body paragraphs
            page_contents.append(f"{prefix}{chunk}")
            metadatas.append(meta)

    return page_contents, metadatas


def embed_in_batches(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed texts in batches; returns float32 array shape (N, D)."""
    all_vecs = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs)
        print(f"  Embedded {min(start + BATCH_SIZE, total):>7,} / {total:,}", end="\r")
    print()
    return np.vstack(all_vecs).astype("float32")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Build FAISS Vector Store")
    print("=" * 60)

    # Load both sources
    kaggle_docs = load_kaggle(KAGGLE_CSV)
    arbeitnow_docs = load_arbeitnow(ARBEITNOW_JSON)
    all_docs = kaggle_docs + arbeitnow_docs
    print(f"\nTotal documents: {len(all_docs):,}")

    # Chunk
    print("\nChunking documents...")
    t0 = time.time()
    page_contents, metadatas = build_chunks(all_docs)
    print(
        f"  {len(page_contents):,} chunks from {len(all_docs):,} docs ({time.time() - t0:.1f}s)"
    )
    print(f"  Avg chunks/doc: {len(page_contents) / len(all_docs):.2f}")

    # Embed
    print(f"\nLoading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)
    print("Embedding chunks...")
    t0 = time.time()
    vectors = embed_in_batches(page_contents, model)
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {vectors.shape}")

    # Normalize for cosine similarity (FAISS IndexFlatIP on unit vectors = cosine)
    print("\nNormalizing vectors for cosine similarity...")
    faiss.normalize_L2(vectors)

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine
    index.add(vectors)  # type: ignore
    print(f"  Index built: {index.ntotal:,} vectors, dim={dim}")

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"  Saved index to {INDEX_PATH}")

    # Save docstore (page_content + metadata, parallel to index)
    docstore = [
        {"page_content": pc, "metadata": meta}
        for pc, meta in zip(page_contents, metadatas)
    ]
    with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False)
    print(f"  Saved docstore to {DOCSTORE_PATH} ({len(docstore):,} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
