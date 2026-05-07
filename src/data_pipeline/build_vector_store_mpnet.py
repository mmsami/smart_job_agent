"""
Build the FAISS vector index from Kaggle + Arbeitnow job documents using all-mpnet-base-v2.

Loads both sources, maps to JobDocument, applies paragraph-based chunking
with fixed-size fallback (approx 295 words ≈ 384 tokens, ~38-word overlap), embeds
with all-mpnet-base-v2, and saves the index to data/vector_store/.

Naming convention: script and output files include the model short-name so
that switching to a different model (e.g. all-MiniLM-L6-v2) produces a
separate index without overwriting this one.

Usage:
    python -m src.data_pipeline.build_vector_store_mpnet

Output:
    data/vector_store/faiss_mpnet.index   — FAISS flat index (normalized, cosine-ready)
    data/vector_store/docstore_mpnet.json — parallel list of metadata + page_content

Prerequisites:
    data/kaggle_cleaned/postings_cleaned.csv   (from parse_kaggle.py)
    data/arbeitnow/arbeitnow_jobs.json         (from fetch_arbeitnow.py)
"""

import json
import logging
import os
import time
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

try:
    from src.data_pipeline.schemas import JobDocument
except ImportError:
    from schemas import JobDocument

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-mpnet-base-v2"

# all-mpnet-base-v2 hard token limit is 384 tokens.
# Word count approximation: tokens ≈ words × 1.3, so max safe words ≈ 295.
# Prefix ("Title at Company. Skills. ") is prepended per chunk — its word
# count is subtracted in build_chunks to keep total under the model limit.
MODEL_MAX_TOKENS = 384                        # hard limit for all-mpnet-base-v2
MAX_WORDS = int(MODEL_MAX_TOKENS / 1.3)       # ≈ 295 words → safe chunk body
OVERLAP_WORDS = int(50 / 1.3)                 # ≈ 38 words overlap between windows

BATCH_SIZE = 32  # Small batch to avoid OOM on Mac

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_cleaned", "postings_cleaned.csv")
ARBEITNOW_JSON = os.path.join(DATA_DIR, "arbeitnow", "arbeitnow_jobs.json")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")
os.makedirs(VECTOR_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_mpnet.index")
DOCSTORE_PATH = os.path.join(VECTOR_DIR, "docstore_mpnet.json")
DESCRIPTIONS_PATH = os.path.join(VECTOR_DIR, "job_descriptions_mpnet.json")


def get_str(value: object) -> Optional[str]:
    return str(value) if value is not None and bool(pd.notna(value)) else None


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


def split_into_chunks(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """
    Paragraph-based chunking with fixed-size fallback.

    Args:
        text:      Input text to chunk.
        max_words: Maximum words per chunk (caller subtracts prefix words so
                   the final embedded string stays within MODEL_MAX_TOKENS).

    Steps:
    1. Split on double-newlines.
    2. If paragraph fits within MODEL_MAX_TOKENS → keep as one chunk.
    3. If longer → slide a max_words window with OVERLAP_WORDS overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para.split()) <= max_words:
            chunks.append(para)
        else:
            words = para.split()
            step = max(max_words - OVERLAP_WORDS, 1)
            start = 0
            while start < len(words):
                window = words[start : start + max_words]
                chunks.append(" ".join(window))
                if start + max_words >= len(words):
                    break
                start += step

    # Last-resort: no paragraphs found — take first max_words words
    return chunks if chunks else [" ".join(text.split()[:max_words])]


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
        except Exception as e:
            logger.warning(f"Skipped Kaggle row (job_id={row.get('job_id')}): {e}")
            skipped += 1

    # Deduplicate by (title, company) — staffing agencies post the same role many times
    # with different job IDs. Keeps the first occurrence (lowest job_id in CSV order).
    seen: set[tuple[str, str]] = set()
    deduped: list[JobDocument] = []
    for doc in docs:
        key = (doc.title.lower().strip(), doc.company.lower().strip())
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    duplicates = len(docs) - len(deduped)
    if duplicates:
        print(f"  Removed {duplicates:,} duplicate title+company pairs")

    print(f"  Loaded {len(deduped):,} Kaggle docs ({skipped:,} skipped, {duplicates:,} deduped)")
    return deduped


def load_arbeitnow(path: str) -> list[JobDocument]:
    print(f"Loading Arbeitnow data from {path}...")
    with open(path, encoding="utf-8") as f:
        raw_list = json.load(f)

    docs: list[JobDocument] = []
    skipped = 0
    for raw in raw_list:
        try:
            docs.append(JobDocument(**raw))
        except Exception as e:
            logger.warning(f"Skipped Arbeitnow entry (job_id={raw.get('job_id')}): {e}")
            skipped += 1

    print(f"  Loaded {len(docs):,} Arbeitnow docs ({skipped:,} skipped)")
    return docs


# ── Chunk + embed ──────────────────────────────────────────────────────
def build_chunks(docs: list[JobDocument]) -> tuple[list[str], list[dict]]:
    """
    Returns:
        page_contents — list of strings to embed (prefixed for signal boosting)
        metadatas     — parallel list of metadata dicts

    Prefix word count is subtracted from max_words so the full embedded
    string (prefix + chunk body) stays within MODEL_MAX_TOKENS.
    """
    page_contents: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        # Skip empty descriptions — JobDocument validator normally prevents
        # these, but guard here in case of future schema changes.
        if not doc.description.strip():
            logger.warning(f"Skipped doc with empty description (job_id={doc.job_id})")
            continue

        prefix = doc.to_page_content_prefix()
        prefix_words = len(prefix.split())
        # Reserve space for prefix so total embedded string ≤ MODEL_MAX_TOKENS
        effective_max = max(MAX_WORDS - prefix_words, min(50, MAX_WORDS))  # floor respects token budget

        chunks = split_into_chunks(doc.description, max_words=effective_max)
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Build FAISS Vector Store (all-mpnet-base-v2)")
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
    if not page_contents:
        raise ValueError("No documents to index — check that data files exist and loaded correctly")

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

    # Save job_id → description lookup (separate from docstore to avoid repeating
    # description text once per chunk; retrieve_jobs reads this to populate JobRecord)
    descriptions = {
        doc.job_id: doc.description
        for doc in all_docs
        if doc.description.strip()
    }
    with open(DESCRIPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False)
    print(f"  Saved descriptions lookup to {DESCRIPTIONS_PATH} ({len(descriptions):,} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
