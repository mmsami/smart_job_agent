"""
job_search.py — function that takes a person's profile and finds the 20 best job matches

Results logged to: project/iterations/job_search_results.md
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.workflow.models import CVProfile, JobRecord, JobSearchPreferences
except ImportError:
    from workflow.models import CVProfile, JobRecord, JobSearchPreferences

try:
    from src.workflow.mocks import mock_cv_mid_tech, mock_preferences_mid_tech
except ImportError:
    from workflow.mocks import mock_cv_mid_tech, mock_preferences_mid_tech


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "iterations" / "job_search_results.md"
INDEX_PATH = PROJECT_ROOT / "data" / "vector_store" / "faiss_minilm.index"
DOCSTORE_PATH = PROJECT_ROOT / "data" / "vector_store" / "docstore_minilm.json"
DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "vector_store" / "job_descriptions_minilm.json"

# retriever.search(cv, prefs) → list[JobRecord] (ranked by relevance)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load raw FAISS index + parallel docstore — graceful failure if index not present
try:
    index = faiss.read_index(str(INDEX_PATH), faiss.IO_FLAG_MMAP)
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    job_texts = [d["page_content"] for d in docstore]
    job_metadata = [d["metadata"] for d in docstore]

    assert len(job_texts) == index.ntotal, (
        f"Docstore/index mismatch: {len(job_texts):,} entries vs {index.ntotal:,} vectors"
    )
    logger.info(
        f"Loaded index: {index.ntotal:,} vectors | docstore: {len(docstore):,} entries"
    )
except FileNotFoundError:
    index = None  # type: ignore[assignment]
    job_texts = []
    job_metadata = []
    logger.warning(
        "FAISS index not found — download from Google Drive before running retrieval. "
        f"Expected: {INDEX_PATH}"
    )

# Load job_id → description lookup (built alongside the index by build_vector_store_minilm.py)
try:
    with open(DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
        job_descriptions: dict[str, str] = json.load(f)
    logger.info(f"Loaded descriptions lookup: {len(job_descriptions):,} entries")
except FileNotFoundError:
    job_descriptions = {}
    logger.warning(f"Descriptions lookup not found at {DESCRIPTIONS_PATH} — rebuild index to fix")


def serialize_cv_profile(cv: CVProfile) -> str:
    parts: list[str] = []

    if cv.experience_level:
        parts.append(f"Experience level: {cv.experience_level}")

    if cv.years_experience is not None:
        parts.append(f"Years of experience: {cv.years_experience}")

    if cv.skills:
        parts.append(f"Skills: {', '.join(cv.skills)}")

    if cv.tools:
        parts.append(f"Tools: {', '.join(cv.tools)}")

    if cv.industries:
        parts.append(f"Industries: {', '.join(cv.industries)}")

    if cv.job_titles_held:
        parts.append(f"Past roles: {', '.join(cv.job_titles_held)}")

    if cv.domain_keywords:
        parts.append(f"Domain knowledge: {', '.join(cv.domain_keywords)}")

    if cv.education_level:
        parts.append(f"Education: {cv.education_level}")

    if cv.field_of_study:
        parts.append(f"Field: {cv.field_of_study}")

    if cv.certifications:
        parts.append(f"Certifications: {', '.join(cv.certifications)}")

    if cv.languages:
        parts.append(f"Languages: {', '.join(cv.languages)}")

    if cv.current_location:
        parts.append(f"Current location: {cv.current_location}")

    return ". ".join(parts)


def serialize_preferences(pref: JobSearchPreferences) -> str:
    parts: list[str] = []

    parts.append(f"Target location: {pref.target_location}")
    parts.append(f"Work type: {pref.work_type}")
    parts.append(f"Employment type: {pref.employment_type}")
    parts.append(f"Willing to relocate: {pref.willing_to_relocate}")
    parts.append(f"Remote preference: {pref.remote_preference}")

    if pref.target_roles:
        parts.append(f"Target roles: {', '.join(pref.target_roles)}")

    if pref.industry_preference:
        parts.append(f"Preferred industries: {', '.join(pref.industry_preference)}")

    return ". ".join(parts)


def embed_profile_and_preferences(
    cv: CVProfile,
    pref: JobSearchPreferences,
) -> np.ndarray:
    """
    Returns a single embedding vector representing both
    the candidate's profile and their job preferences.
    """
    cv_text = serialize_cv_profile(cv)
    pref_text = serialize_preferences(pref)
    combined_text = f"Candidate profile: {cv_text}. Job preferences: {pref_text}"
    return model.encode(combined_text, convert_to_numpy=True).astype("float32")


def search_jobs(
    query_embedding: np.ndarray,
    index: faiss.Index,
    job_texts: list[str],
    job_metadata: Optional[list[dict[str, Any]]] = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Returns top_k most similar jobs using cosine similarity."""
    query_embedding = np.array(query_embedding).astype("float32")

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)  # type: ignore[call-arg]

    results = []
    for score, idx in zip(scores[0], indices[0]):
        job_info: dict[str, Any] = {
            "score": float(score),
            "job_description": job_texts[idx],
        }
        if job_metadata is not None:
            job_info.update(job_metadata[idx])
        results.append(job_info)

    logger.info(f"Search complete: {len(results)} results returned")
    return results


def write_results(results: list[dict[str, Any]], out_path: Path) -> None:
    """Write top-k results to a markdown file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Job Search Results\n\n")
        f.write("| # | Score | Title | Company | Location | Source |\n")
        f.write("|---|-------|-------|---------|----------|--------|\n")
        for i, r in enumerate(results, 1):
            score = f"{r['score']:.4f}"
            title = r.get("title", "N/A")
            company = r.get("company", "N/A")
            location = r.get("location", "N/A")
            source = r.get("source", "N/A")
            f.write(
                f"| {i} | {score} | {title} | {company} | {location} | {source} |\n"
            )
    logger.info(f"Results written to {out_path}")


def retrieve_jobs(
    cv: CVProfile,
    prefs: JobSearchPreferences,
    top_k: int = 20,
    source: Optional[str] = "kaggle",
) -> list[JobRecord]:
    """
    Public pipeline interface: CV + preferences → list[JobRecord].

    Wraps embed → search → convert to typed JobRecord objects.
    This is what reranker.py calls — NOT search_jobs() directly.

    Args:
        cv: Structured CVProfile from cv_profiler.
        prefs: Job search preferences from user input.
        top_k: Number of results to return (default 20 for reranker input).
        source: Filter by data source — "kaggle" (evaluation default), "arbeitnow",
                or None for the full index. Kaggle is the default to ensure
                evaluation uses only in-distribution data.

    Returns:
        list[JobRecord] sorted by cosine similarity, length <= top_k.
    """
    if index is None:
        raise RuntimeError(
            "FAISS index not loaded. Download from Google Drive and place at "
            f"{INDEX_PATH}"
        )

    query_embedding = embed_profile_and_preferences(cv, prefs)

    # Over-fetch to absorb source filtering and per-job chunk dedup.
    # Kaggle is 99.2% of index; +40 covers both without over-scanning.
    fetch_k = top_k + 40 if source else top_k + 20
    raw = search_jobs(
        query_embedding=query_embedding,
        index=index,
        job_texts=job_texts,
        job_metadata=job_metadata,
        top_k=fetch_k,
    )

    records: list[JobRecord] = []
    seen_job_ids: set[str] = set()
    for r in raw:
        if source and r.get("source") != source:
            continue
        job_id = str(r.get("job_id", ""))
        if job_id in seen_job_ids:
            continue
        seen_job_ids.add(job_id)
        records.append(
            JobRecord(
                job_id=job_id,
                title=str(r.get("title", "")),
                company=str(r.get("company", "")),
                description=job_descriptions.get(job_id, ""),
                location=r.get("location"),
                experience_level=r.get("experience_level"),
                work_type=r.get("work_type"),
                min_salary=float(r["min_salary"])
                if r.get("min_salary") is not None
                else None,
                max_salary=float(r["max_salary"])
                if r.get("max_salary") is not None
                else None,
                url=r.get("url"),
                skill_labels=r.get("skill_labels"),
                source=str(r.get("source", "kaggle")),
                score=float(r["score"]),
            )
        )
        if len(records) >= top_k:
            break

    logger.info(
        f"retrieve_jobs: {len(records)} JobRecords returned "
        f"(source={source!r}, top_k={top_k})"
    )
    return records


if __name__ == "__main__":
    query_embedding = embed_profile_and_preferences(
        mock_cv_mid_tech,
        mock_preferences_mid_tech,
    )

    if index is not None:
        results = search_jobs(
            query_embedding=query_embedding,
            index=index,
            job_texts=job_texts,
            job_metadata=job_metadata,
            top_k=20,
        )
    else:
        # Handle the error or initialize the index
        raise ValueError("Index has not been initialized.")

    write_results(results, RESULTS_FILE)
