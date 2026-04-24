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
    from src.workflow.models import CVProfile, JobSearchPreferences
except ImportError:
    from workflow.models import CVProfile, JobSearchPreferences

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

# retriever.search(cv, prefs) → list[JobRecord] (ranked by relevance)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load raw FAISS index + parallel docstore
index = faiss.read_index(str(INDEX_PATH))
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


if __name__ == "__main__":
    query_embedding = embed_profile_and_preferences(
        mock_cv_mid_tech,
        mock_preferences_mid_tech,
    )

    results = search_jobs(
        query_embedding=query_embedding,
        index=index,
        job_texts=job_texts,
        job_metadata=job_metadata,
        top_k=20,
    )

    write_results(results, RESULTS_FILE)
