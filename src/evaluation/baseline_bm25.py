"""
BM25 baseline retrieval — keyword search over title + description.

Retrieves top-k jobs using BM25, producing JobRecord output
identical to FAISS retrieval for fair evaluation comparison.

Usage:
    retriever = BM25Retriever()
    top_20 = retriever.search(parsed_cv, preferences, k=20)
"""

import hashlib
import json
import logging
import math
import os
import re
from typing import Any, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

from src.workflow.retrieval_filters import passes_seniority_filter

try:
    from src.workflow.models import CVProfile, JobRecord, JobSearchPreferences
except ImportError:
    from workflow.models import CVProfile, JobRecord, JobSearchPreferences

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_cleaned", "postings_cleaned.csv")
KAGGLE_CSV_SAMPLE = os.path.join(
    DATA_DIR, "kaggle_cleaned_sample", "postings_sample.csv"
)
ARBEITNOW_JSON = os.path.join(DATA_DIR, "arbeitnow", "arbeitnow_jobs.json")

# ── Stopwords ──────────────────────────────────────────────────────────
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "your",
    "can",
    "could",
    "has",
    "have",
    "him",
    "his",
    "how",
    "may",
    "must",
    "she",
    "should",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

# ── Optional fields shared between loaders and JobRecord builder ───────
# Avoids repeating the same list in multiple places
OPTIONAL_JOB_FIELDS = [
    "location",
    "experience_level",
    "work_type",
    "min_salary",
    "max_salary",
    "url",
    "skill_labels",
]


# ── Helpers ────────────────────────────────────────────────────────────


def _nan_to_none(v: Any) -> Any:
    """Convert pandas NaN to None for Optional JobRecord fields.

    pandas represents missing values as float NaN even in non-numeric columns.
    Pydantic models expect None for optional fields, not NaN.
    The isinstance guard prevents math.isnan() from crashing on non-float types.
    """
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return v


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords.

    Preserves + and # for tech tokens (C++, C#).
    Treats hyphens and dots as separators (full-time → full time, node.js → node js).
    """
    normalized = re.sub(r"[^\w\s+#]", " ", str(text).lower())
    return [t for t in normalized.split() if t and t not in STOPWORDS]


def _clean_df_row(row: dict) -> dict:
    """Normalize a raw DataFrame row into a clean job dict.

    Handles NaN → None, type coercion, and column name remapping
    (e.g. company_name → company) in one place for both data sources.
    """

    def safe_str(v) -> str:
        return str(v) if v is not None else ""

    def safe_float(v) -> Optional[float]:
        return float(v) if v not in (None, "") else None

    return {
        "title": safe_str(row.get("title")),
        "company": safe_str(row.get("company_name") or row.get("company")),
        "description": safe_str(row.get("description")),
        "location": row.get("location"),
        "experience_level": row.get("experience_level")
        or row.get("formatted_experience_level"),
        "work_type": row.get("work_type") or row.get("formatted_work_type"),
        "min_salary": safe_float(row.get("min_salary")),
        "max_salary": safe_float(row.get("max_salary")),
        "url": row.get("url") or row.get("application_url"),
        "skill_labels": row.get("skill_labels"),
    }


# ── Main class ─────────────────────────────────────────────────────────


class BM25Retriever:
    """BM25-based job retrieval (keyword baseline).

    Builds an index once at startup from Kaggle CSV + Arbeitnow JSON.
    Title is weighted 2x by repetition at index time; skills are weighted
    3x at query time via token repetition — BM25 has no native field weights.

    Args:
        k1: Term frequency saturation (default 1.5 — original BM25 paper value).
            Higher = more reward for repeated terms.
        b:  Length normalization (default 0.75 — original BM25 paper value).
            1.0 = full normalization, 0 = ignore length entirely.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.jobs: list[dict] = []
        self.corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.k1 = k1
        self.b = b
        self._load_and_index()

    # ── Index building ─────────────────────────────────────────────────

    def _load_and_index(self) -> None:
        """Load both data sources, deduplicate, and build BM25 index."""
        logger.info("Loading data for BM25 indexing...")

        self._load_kaggle()
        self._load_arbeitnow()
        self._deduplicate()
        self._build_index()

    def _load_kaggle(self) -> None:
        """Load Kaggle CSV (full dataset preferred, falls back to sample)."""
        csv_path = KAGGLE_CSV if os.path.exists(KAGGLE_CSV) else KAGGLE_CSV_SAMPLE
        if not os.path.exists(csv_path):
            logger.warning("No Kaggle CSV found — skipping")
            return

        df = pd.read_csv(csv_path)
        df = df.where(pd.notna(df), other=None)  # NaN → None before dict conversion

        for idx, row in enumerate(df.to_dict("records")):
            raw_id = row.get("job_id")
            job = _clean_df_row(row)
            job["job_id"] = str(raw_id) if raw_id is not None else f"kaggle_{idx}"
            job["source"] = "kaggle"
            self.jobs.append(job)

        logger.info(f"  Loaded {self._count_source('kaggle'):,} Kaggle jobs")

    def _load_arbeitnow(self) -> None:
        """Load Arbeitnow JSON."""
        if not os.path.exists(ARBEITNOW_JSON):
            logger.warning("No Arbeitnow JSON found — skipping")
            return

        with open(ARBEITNOW_JSON, encoding="utf-8") as f:
            raw_list = json.load(f)

        # Normalize via DataFrame so NaN handling is identical to Kaggle
        df = pd.DataFrame(raw_list)
        df = df.where(pd.notna(df), other=None)

        for idx, row in enumerate(df.to_dict("records")):
            raw_id = row.get("job_id")
            job = _clean_df_row(row)
            job["job_id"] = (
                str(raw_id)
                if raw_id is not None
                else f"arbeitnow_{hashlib.md5((job['title'] + job['company']).encode()).hexdigest()[:16]}"
            )
            job["source"] = row.get("source", "arbeitnow")
            self.jobs.append(job)

        logger.info(f"  Loaded {self._count_source('arbeitnow'):,} Arbeitnow jobs")

    def _deduplicate(self) -> None:
        """Remove duplicate (title, company) pairs, keeping first occurrence."""
        logger.info(f"Total jobs before dedup: {len(self.jobs):,}")

        seen: set[str] = set()
        deduped = []
        for job in self.jobs:
            key = f"{job['title'].lower()}|{job['company'].lower()}"
            if key not in seen:
                seen.add(key)
                deduped.append(job)

        removed = len(self.jobs) - len(deduped)
        if removed:
            logger.info(f"  Removed {removed:,} duplicate title+company pairs")

        self.jobs = deduped
        logger.info(f"Total jobs after dedup: {len(self.jobs):,}")

    def _build_index(self) -> None:
        """Tokenize all jobs and build BM25Okapi index."""
        logger.info("Building BM25 index...")

        for job in self.jobs:
            title_tokens = _tokenize(job["title"])
            desc_tokens = _tokenize(job["description"])[:100]
            # Title repeated twice — cheap field weighting since BM25 has none
            self.corpus.append(title_tokens * 2 + desc_tokens)

        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        logger.info(
            f"BM25 index built: {len(self.corpus):,} documents (k1={self.k1}, b={self.b})"
        )

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        cv_profile: CVProfile,
        preferences: JobSearchPreferences,
        k: int = 20,
        source: Optional[str] = "kaggle",
    ) -> list[JobRecord]:
        """Return top-k jobs ranked by BM25 score.

        Args:
            cv_profile:   Factual data extracted from CV (who the person is).
            preferences:  User job search preferences (what they want).
            k:            Number of results to return.
            source:       Filter by data source. "kaggle" (default), "arbeitnow", or None (all).

        Returns:
            List of JobRecord sorted by BM25 score descending.
        """
        if not self.bm25:
            raise RuntimeError("BM25 index not initialized")

        query_tokens = self._build_query_tokens(cv_profile, preferences)
        if not query_tokens:
            raise ValueError(
                "BM25 query is empty — CV profile and preferences have no usable tokens"
            )

        scores = self.bm25.get_scores(query_tokens)

        # Fetch k*3 candidates to absorb losses from source/dedup/seniority filters
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : k * 3
        ]

        return self._collect_results(top_indices, scores, cv_profile, source, k)

    def _build_query_tokens(
        self,
        cv_profile: CVProfile,
        preferences: JobSearchPreferences,
    ) -> list[str]:
        """Build weighted BM25 query from CV signals and search preferences.

        Token repetition is the only way to weight fields in BM25.
        Higher weight = more repetitions = stronger influence on score.
        """
        tokens: list[str] = []

        # (field_list, weight) — skills matter most, titles second, rest equal
        weighted_list_fields = [
            (cv_profile.skills, 3),  # strongest signal
            (cv_profile.job_titles_held, 2),  # role matching
            (cv_profile.certifications, 1),
            (cv_profile.industries, 1),
            (cv_profile.domain_keywords, 1),
            (cv_profile.tools, 1),
            (preferences.target_roles, 1),
            (preferences.industry_preference, 1),
        ]
        for field_list, weight in weighted_list_fields:
            for item in field_list:
                tokens.extend(_tokenize(item) * weight)

        # Single-value fields (weight 1)
        remote = preferences.remote_preference
        single_fields = [
            cv_profile.education_level,
            cv_profile.experience_level,
            cv_profile.field_of_study,
            preferences.work_type,
            remote if remote != "flexible" else None,  # "flexible" adds no signal
        ]
        for value in single_fields:
            if value:
                tokens.extend(_tokenize(value))

        # Languages — skip stopwords (e.g. "it" is both a language and a stopword)
        for lang in cv_profile.languages:
            lang_lower = str(lang).lower()
            if lang_lower not in STOPWORDS:
                tokens.append(lang_lower)

        return tokens

    def _collect_results(
        self,
        top_indices: list[int],
        scores,
        cv_profile: CVProfile,
        source: Optional[str],
        k: int,
    ) -> list[JobRecord]:
        """Apply filters, deduplicate, and build final JobRecord list."""
        seen_ids: set[str] = set()
        seen_title_company: set[str] = set()
        results: list[JobRecord] = []

        for idx in top_indices:
            if len(results) >= k:
                break

            job = self.jobs[idx]

            if not self._passes_all_filters(
                job, cv_profile, source, seen_ids, seen_title_company
            ):
                continue

            results.append(self._to_job_record(job, float(scores[idx])))

        return results

    def _passes_all_filters(
        self,
        job: dict,
        cv_profile: CVProfile,
        source: Optional[str],
        seen_ids: set,
        seen_title_company: set,
    ) -> bool:
        """Return True only if job passes source, dedup, and seniority checks."""

        # 1. Source filter
        if source is not None and job["source"] != source:
            return False

        # 2. Deduplication — two keys because either alone can miss cases:
        #    - same job reposted with a new ID → caught by title+company key
        #    - two different jobs at same company → caught by job_id key
        job_id = job["job_id"]
        title_company = f"{job['title'].lower()}|{job['company'].lower()}"

        if job_id in seen_ids or title_company in seen_title_company:
            return False

        seen_ids.add(job_id)
        seen_title_company.add(title_company)

        # 3. Seniority hard filter
        if not self._passes_seniority_filter(job, cv_profile):
            return False

        return True

    def _to_job_record(self, job: dict, score: float) -> JobRecord:
        """Convert raw job dict to a typed JobRecord, sanitizing optional fields."""
        return JobRecord(
            job_id=job["job_id"],
            title=job["title"],
            company=job["company"],
            description=job["description"],
            source=job["source"],
            score=score,
            # Apply _nan_to_none on all optional fields in one pass
            **{field: _nan_to_none(job.get(field)) for field in OPTIONAL_JOB_FIELDS},
        )

    def _passes_seniority_filter(self, job: dict, cv_profile: CVProfile) -> bool:
        exp_level = (_nan_to_none(job.get("experience_level")) or "").lower()
        title = (_nan_to_none(job.get("title")) or "").lower()
        cv_level = (cv_profile.experience_level or "").lower()
        return passes_seniority_filter(exp_level, title, cv_level)

    # ── Utilities ──────────────────────────────────────────────────────

    def _count_source(self, source: str) -> int:
        return sum(1 for j in self.jobs if j["source"] == source)


# ── Singleton wrapper ──────────────────────────────────────────────────

_retriever_instance: Optional[BM25Retriever] = None


def search_bm25(
    cv_profile: CVProfile,
    preferences: JobSearchPreferences,
    k: int = 20,
    source: Optional[str] = "kaggle",
) -> list[JobRecord]:
    """Singleton wrapper — builds index once, reuses on subsequent calls.

    Args:
        source: "kaggle" (default for eval), "arbeitnow", or None (full index).
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BM25Retriever()
    return _retriever_instance.search(cv_profile, preferences, k=k, source=source)
