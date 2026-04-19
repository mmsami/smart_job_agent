"""
BM25 baseline retrieval — keyword search over title + description.

This retrieves top 20 jobs using BM25, producing the same JobRecord output
as FAISS retrieval for fair evaluation comparison.

Usage:
    retriever = BM25Retriever()
    top_20 = retriever.search(parsed_cv, k=20)
"""

import json
import logging
import os
import re
from typing import Optional

import pandas as pd
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

try:
    from src.workflow.models import JobRecord, CVProfile, JobSearchPreferences
except ImportError:
    from workflow.models import JobRecord, CVProfile, JobSearchPreferences

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_cleaned", "postings_cleaned.csv")
KAGGLE_CSV_SAMPLE = os.path.join(DATA_DIR, "kaggle_cleaned_sample", "postings_sample.csv")
ARBEITNOW_JSON = os.path.join(DATA_DIR, "arbeitnow", "arbeitnow_jobs.json")

# ── Preprocessing ──────────────────────────────────────────────────────
# Standard English stopwords (common words filtered from indexing + queries)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
    "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "will", "with", "you", "your", "can",
    "could", "has", "have", "him", "his", "how", "may", "must", "she", "should",
    "what", "when", "where", "which", "who", "why"
}


def _tokenize_with_stopwords(text: str) -> list[str]:
    """Tokenize text and remove stopwords.

    Uses regex substitution to handle:
    - Punctuation attached to words: 'Python,' → 'python', 'Java.' → 'java'
    - Hyphenated terms: 'full-time' → ['full', 'time'] (improves cross-format matching)
    - Tech tokens: 'C++' and 'C#' preserved (+ and # kept in char set)
    - 'node.js' → ['node', 'js'] (dot treated as separator — acceptable tradeoff)
    """
    normalized = re.sub(r"[^\w\s+#]", " ", str(text).lower())
    return [t for t in normalized.split() if t and t not in STOPWORDS]


class BM25Retriever:
    """BM25-based job retrieval (keyword baseline).

    Uses standard preprocessing (lowercasing, stopword removal) and configurable
    BM25 hyperparameters (k1, b) for light tuning to ensure robust baseline.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Load data and build BM25 index.

        Args:
            k1: BM25 term frequency saturation parameter (default 1.5, range 1.2-2.0)
            b: BM25 length normalization parameter (default 0.75, range 0.5-0.75)
        """
        self.jobs = []  # list of dicts for fast access
        self.bm25 = None
        self.corpus = []  # tokenized documents for BM25
        self.k1 = k1
        self.b = b

        self._load_and_index()

    def _load_and_index(self):
        """Load Kaggle + Arbeitnow data, tokenize, and build BM25 index."""
        logger.info("Loading data for BM25 indexing...")

        # Load Kaggle (full dataset preferred; fall back to sample if not downloaded)
        csv_path = KAGGLE_CSV if os.path.exists(KAGGLE_CSV) else KAGGLE_CSV_SAMPLE
        if os.path.exists(csv_path):
            df_kaggle = pd.read_csv(csv_path)
            # Replace NaN with None before converting to dicts — avoids pandas type
            # ambiguity in pd.notna() checks and gives plain Python dicts.
            df_kaggle = df_kaggle.where(pd.notna(df_kaggle), other=None)
            for idx, row in enumerate(df_kaggle.to_dict("records")):
                raw_id = row.get("job_id")
                job_id = str(raw_id) if raw_id is not None else f"kaggle_{idx}"
                min_sal = row.get("min_salary")
                max_sal = row.get("max_salary")
                self.jobs.append({
                    "job_id": job_id,
                    "title": str(row["title"]) if row.get("title") is not None else "",
                    "company": str(row["company_name"]) if row.get("company_name") is not None else "",
                    "description": str(row["description"]) if row.get("description") is not None else "",
                    "location": row.get("location"),
                    "experience_level": row.get("formatted_experience_level"),
                    "work_type": row.get("formatted_work_type"),
                    "min_salary": float(min_sal) if min_sal not in (None, "") else None,
                    "max_salary": float(max_sal) if max_sal not in (None, "") else None,
                    "url": row.get("application_url"),
                    "skill_labels": row.get("skill_labels"),
                    "source": "kaggle",
                })
            logger.info(f"  Loaded {len([j for j in self.jobs if j['source'] == 'kaggle']):,} Kaggle jobs")

        # Load Arbeitnow
        if os.path.exists(ARBEITNOW_JSON):
            with open(ARBEITNOW_JSON, encoding="utf-8") as f:
                raw_list = json.load(f)
            for raw in raw_list:
                raw_id = raw.get("job_id")
                job_id = str(raw_id) if raw_id is not None else f"arbeitnow_{hash(str(raw.get('title', '')) + str(raw.get('company', '')))}"
                self.jobs.append({
                    "job_id": job_id,
                    "title": str(raw.get("title") or ""),
                    "company": str(raw.get("company") or ""),
                    "description": raw.get("description"),
                    "location": raw.get("location"),
                    "experience_level": raw.get("experience_level"),
                    "work_type": raw.get("work_type"),
                    "min_salary": raw.get("min_salary"),
                    "max_salary": raw.get("max_salary"),
                    "url": raw.get("url"),
                    "skill_labels": raw.get("skill_labels"),
                    "source": raw.get("source", "arbeitnow"),
                })
            logger.info(f"  Loaded {len([j for j in self.jobs if j['source'] == 'arbeitnow']):,} Arbeitnow jobs")

        logger.info(f"Total jobs: {len(self.jobs):,}")

        # Build BM25 index on title + description
        logger.info("Building BM25 index on title + description...")
        for job in self.jobs:
            # Tokenize with stopword removal: title (weighted more) + description
            title_tokens = _tokenize_with_stopwords(job["title"])
            desc_tokens = _tokenize_with_stopwords(job["description"])[:100]  # limit to first 100 words

            # Combine with title repeated for emphasis (increases weight)
            tokens = title_tokens + title_tokens + desc_tokens  # title appears twice for weight
            self.corpus.append(tokens)

        # Initialize BM25 with tuned hyperparameters
        self.bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built: {len(self.corpus)} documents (k1={self.k1}, b={self.b})")

    def search(self, cv_profile: CVProfile, preferences: JobSearchPreferences, k: int = 20, source: Optional[str] = None) -> list[JobRecord]:
        """
        Search for top-k jobs matching the CV profile + user preferences using BM25.

        Args:
            cv_profile: Factual data extracted from CV (who the person is)
            preferences: User-provided job search preferences (what they want)
            k: Number of results to return (default 20)
            source: Filter by dataset source (None=both, "kaggle"=Kaggle-only, "arbeitnow"=Arbeitnow-only)

        Returns:
            List of JobRecord sorted by BM25 score (descending)
        """
        if not self.bm25:
            raise RuntimeError("BM25 index not initialized")

        # ── CV Profile signals (who you are) ───────────────────────────
        query_tokens = []

        # Skills — primary signal (×3 repetition increases BM25 term frequency weight)
        for skill in cv_profile.skills:
            query_tokens.extend(_tokenize_with_stopwords(skill) * 3)

        # Certifications — strong signal (jobs explicitly require CPA, PMP, etc.)
        for cert in cv_profile.certifications:
            query_tokens.extend(_tokenize_with_stopwords(cert))

        # Past job titles — role matching signal (×2 for emphasis)
        for title in cv_profile.job_titles_held:
            query_tokens.extend(_tokenize_with_stopwords(title) * 2)

        # Industries — domain experience signal
        for industry in cv_profile.industries:
            query_tokens.extend(_tokenize_with_stopwords(industry))

        # Languages — some jobs explicitly require language fluency
        for lang in cv_profile.languages:
            lang_lower = str(lang).lower()
            if lang_lower not in STOPWORDS:
                query_tokens.append(lang_lower)

        # Education level — maps to degree keywords in job descriptions
        if cv_profile.education_level:
            edu_lower = cv_profile.education_level.lower()
            if edu_lower not in STOPWORDS:
                query_tokens.append(edu_lower)

        # Experience level — seniority signal from CV
        exp_lower = cv_profile.experience_level.lower()
        if exp_lower not in STOPWORDS:
            query_tokens.append(exp_lower)

        # Domain keywords — specific professional terms from CV work history
        # (GAAP, IFRS, SOX, reconciliation, audit — appear verbatim in job descriptions)
        for kw in cv_profile.domain_keywords:
            query_tokens.extend(_tokenize_with_stopwords(kw))

        # Specific tools — software beyond broad skills (NetSuite, Oracle, QuickBooks)
        for tool in cv_profile.tools:
            query_tokens.extend(_tokenize_with_stopwords(tool))

        # Field of study — maps to "degree in Accounting required" phrases
        if cv_profile.field_of_study:
            query_tokens.extend(_tokenize_with_stopwords(cv_profile.field_of_study))

        # ── JobSearchPreferences signals (what you want) ───────────────
        # Work type preference (full-time, remote, etc.)
        if preferences.work_type:
            query_tokens.extend(_tokenize_with_stopwords(preferences.work_type))

        # Target roles — specific roles user is aiming for
        for role in preferences.target_roles:
            query_tokens.extend(_tokenize_with_stopwords(role))

        # Industry preference — industries user wants to work in
        for industry in preferences.industry_preference:
            query_tokens.extend(_tokenize_with_stopwords(industry))

        # Remote preference — "hybrid", "remote", "onsite"
        if preferences.remote_preference != "flexible":
            remote_lower = str(preferences.remote_preference).lower()
            if remote_lower not in STOPWORDS:
                query_tokens.append(remote_lower)

        if not query_tokens:
            raise ValueError("BM25 query is empty — CV profile and preferences have no usable tokens")

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score, fetch more than k to allow for filter + dedup losses
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 3]

        # Build results with deduplication + seniority filter
        seen_job_ids: set = set()
        seen_title_company: set = set()
        results = []

        for idx in top_indices:
            if len(results) >= k:
                break

            job = self.jobs[idx]

            # ── Source filter ───────────────────────────────────────────
            if source is not None and job["source"] != source:
                continue

            # ── Deduplication ───────────────────────────────────────────
            # Primary: job_id. Secondary: title+company (catches same posting with diff IDs)
            dedup_key = job["job_id"]
            title_company_key = f"{str(job['title']).lower()}|{str(job['company']).lower()}"
            if dedup_key in seen_job_ids or title_company_key in seen_title_company:
                continue
            seen_job_ids.add(dedup_key)
            seen_title_company.add(title_company_key)

            # ── Seniority hard filter ───────────────────────────────────
            if not self._passes_seniority_filter(job, cv_profile):
                continue

            results.append(
                JobRecord(
                    job_id=job["job_id"],
                    title=job["title"],
                    company=job["company"],
                    description=job["description"],
                    location=job.get("location"),
                    experience_level=job.get("experience_level"),
                    work_type=job.get("work_type"),
                    min_salary=job.get("min_salary"),
                    max_salary=job.get("max_salary"),
                    url=job.get("url"),
                    skill_labels=job.get("skill_labels"),
                    source=job["source"],
                    score=float(scores[idx]),
                )
            )

        return results

    # ── Seniority filter helpers ────────────────────────────────────────
    _SENIOR_EXCLUDE_EXP = {"entry level", "associate", "internship"}
    _SENIOR_EXCLUDE_TITLE = {"staff ", "junior", "jr.", "intern", "entry level"}
    _ENTRY_EXCLUDE_EXP = {"director", "executive", "c-suite"}
    _ENTRY_EXCLUDE_TITLE = {"director", "vp ", "vice president", "chief ", "c-level", "head of", "partner"}

    def _passes_seniority_filter(self, job: dict, cv_profile: CVProfile) -> bool:
        """Hard seniority filter — skips clearly mismatched experience levels."""
        exp_level = (job.get("experience_level") or "").lower()
        title = (job.get("title") or "").lower()
        cv_level = (cv_profile.experience_level or "").lower()

        if cv_level == "senior":
            # Exclude entry/internship by metadata
            if exp_level in self._SENIOR_EXCLUDE_EXP:
                return False
            # Exclude obvious junior titles when metadata is missing
            if not exp_level and any(kw in title for kw in self._SENIOR_EXCLUDE_TITLE):
                return False

        elif cv_level == "entry":
            # Exclude director/executive by metadata
            if exp_level in self._ENTRY_EXCLUDE_EXP:
                return False
            # Exclude obvious senior titles when metadata is missing
            if not exp_level and any(kw in title for kw in self._ENTRY_EXCLUDE_TITLE):
                return False

        return True


# ── Standalone function for easy testing ───────────────────────────────
_retriever_instance: Optional[BM25Retriever] = None


def search_bm25(cv_profile: CVProfile, preferences: JobSearchPreferences, k: int = 20, source: Optional[str] = None) -> list[JobRecord]:
    """Singleton wrapper — creates retriever on first call, reuses on subsequent calls.

    Args:
        source: Filter by dataset source (None=both, "kaggle"=Kaggle-only, "arbeitnow"=Arbeitnow-only)
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = BM25Retriever()
    return _retriever_instance.search(cv_profile, preferences, k=k, source=source)
