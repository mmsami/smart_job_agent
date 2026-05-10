"""
Shared post-retrieval filters applied consistently across all retrieval paths:
BM25 (baseline_bm25.py), FAISS eval (run_evaluation.py), FAISS live (job_search.py).
"""

SENIOR_EXCLUDE_EXP: frozenset[str] = frozenset({"entry level", "associate", "internship"})
SENIOR_EXCLUDE_TITLE: frozenset[str] = frozenset({"staff ", "junior", "jr.", "intern", "entry level"})
ENTRY_EXCLUDE_EXP: frozenset[str] = frozenset({
    "director",
    "executive",
    "c-suite",
    "senior",
    "mid-senior level",
})
ENTRY_EXCLUDE_TITLE: frozenset[str] = frozenset({
    "director",
    "vp ",
    "vice president",
    "chief ",
    "c-level",
    "head of",
    "partner",
    "senior ",
    " sr ",
    "lead ",
    "principal",
})


def passes_seniority_filter(job_exp: str, job_title: str, cv_level: str) -> bool:
    """Return False if job seniority clearly mismatches the CV level.

    Args:
        job_exp: experience_level from job metadata, already lowercased.
        job_title: job title, already lowercased.
        cv_level: cv_profile.experience_level, already lowercased.
    """
    if cv_level == "senior":
        if job_exp in SENIOR_EXCLUDE_EXP:
            return False
        if not job_exp and any(kw in job_title for kw in SENIOR_EXCLUDE_TITLE):
            return False
    elif cv_level == "entry":
        if job_exp in ENTRY_EXCLUDE_EXP:
            return False
        if not job_exp and any(kw in job_title for kw in ENTRY_EXCLUDE_TITLE):
            return False
    return True
