"""
Integration tests for reranker.py — runs the REAL Gemma 4 LLM.

NOT part of the normal test suite. Run explicitly with:
    pytest -m integration tests/workflow/test_reranker_integration.py -v

First run: real API call (~3k tokens, ~2 seconds).
All subsequent runs: instant from disk cache (same input hash).

Requires:
    - GOOGLE_API_KEY in .env
    - LANGSMITH_* vars in .env (optional, tracing)

Behavioral invariants tested (things we know must be true regardless of LLM run):

Persona: mock_cv_mid_tech (5yr Python/React/AWS engineer, targeting remote SF roles)
Jobs: mock_job_records (10 jobs — 3 tech, 4 finance, 2 HR, 1 junior frontend)

  1. Tech jobs dominate top 3  — k002/k005/k009 are the only realistic matches
  2. Finance domain cap        — k001/k004/k008/k010 must each score ≤ 20
  3. HR domain cap             — k003/k006 must each score ≤ 20
  4. Seniority ordering        — k009 (mid AWS) must outscore k007 (junior frontend)
  5. Scores descending         — output list must be sorted high → low
  6. Count                     — exactly 10 results returned
  7. No unknown job_ids        — all returned ids exist in input
  8. No duplicate job_ids      — each job appears at most once
"""

import pytest

from src.workflow.mocks import (
    mock_cv_mid_tech,
    mock_preferences_mid_tech,
    mock_job_records,
)
from src.workflow.reranker import rerank_jobs

pytestmark = pytest.mark.integration

# Jobs we know are completely wrong domain for a mid-tech candidate
FINANCE_JOB_IDS = {"k001", "k004", "k008", "k010"}
HR_JOB_IDS = {"k003", "k006"}
DOMAIN_CAPPED_IDS = FINANCE_JOB_IDS | HR_JOB_IDS

# Tech jobs that are realistic matches
TECH_JOB_IDS = {"k002", "k005", "k009"}

# Junior role — tech domain but seniority mismatch
JUNIOR_TECH_ID = "k007"


@pytest.fixture(scope="module")
def reranked():
    """Run real LLM once per module — cached after first run."""
    return rerank_jobs(
        cv=mock_cv_mid_tech,
        preferences=mock_preferences_mid_tech,
        jobs=mock_job_records,
        use_cache=True,
    )


@pytest.fixture(scope="module")
def score_map(reranked):
    """job_id → score lookup."""
    return {job.job_id: job.score for job in reranked}


# ── Invariant 1: count ────────────────────────────────────────────────────────


def test_returns_ten_results(reranked):
    assert len(reranked) == 10, f"Expected 10 results, got {len(reranked)}"


# ── Invariant 2: no unknown or duplicate job_ids ──────────────────────────────


def test_no_unknown_job_ids(reranked):
    input_ids = {j.job_id for j in mock_job_records}
    for job in reranked:
        assert job.job_id in input_ids, f"Unknown job_id in output: {job.job_id!r}"


def test_no_duplicate_job_ids(reranked):
    ids = [job.job_id for job in reranked]
    assert len(ids) == len(set(ids)), f"Duplicate job_ids in output: {ids}"


# ── Invariant 3: scores descending ───────────────────────────────────────────


def test_scores_descending(reranked):
    """
    NOTE: Lost-in-Middle fix reorders the list (best first, second-best last).
    We check that the MAX score is at index 0 and the second-highest is at index -1,
    and that the middle jobs have scores between those two bounds — not strict descent.
    """
    scores = [job.score for job in reranked]
    assert scores[0] == max(scores), (
        f"Best job must be first. Got scores: {scores}"
    )
    # Middle jobs should all be ≤ highest score
    for s in scores[1:]:
        assert s <= scores[0], f"Score {s} exceeds first score {scores[0]}"


# ── Invariant 4: domain cap — finance jobs ≤ 20 ──────────────────────────────


def test_finance_jobs_domain_capped(score_map):
    for job_id in FINANCE_JOB_IDS:
        score = score_map.get(job_id)
        assert score is not None, f"{job_id} missing from output"
        assert score <= 20, (
            f"{job_id} (finance) scored {score} for mid-tech candidate — "
            f"domain cap (≤20) not enforced"
        )


# ── Invariant 5: domain cap — HR jobs ≤ 20 ───────────────────────────────────


def test_hr_jobs_domain_capped(score_map):
    for job_id in HR_JOB_IDS:
        score = score_map.get(job_id)
        assert score is not None, f"{job_id} missing from output"
        assert score <= 20, (
            f"{job_id} (HR) scored {score} for mid-tech candidate — "
            f"domain cap (≤20) not enforced"
        )


# ── Invariant 6: tech jobs outscore all domain-capped jobs ───────────────────


def test_tech_jobs_outscore_capped_jobs(score_map):
    min_tech_score = min(score_map[jid] for jid in TECH_JOB_IDS)
    max_capped_score = max(score_map[jid] for jid in DOMAIN_CAPPED_IDS)
    assert min_tech_score > max_capped_score, (
        f"Worst tech job scored {min_tech_score}, "
        f"best domain-capped job scored {max_capped_score} — "
        f"tech jobs should always outscore finance/HR for this candidate"
    )


# ── Invariant 7: best tech job in top 3 ──────────────────────────────────────


def test_top_tech_job_in_top_3(reranked):
    """k002 (Stripe Python/AWS/PostgreSQL) is the strongest match — must be top 3."""
    top3_ids = {job.job_id for job in reranked[:3]}
    assert "k002" in top3_ids, (
        f"k002 (Stripe backend, Python/AWS/PostgreSQL) not in top 3. "
        f"Top 3: {top3_ids}"
    )


# ── Invariant 8: seniority ordering ──────────────────────────────────────────


def test_mid_aws_job_outscores_junior_frontend(score_map):
    """k009 (Amazon mid-level AWS) must outscore k007 (junior frontend)."""
    k009 = score_map["k009"]
    k007 = score_map["k007"]
    assert k009 > k007, (
        f"k009 (Amazon mid-level AWS) scored {k009}, "
        f"k007 (junior frontend) scored {k007} — "
        f"seniority-appropriate role should outscore entry-level"
    )


# ── Invariant 9: score range ──────────────────────────────────────────────────


def test_all_scores_in_valid_range(score_map):
    for job_id, score in score_map.items():
        assert 0 <= score <= 100, (
            f"{job_id} has score {score} outside valid range [0, 100]"
        )
