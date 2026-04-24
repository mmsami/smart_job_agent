"""
Tests for reranker.py — contract validation, no live LLM calls.

Tests cover:
  - Output contract: returns list[JobRecord], len <= 10
  - Scores updated to LLM-assigned values (not original search scores)
  - All returned job_ids are from the input set
  - "Lost in the Middle" fix applied (best first, second-best last)
  - Cache hit returns same result
  - Retry and error handling (bad JSON, missing fields)
  - Truncation: descriptions > 2500 chars are truncated before LLM call
  - Empty input raises ValueError
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.workflow.models import CVProfile, JobSearchPreferences, JobRecord
from src.workflow.mocks import (
    mock_cv_mid_tech,
    mock_preferences_mid_tech,
    mock_job_records,
)
from src.workflow.reranker import (
    _apply_lost_in_middle_fix,
    _build_user_message,
    _input_hash,
    _truncate_description,
    _validate_output,
    DESCRIPTION_CHAR_LIMIT,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ten_jobs() -> list[JobRecord]:
    """First 10 mock job records."""
    return mock_job_records[:10]


@pytest.fixture
def valid_llm_response(ten_jobs) -> dict:
    """Minimal valid LLM response for ten_jobs."""
    return {
        "reranked_jobs": [
            {"job_id": j.job_id, "score": 90 - i * 5, "reasoning": f"Reason {i}"}
            for i, j in enumerate(ten_jobs[:10])
        ]
    }


# ── _truncate_description ─────────────────────────────────────────────────────


def test_truncate_short_description():
    text = "Short description"
    assert _truncate_description(text) == text


def test_truncate_long_description():
    text = "x" * (DESCRIPTION_CHAR_LIMIT + 100)
    result = _truncate_description(text)
    assert len(result) <= DESCRIPTION_CHAR_LIMIT + len("... [truncated]")
    assert result.endswith("... [truncated]")


def test_truncate_exact_limit():
    text = "x" * DESCRIPTION_CHAR_LIMIT
    assert _truncate_description(text) == text


# ── _build_user_message ───────────────────────────────────────────────────────


def test_build_user_message_contains_cv(ten_jobs):
    msg = _build_user_message(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs)
    assert "CVProfile" in msg
    assert "Python" in msg  # mid_tech skill


def test_build_user_message_contains_preferences(ten_jobs):
    msg = _build_user_message(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs)
    assert "Job Search Preferences" in msg
    assert "San Francisco" in msg


def test_build_user_message_contains_all_job_ids(ten_jobs):
    msg = _build_user_message(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs)
    for job in ten_jobs:
        assert job.job_id in msg


def test_build_user_message_truncates_long_description():
    long_job = mock_job_records[0].model_copy(
        update={"description": "y" * (DESCRIPTION_CHAR_LIMIT + 500)}
    )
    msg = _build_user_message(mock_cv_mid_tech, mock_preferences_mid_tech, [long_job])
    assert "[truncated]" in msg


# ── _validate_output ──────────────────────────────────────────────────────────


def test_validate_output_valid(ten_jobs, valid_llm_response):
    input_ids = {j.job_id for j in ten_jobs}
    result = _validate_output(valid_llm_response, input_ids, min_results=10)
    assert len(result) == 10
    assert all("job_id" in r and "score" in r for r in result)


def test_validate_output_sorted_descending(ten_jobs):
    # Feed unsorted scores — output must be sorted descending
    input_ids = {j.job_id for j in ten_jobs}
    shuffled = [
        {"job_id": j.job_id, "score": float(i * 3), "reasoning": "x"}
        for i, j in enumerate(ten_jobs)  # ascending order
    ]
    result = _validate_output({"reranked_jobs": shuffled}, input_ids, min_results=10)
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_validate_output_missing_key():
    with pytest.raises(ValueError, match="Missing or empty"):
        _validate_output({"wrong_key": []}, {"k001"}, min_results=1)


def test_validate_output_too_few_results():
    data = {"reranked_jobs": [{"job_id": "k001", "score": 80, "reasoning": "x"}]}
    with pytest.raises(ValueError, match="Expected at least"):
        _validate_output(data, {"k001", "k002"}, min_results=2)


def test_validate_output_unknown_job_id():
    data = {"reranked_jobs": [{"job_id": "UNKNOWN_ID", "score": 80, "reasoning": "x"}]}
    with pytest.raises(ValueError, match="Unknown job_id"):
        _validate_output(data, {"k001", "k002"}, min_results=1)


def test_validate_output_duplicate_job_id():
    data = {"reranked_jobs": [
        {"job_id": "k001", "score": 80, "reasoning": "x"},
        {"job_id": "k001", "score": 70, "reasoning": "y"},
    ]}
    with pytest.raises(ValueError, match="Duplicate job_id"):
        _validate_output(data, {"k001"}, min_results=1)


def test_validate_output_non_numeric_score():
    data = {"reranked_jobs": [{"job_id": "k001", "score": "high", "reasoning": "x"}]}
    with pytest.raises(ValueError, match="score must be numeric"):
        _validate_output(data, {"k001"}, min_results=1)


def test_validate_output_missing_score():
    data = {"reranked_jobs": [{"job_id": "k001"}]}  # no score
    with pytest.raises(ValueError, match="missing job_id or score"):
        _validate_output(data, {"k001"}, min_results=1)


def test_validate_output_non_dict():
    with pytest.raises(ValueError, match="non-dict"):
        _validate_output(["not", "a", "dict"], {"k001"}, min_results=1)


# ── _apply_lost_in_middle_fix ─────────────────────────────────────────────────


def test_lost_in_middle_fix_order(ten_jobs):
    result = _apply_lost_in_middle_fix(ten_jobs)
    assert result[0].job_id == ten_jobs[0].job_id, "Best job must be first"
    assert result[-1].job_id == ten_jobs[1].job_id, "Second-best job must be last"
    assert len(result) == len(ten_jobs)


def test_lost_in_middle_fix_single_job():
    single = mock_job_records[:1]
    result = _apply_lost_in_middle_fix(single)
    assert result == single


def test_lost_in_middle_fix_two_jobs():
    two = mock_job_records[:2]
    result = _apply_lost_in_middle_fix(two)
    assert result[0].job_id == two[0].job_id
    assert result[1].job_id == two[1].job_id


def test_lost_in_middle_fix_preserves_all_jobs(ten_jobs):
    result = _apply_lost_in_middle_fix(ten_jobs)
    assert {j.job_id for j in result} == {j.job_id for j in ten_jobs}


# ── _input_hash ───────────────────────────────────────────────────────────────


def test_input_hash_deterministic():
    h1 = _input_hash(mock_cv_mid_tech, mock_preferences_mid_tech, mock_job_records[:5])
    h2 = _input_hash(mock_cv_mid_tech, mock_preferences_mid_tech, mock_job_records[:5])
    assert h1 == h2


def test_input_hash_differs_on_different_input():
    h1 = _input_hash(mock_cv_mid_tech, mock_preferences_mid_tech, mock_job_records[:5])
    h2 = _input_hash(mock_cv_mid_tech, mock_preferences_mid_tech, mock_job_records[5:])
    assert h1 != h2


# ── rerank_jobs (mocked LLM) ──────────────────────────────────────────────────


def _make_mock_response(job_ids: list[str]) -> MagicMock:
    """Build a mock google.genai response that returns a valid reranked_jobs list."""
    reranked = [
        {"job_id": jid, "score": 90 - i * 5, "reasoning": f"Reason for {jid}"}
        for i, jid in enumerate(job_ids)
    ]
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"reranked_jobs": reranked})
    return mock_resp


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_returns_job_records(mock_client, mock_cache, ten_jobs):
    job_ids = [j.job_id for j in ten_jobs]
    mock_client.models.generate_content.return_value = _make_mock_response(job_ids)
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    assert isinstance(results, list)
    assert len(results) == 10
    assert all(isinstance(r, JobRecord) for r in results)


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_scores_updated(mock_client, mock_cache, ten_jobs):
    job_ids = [j.job_id for j in ten_jobs]
    original_scores = {j.job_id: j.score for j in ten_jobs}
    mock_client.models.generate_content.return_value = _make_mock_response(job_ids)
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    # LLM scores (90, 85, 80...) differ from original search scores
    for result in results:
        assert result.score != original_scores.get(result.job_id) or result.score in [90 - i * 5 for i in range(10)]


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_job_ids_from_input(mock_client, mock_cache, ten_jobs):
    job_ids = [j.job_id for j in ten_jobs]
    input_ids = set(job_ids)
    mock_client.models.generate_content.return_value = _make_mock_response(job_ids)
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    for r in results:
        assert r.job_id in input_ids


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_lost_in_middle_applied(mock_client, mock_cache, ten_jobs):
    """Second-best job (index 1 in LLM output) must be last in returned list."""
    job_ids = [j.job_id for j in ten_jobs]
    mock_client.models.generate_content.return_value = _make_mock_response(job_ids)
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    # After Lost-in-Middle fix: first job in LLM output → first result
    assert results[0].job_id == job_ids[0]
    # Second job in LLM output → last result
    assert results[-1].job_id == job_ids[1]


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_retry_on_bad_json(mock_client, mock_cache, ten_jobs):
    job_ids = [j.job_id for j in ten_jobs]
    bad_resp = MagicMock()
    bad_resp.text = "not valid json {"
    good_resp = _make_mock_response(job_ids)

    mock_client.models.generate_content.side_effect = [bad_resp, good_resp]
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    with patch("src.workflow.reranker.time.sleep"):
        results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    assert len(results) == 10
    assert mock_client.models.generate_content.call_count == 2


def test_rerank_empty_input_raises():
    from src.workflow.reranker import rerank_jobs

    with pytest.raises(ValueError, match="empty"):
        rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, [], use_cache=False)


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_cache_hit_skips_llm(mock_client, mock_cache, ten_jobs):
    cached = [j.model_dump() for j in ten_jobs[:10]]
    mock_cache.__contains__ = MagicMock(return_value=True)
    mock_cache.__getitem__ = MagicMock(return_value=cached)

    from src.workflow.reranker import rerank_jobs

    results = rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=True)

    mock_client.models.generate_content.assert_not_called()
    assert len(results) == 10


@patch("src.workflow.reranker._cache")
@patch("src.workflow.reranker._client")
def test_rerank_single_call_to_llm(mock_client, mock_cache, ten_jobs):
    """Verify exactly one LLM call for any number of input jobs (batch design)."""
    job_ids = [j.job_id for j in ten_jobs]
    mock_client.models.generate_content.return_value = _make_mock_response(job_ids)
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reranker import rerank_jobs

    rerank_jobs(mock_cv_mid_tech, mock_preferences_mid_tech, ten_jobs, use_cache=False)

    assert mock_client.models.generate_content.call_count == 1
