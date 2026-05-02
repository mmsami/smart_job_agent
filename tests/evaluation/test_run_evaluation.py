"""
Tests for run_evaluation.py — focused on perform_retrieval correctness.

Critical invariants:
  - All 4 methods return JobRecords with non-empty descriptions
  - faiss_raw reads descriptions from job_descriptions dict, not job_texts
  - get_raw_profile wraps raw text in skills field
  - Unknown method raises ValueError
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.run_evaluation import DEFAULT_PREFS, get_raw_profile, perform_retrieval
from src.workflow.models import CVProfile, JobRecord


# ── Fixtures ──────────────────────────────────────────────────────────────────

RAW_TEXT = "Experienced Python developer with 5 years in fintech and ML pipelines."

PARSED_PROFILE = CVProfile(
    skills=["Python", "SQL", "Machine Learning"],
    experience_level="mid",
    current_location="New York, NY",
    industries=["Fintech"],
    domain_keywords=["ML", "data pipelines"],
    tools=["pandas", "sklearn"],
    languages=["English"],
    certifications=[],
    job_titles_held=["Data Engineer"],
)

_JOB_RECORD = JobRecord(
    job_id="42",
    title="Data Engineer",
    company="Acme Corp",
    description="Build and maintain ETL pipelines using Python and Spark.",
    location="New York, NY",
    experience_level="mid",
    work_type="full-time",
    min_salary=90000.0,
    max_salary=130000.0,
    url=None,
    skill_labels="Python, Spark",
    source="kaggle",
    score=0.85,
)

# search_jobs returns raw dicts — description intentionally missing (comes from lookup)
_SEARCH_RESULT = {
    "job_id": "42",
    "title": "Data Engineer",
    "company": "Acme Corp",
    "job_description": "",  # page_content — not the real description
    "location": "New York, NY",
    "experience_level": "mid",
    "work_type": "full-time",
    "min_salary": 90000.0,
    "max_salary": 130000.0,
    "url": None,
    "skill_labels": "Python, Spark",
    "source": "kaggle",
    "score": 0.85,
}

_JOB_DESCRIPTIONS = {"42": "Build and maintain ETL pipelines using Python and Spark."}


def _make_bm25():
    bm25 = MagicMock()
    bm25.search.return_value = [_JOB_RECORD]
    return bm25


# ── get_raw_profile ───────────────────────────────────────────────────────────

def test_get_raw_profile_wraps_text_in_skills():
    profile = get_raw_profile(RAW_TEXT)
    assert profile.skills == [RAW_TEXT]


def test_get_raw_profile_has_required_fields():
    profile = get_raw_profile(RAW_TEXT)
    assert profile.experience_level == "mid"
    assert profile.industries == []
    assert profile.tools == []


# ── perform_retrieval: bm25_raw ───────────────────────────────────────────────

def test_bm25_raw_returns_records_with_descriptions():
    bm25 = _make_bm25()
    results = perform_retrieval("bm25_raw", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    assert len(results) > 0
    assert all(r.description for r in results), "bm25_raw: descriptions must be non-empty"


def test_bm25_raw_passes_raw_profile_to_bm25():
    bm25 = _make_bm25()
    perform_retrieval("bm25_raw", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    called_profile = bm25.search.call_args[0][0]
    assert called_profile.skills == [RAW_TEXT], "bm25_raw must pass raw text as skills"


# ── perform_retrieval: bm25_parsed ────────────────────────────────────────────

def test_bm25_parsed_returns_records_with_descriptions():
    bm25 = _make_bm25()
    results = perform_retrieval("bm25_parsed", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    assert len(results) > 0
    assert all(r.description for r in results), "bm25_parsed: descriptions must be non-empty"


def test_bm25_parsed_passes_structured_profile_to_bm25():
    bm25 = _make_bm25()
    perform_retrieval("bm25_parsed", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    called_profile = bm25.search.call_args[0][0]
    assert called_profile.skills == PARSED_PROFILE.skills


# ── perform_retrieval: faiss_raw ──────────────────────────────────────────────

@patch("src.evaluation.run_evaluation.job_descriptions", _JOB_DESCRIPTIONS)
@patch("src.evaluation.run_evaluation.faiss_lib.normalize_L2")
@patch("src.evaluation.run_evaluation.search_jobs", return_value=[_SEARCH_RESULT])
@patch("src.evaluation.run_evaluation._get_embed_model")
def test_faiss_raw_description_comes_from_lookup_not_job_texts(
    mock_embed, mock_search, mock_norm
):
    mock_embed.return_value.encode.return_value = np.zeros((1, 384), dtype="float32")
    bm25 = _make_bm25()

    results = perform_retrieval("faiss_raw", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)

    assert len(results) == 1
    # Must use job_descriptions lookup, not the empty job_description from search_jobs
    assert results[0].description == _JOB_DESCRIPTIONS["42"]


@patch("src.evaluation.run_evaluation.job_descriptions", _JOB_DESCRIPTIONS)
@patch("src.evaluation.run_evaluation.faiss_lib.normalize_L2")
@patch("src.evaluation.run_evaluation.search_jobs", return_value=[_SEARCH_RESULT])
@patch("src.evaluation.run_evaluation._get_embed_model")
def test_faiss_raw_descriptions_non_empty(mock_embed, mock_search, mock_norm):
    mock_embed.return_value.encode.return_value = np.zeros((1, 384), dtype="float32")
    bm25 = _make_bm25()

    results = perform_retrieval("faiss_raw", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    assert all(r.description for r in results), "faiss_raw: descriptions must be non-empty"


@patch("src.evaluation.run_evaluation.job_descriptions", {})
@patch("src.evaluation.run_evaluation.faiss_lib.normalize_L2")
@patch("src.evaluation.run_evaluation.search_jobs", return_value=[_SEARCH_RESULT])
@patch("src.evaluation.run_evaluation._get_embed_model")
def test_faiss_raw_missing_job_id_in_lookup_returns_empty_string(
    mock_embed, mock_search, mock_norm
):
    mock_embed.return_value.encode.return_value = np.zeros((1, 384), dtype="float32")
    bm25 = _make_bm25()

    results = perform_retrieval("faiss_raw", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    assert results[0].description == ""


# ── perform_retrieval: faiss_parsed ──────────────────────────────────────────

@patch("src.evaluation.run_evaluation.retrieve_jobs", return_value=[_JOB_RECORD])
def test_faiss_parsed_returns_records_with_descriptions(mock_retrieve):
    bm25 = _make_bm25()
    results = perform_retrieval("faiss_parsed", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    assert len(results) > 0
    assert all(r.description for r in results), "faiss_parsed: descriptions must be non-empty"


@patch("src.evaluation.run_evaluation.retrieve_jobs", return_value=[_JOB_RECORD])
def test_faiss_parsed_delegates_to_retrieve_jobs(mock_retrieve):
    bm25 = _make_bm25()
    perform_retrieval("faiss_parsed", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
    mock_retrieve.assert_called_once()


# ── perform_retrieval: unknown method ─────────────────────────────────────────

def test_unknown_method_raises_value_error():
    bm25 = _make_bm25()
    with pytest.raises(ValueError, match="Unknown method"):
        perform_retrieval("bm25_magic", RAW_TEXT, PARSED_PROFILE, DEFAULT_PREFS, bm25)
