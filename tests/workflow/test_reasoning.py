"""
Tests for reasoning.py — contract validation, no live LLM calls.

Tests cover:
  - Output contract: returns ReasoningReport with correct structure
  - All 3 providers: gemini, deepseek, claude (mocked via _openrouter_client)
  - Postprocessing: fake missing skills removed, deduplication, cap at 3
  - Cache hit returns same result without LLM call
  - Retry on bad JSON / schema validation error
  - Input validation: empty jobs, >10 jobs raises ValueError
  - Truncation: descriptions > DESCRIPTION_CHAR_LIMIT chars truncated
  - Cache keys differ across providers (separate experiment results)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.workflow.mocks import mock_cv_mid_tech, mock_cv_senior_finance, mock_job_records
from src.workflow.models import CVProfile, JobRecord
from src.workflow.reasoning import (
    DESCRIPTION_CHAR_LIMIT,
    JobExplanation,
    ReasoningReport,
    _build_user_message,
    _cache_key,
    _cv_known_terms,
    _filter_missing_skills,
    _normalize_text_list,
    _postprocess,
    _truncate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ten_jobs() -> list[JobRecord]:
    return mock_job_records[:10]


@pytest.fixture
def valid_report(ten_jobs) -> ReasoningReport:
    """Minimal valid ReasoningReport for ten_jobs."""
    return ReasoningReport(
        cv_summary="Experienced mid-level software engineer with Python and React skills.",
        job_explanations=[
            JobExplanation(
                job_id=j.job_id,
                title=j.title,
                company=j.company,
                match_reason=f"Matches due to Python and cloud skills.",
                missing_skills=["Kubernetes"] if i == 0 else [],
            )
            for i, j in enumerate(ten_jobs)
        ],
        overall_missing_skills=["Kubernetes", "Go", "Terraform"],
        recommendation="Strong match for backend roles. Consider upskilling in Kubernetes.",
    )


def _make_openrouter_mock(report: ReasoningReport) -> MagicMock:
    """Mock openai ChatCompletion response returning a valid ReasoningReport."""
    mock_choice = MagicMock()
    mock_choice.message.content = report.model_dump_json()
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ── _truncate ─────────────────────────────────────────────────────────────────


def test_truncate_short_unchanged():
    text = "Short description"
    assert _truncate(text) == text


def test_truncate_long_description():
    text = "x" * (DESCRIPTION_CHAR_LIMIT + 100)
    result = _truncate(text)
    assert result.endswith("... [truncated]")
    assert len(result) == DESCRIPTION_CHAR_LIMIT + len("... [truncated]")


def test_truncate_exact_limit():
    text = "x" * DESCRIPTION_CHAR_LIMIT
    assert _truncate(text) == text


def test_truncate_none_returns_empty():
    assert _truncate("") == ""


# ── _build_user_message ───────────────────────────────────────────────────────


def test_build_user_message_contains_cv(ten_jobs):
    msg = _build_user_message(mock_cv_mid_tech, ten_jobs)
    assert "CVProfile" in msg
    assert "Python" in msg


def test_build_user_message_contains_all_job_ids(ten_jobs):
    msg = _build_user_message(mock_cv_mid_tech, ten_jobs)
    for job in ten_jobs:
        assert job.job_id in msg


def test_build_user_message_truncates_long_description():
    long_job = mock_job_records[0].model_copy(
        update={"description": "y" * (DESCRIPTION_CHAR_LIMIT + 500)}
    )
    msg = _build_user_message(mock_cv_mid_tech, [long_job])
    assert "[truncated]" in msg


# ── _cache_key ────────────────────────────────────────────────────────────────


def test_cache_key_deterministic(ten_jobs):
    k1 = _cache_key(mock_cv_mid_tech, ten_jobs)
    k2 = _cache_key(mock_cv_mid_tech, ten_jobs)
    assert k1 == k2


def test_cache_key_differs_on_different_cv(ten_jobs):
    k1 = _cache_key(mock_cv_mid_tech, ten_jobs)
    k2 = _cache_key(mock_cv_senior_finance, ten_jobs)
    assert k1 != k2


def test_cache_key_differs_on_different_jobs():
    k1 = _cache_key(mock_cv_mid_tech, mock_job_records[:5])
    k2 = _cache_key(mock_cv_mid_tech, mock_job_records[5:10])
    assert k1 != k2


# ── _normalize_text_list ──────────────────────────────────────────────────────


def test_normalize_deduplicates_case_insensitive():
    result = _normalize_text_list(["Python", "python", "PYTHON"])
    assert result == ["Python"]


def test_normalize_strips_whitespace():
    result = _normalize_text_list(["  Go  ", "Go"])
    assert result == ["Go"]


def test_normalize_removes_empty():
    result = _normalize_text_list(["", "  ", "Rust"])
    assert result == ["Rust"]


# ── _cv_known_terms ───────────────────────────────────────────────────────────


def test_cv_known_terms_includes_skills():
    known = _cv_known_terms(mock_cv_mid_tech)
    assert "python" in known


def test_cv_known_terms_includes_tools():
    known = _cv_known_terms(mock_cv_senior_finance)
    assert "netsuite" in known


def test_cv_known_terms_all_lowercase():
    known = _cv_known_terms(mock_cv_mid_tech)
    assert all(t == t.casefold() for t in known)


# ── _filter_missing_skills ────────────────────────────────────────────────────


def test_filter_removes_skills_in_cv():
    known = {"python", "react"}
    result = _filter_missing_skills(["Python", "Kubernetes", "React"], known)
    assert "Python" not in result
    assert "React" not in result
    assert "Kubernetes" in result


def test_filter_deduplicates():
    result = _filter_missing_skills(["Go", "go", "GO"], set())
    assert result == ["Go"]


def test_filter_removes_empty_strings():
    result = _filter_missing_skills(["", "  ", "Rust"], set())
    assert result == ["Rust"]


# ── _postprocess ──────────────────────────────────────────────────────────────


def test_postprocess_removes_fake_missing_skills(ten_jobs):
    report = ReasoningReport(
        cv_summary="Test",
        job_explanations=[
            JobExplanation(
                job_id=ten_jobs[0].job_id,
                title=ten_jobs[0].title,
                company=ten_jobs[0].company,
                match_reason="Good match",
                missing_skills=["Python", "COBOL"],
            )
        ],
        overall_missing_skills=["Python"],
        recommendation="Good",
    )
    result = _postprocess(report, mock_cv_mid_tech)
    assert "Python" not in result.job_explanations[0].missing_skills
    assert "COBOL" in result.job_explanations[0].missing_skills


def test_postprocess_caps_overall_missing_at_3(ten_jobs):
    report = ReasoningReport(
        cv_summary="Test",
        job_explanations=[
            JobExplanation(
                job_id=j.job_id,
                title=j.title,
                company=j.company,
                match_reason="Match",
                missing_skills=["Go", "Rust", "Erlang", "Haskell", "Julia"],
            )
            for j in ten_jobs[:2]
        ],
        overall_missing_skills=[],
        recommendation="OK",
    )
    result = _postprocess(report, mock_cv_mid_tech)
    assert len(result.overall_missing_skills) <= 3


def test_postprocess_strips_whitespace_from_fields(ten_jobs):
    report = ReasoningReport(
        cv_summary="  Summary with spaces  ",
        job_explanations=[
            JobExplanation(
                job_id=ten_jobs[0].job_id,
                title=ten_jobs[0].title,
                company=ten_jobs[0].company,
                match_reason="  Good match  ",
                missing_skills=[],
            )
        ],
        overall_missing_skills=[],
        recommendation="  Do it  ",
    )
    result = _postprocess(report, mock_cv_mid_tech)
    assert result.cv_summary == "Summary with spaces"
    assert result.job_explanations[0].match_reason == "Good match"
    assert result.recommendation == "Do it"


# ── _validate_explanations ───────────────────────────────────────────────────


def test_validate_explanations_wrong_count(ten_jobs):
    from src.workflow.reasoning import _validate_explanations

    report = ReasoningReport(
        cv_summary="x",
        job_explanations=[
            JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                           match_reason="ok", missing_skills=[])
            for j in ten_jobs[:8]
        ],
        overall_missing_skills=[],
        recommendation="ok",
    )
    with pytest.raises(ValueError, match="Expected 10"):
        _validate_explanations(report, ten_jobs)


def test_validate_explanations_unknown_job_id(ten_jobs):
    from src.workflow.reasoning import _validate_explanations

    explanations = [
        JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                       match_reason="ok", missing_skills=[])
        for j in ten_jobs
    ]
    explanations[0] = explanations[0].model_copy(update={"job_id": "FAKE_ID"})
    report = ReasoningReport(
        cv_summary="x", job_explanations=explanations,
        overall_missing_skills=[], recommendation="ok",
    )
    with pytest.raises(ValueError, match="unknown job_ids"):
        _validate_explanations(report, ten_jobs)


def test_validate_explanations_passes_on_valid(ten_jobs):
    from src.workflow.reasoning import _validate_explanations

    report = ReasoningReport(
        cv_summary="x",
        job_explanations=[
            JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                           match_reason="ok", missing_skills=[])
            for j in ten_jobs
        ],
        overall_missing_skills=[],
        recommendation="ok",
    )
    _validate_explanations(report, ten_jobs)


# ── _postprocess overall_missing always from per-job pool ─────────────────────


def test_postprocess_overall_missing_from_per_job_pool(ten_jobs):
    report = ReasoningReport(
        cv_summary="x",
        job_explanations=[
            JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                           match_reason="ok", missing_skills=["COBOL"])
            for j in ten_jobs[:1]
        ] + [
            JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                           match_reason="ok", missing_skills=[])
            for j in ten_jobs[1:]
        ],
        overall_missing_skills=["Python"],
        recommendation="ok",
    )
    result = _postprocess(report, mock_cv_mid_tech)
    assert "Python" not in result.overall_missing_skills
    assert "COBOL" in result.overall_missing_skills


def test_postprocess_overall_missing_capped_at_3(ten_jobs):
    report = ReasoningReport(
        cv_summary="x",
        job_explanations=[
            JobExplanation(job_id=j.job_id, title=j.title, company=j.company,
                           match_reason="ok",
                           missing_skills=["COBOL", "Fortran", "LISP", "Pascal", "Erlang"])
            for j in ten_jobs
        ],
        overall_missing_skills=[],
        recommendation="ok",
    )
    result = _postprocess(report, mock_cv_mid_tech)
    assert len(result.overall_missing_skills) == 3


# ── analyze_job_matches — gemini ──────────────────────────────────────────────


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_gemini_returns_report(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches

    result = analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=False)

    assert isinstance(result, ReasoningReport)
    assert result.cv_summary
    assert len(result.job_explanations) == len(ten_jobs)


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_gemini_uses_correct_model(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches, _MODEL_MAP

    analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=False)

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == _MODEL_MAP["gemini"]


# ── analyze_job_matches — deepseek ────────────────────────────────────────────


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_deepseek_returns_report(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches

    result = analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="deepseek", use_cache=False)

    assert isinstance(result, ReasoningReport)
    assert len(result.job_explanations) == len(ten_jobs)


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_deepseek_uses_correct_model(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches, _MODEL_MAP

    analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="deepseek", use_cache=False)

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == _MODEL_MAP["deepseek"]


# ── analyze_job_matches — claude ──────────────────────────────────────────────


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_claude_returns_report(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches

    result = analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="claude", use_cache=False)

    assert isinstance(result, ReasoningReport)


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_analyze_claude_uses_correct_model(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches, _MODEL_MAP

    analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="claude", use_cache=False)

    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == _MODEL_MAP["claude"]


# ── Cache behaviour ───────────────────────────────────────────────────────────


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_cache_hit_skips_llm(mock_or_factory, mock_cache, ten_jobs, valid_report):
    mock_cache.__contains__ = MagicMock(return_value=True)
    mock_cache.__getitem__ = MagicMock(return_value=valid_report.model_dump())

    from src.workflow.reasoning import analyze_job_matches

    result = analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=True)

    mock_or_factory.return_value.chat.completions.create.assert_not_called()
    assert isinstance(result, ReasoningReport)


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_cache_keys_differ_across_providers(mock_or_factory, mock_cache, ten_jobs, valid_report):
    """Gemini and Claude results must be stored under different cache keys."""
    stored_keys: list[str] = []

    def capture_setitem(key, val):
        stored_keys.append(key)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_openrouter_mock(valid_report)
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock(side_effect=capture_setitem)

    from src.workflow.reasoning import analyze_job_matches

    analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=True)
    analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="claude", use_cache=True)

    assert len(stored_keys) == 2
    assert stored_keys[0] != stored_keys[1]


# ── Retry behaviour ───────────────────────────────────────────────────────────


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_bad_json_triggers_retry(mock_or_factory, mock_cache, ten_jobs, valid_report):
    """Bad JSON on attempt 1 triggers retry — succeeds on attempt 2."""
    bad_choice = MagicMock()
    bad_choice.message.content = "not valid json {"
    bad_resp = MagicMock()
    bad_resp.choices = [bad_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        bad_resp,
        _make_openrouter_mock(valid_report),
    ]
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches

    with patch("src.workflow.reasoning.time.sleep"):
        result = analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=False)

    assert isinstance(result, ReasoningReport)
    assert mock_client.chat.completions.create.call_count == 2


@patch("src.workflow.reasoning._cache")
@patch("src.workflow.reasoning._openrouter_client")
def test_all_retries_exhausted_raises(mock_or_factory, mock_cache, ten_jobs):
    """All 3 retries fail → RuntimeError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    mock_or_factory.return_value = mock_client
    mock_cache.__contains__ = MagicMock(return_value=False)
    mock_cache.__setitem__ = MagicMock()

    from src.workflow.reasoning import analyze_job_matches

    with patch("src.workflow.reasoning.time.sleep"):
        with pytest.raises(RuntimeError, match="Reasoning failed"):
            analyze_job_matches(mock_cv_mid_tech, ten_jobs, provider="gemini", use_cache=False)


# ── Input validation ──────────────────────────────────────────────────────────


def test_empty_jobs_raises():
    from src.workflow.reasoning import analyze_job_matches

    with pytest.raises(ValueError, match="empty"):
        analyze_job_matches(mock_cv_mid_tech, [], use_cache=False)


def test_too_many_jobs_raises():
    from src.workflow.reasoning import analyze_job_matches

    eleven_jobs = mock_job_records[:10] + [
        mock_job_records[0].model_copy(update={"job_id": "extra_job"})
    ]
    with pytest.raises(ValueError, match="at most 10"):
        analyze_job_matches(mock_cv_mid_tech, eleven_jobs, use_cache=False)


# ── Fence stripping ───────────────────────────────────────────────────────────


def _apply_fence_strip(raw: str) -> str:
    """Mirrors the fence-strip logic in _call_openrouter."""
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
    return raw


def test_fence_strip_standard_with_closing():
    raw = '```json\n{"cv_summary": "test"}\n```'
    assert _apply_fence_strip(raw) == '{"cv_summary": "test"}'


def test_fence_strip_no_closing_backtick():
    raw = '```json\n{"cv_summary": "test"}'
    assert _apply_fence_strip(raw) == '{"cv_summary": "test"}'


def test_fence_strip_no_json_prefix():
    raw = '```\n{"cv_summary": "test"}\n```'
    assert _apply_fence_strip(raw) == '{"cv_summary": "test"}'


def test_fence_strip_plain_json_unchanged():
    raw = '{"cv_summary": "test"}'
    assert _apply_fence_strip(raw) == raw
