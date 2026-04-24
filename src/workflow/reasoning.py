


## 3) `src/workflow/reasoning.py`


from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field  

from .models import CVProfile, JobRecord


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "reasoning.md"
CACHE_DIR = Path(".cache") / "reasoning"
LOGIC_VERSION = "v1"


class JobExplanation(BaseModel):
    job_id: str
    title: str
    company: str
    match_reason: str
    missing_skills: list[str] = Field(default_factory=list)


class ReasoningReport(BaseModel):
    cv_summary: str
    job_explanations: list[JobExplanation]
    overall_missing_skills: list[str] = Field(default_factory=list)
    recommendation: str


def load_reasoning_prompt() -> str:
    """Load the reasoning prompt template from disk."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _normalize_text_list(values: list[str]) -> list[str]:
    """Normalize strings for safer comparison and deduplication."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not value:
            continue
        norm = value.strip()
        if not norm:
            continue
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(norm)

    return cleaned


def _cv_known_terms(cv: CVProfile) -> set[str]:
    """
    Build a set of terms already present in the CV so we do not report them
    as missing later.
    """
    terms: list[str] = []

    terms.extend(cv.skills)
    terms.extend(cv.certifications)
    terms.extend(cv.languages)
    terms.extend(cv.job_titles_held)
    terms.extend(cv.industries)
    terms.extend(cv.domain_keywords)
    terms.extend(cv.tools)

    if cv.field_of_study:
        terms.append(cv.field_of_study)
    if cv.current_location:
        terms.append(cv.current_location)
    if cv.education_level:
        terms.append(cv.education_level)
    if cv.experience_level:
        terms.append(cv.experience_level)

    return {term.strip().casefold() for term in terms if term and term.strip()}


def _serialize_cv(cv: CVProfile) -> dict[str, Any]:
    """Convert CVProfile to plain dict for prompt payload."""
    return cv.model_dump()


def _serialize_jobs(jobs: list[JobRecord]) -> list[dict[str, Any]]:
    """Convert JobRecord list to plain dicts for prompt payload."""
    return [job.model_dump() for job in jobs]


def _build_llm_messages(cv: CVProfile, jobs: list[JobRecord]) -> list[dict[str, str]]:
    """
    Build a simple system + user message payload for the LLM.
    Adjust this format later if your project uses a specific client SDK.
    """
    prompt = load_reasoning_prompt()

    payload = {
        "cv_profile": _serialize_cv(cv),
        "jobs": _serialize_jobs(jobs),
    }

    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _cache_key(cv: CVProfile, jobs: list[JobRecord]) -> str:
    """Create a stable cache key from prompt version + serialized inputs."""
    payload = {
        "logic_version": LOGIC_VERSION,
        "prompt_text": load_reasoning_prompt(),
        "cv_profile": _serialize_cv(cv),
        "jobs": _serialize_jobs(jobs),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    """Return the cache file path for a given cache key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _save_cache(key: str, report: ReasoningReport) -> None:
    """Save structured report to disk cache."""
    path = _cache_path(key)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _load_cache(key: str) -> ReasoningReport | None:
    """Load cached report if present."""
    path = _cache_path(key)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return ReasoningReport.model_validate(data)


def _filter_missing_skills_against_cv(
    raw_missing_skills: list[str],
    cv_known_terms: set[str],
) -> list[str]:
    """
    Remove any 'missing skill' that is already present in the CV.
    This enforces the teammate requirement that missing skills must
    actually be missing.
    """
    filtered: list[str] = []
    seen: set[str] = set()

    for skill in raw_missing_skills:
        if not skill:
            continue

        normalized = skill.strip()
        if not normalized:
            continue

        folded = normalized.casefold()
        if folded in cv_known_terms:
            continue
        if folded in seen:
            continue

        seen.add(folded)
        filtered.append(normalized)

    return filtered


def _postprocess_report(report: ReasoningReport, cv: CVProfile) -> ReasoningReport:
    """
    Enforce consistency:
    - remove fake missing skills already present in CV
    - deduplicate overall missing skills
    """
    known_terms = _cv_known_terms(cv)

    updated_job_explanations: list[JobExplanation] = []
    overall_pool: list[str] = []

    for item in report.job_explanations:
        cleaned_missing = _filter_missing_skills_against_cv(
            item.missing_skills,
            known_terms,
        )
        updated_item = JobExplanation(
            job_id=item.job_id,
            title=item.title,
            company=item.company,
            match_reason=item.match_reason.strip(),
            missing_skills=cleaned_missing,
        )
        updated_job_explanations.append(updated_item)
        overall_pool.extend(cleaned_missing)

    overall_missing = _normalize_text_list(overall_pool)

    if not overall_missing and report.overall_missing_skills:
        overall_missing = _filter_missing_skills_against_cv(
            report.overall_missing_skills,
            known_terms,
        )

    overall_missing = overall_missing[:3]

    return ReasoningReport(
        cv_summary=report.cv_summary.strip(),
        job_explanations=updated_job_explanations,
        overall_missing_skills=overall_missing,
        recommendation=report.recommendation.strip(),
    )


def _parse_llm_response(raw_text: str) -> ReasoningReport:
    """
    Parse strict JSON returned by the LLM.
    Raises ValueError if parsing fails.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    return ReasoningReport.model_validate(data)


def call_reasoning_llm(messages: list[dict[str, str]]) -> str:
    """
    Placeholder LLM call.

    Replace this with your actual project LLM client call.
    For example:
        response = client.responses.create(...)
        return response.output_text

    For now this raises a clear error so the integration point is obvious.
    """
    raise NotImplementedError(
        "Hook this function to your actual LLM client. "
        "It should return a raw JSON string matching ReasoningReport."
    )


def analyze_job_matches(cv: CVProfile, jobs: list[JobRecord]) -> ReasoningReport:
    """
    Main reasoning entry point.

    Input:
      - CVProfile
      - 10 JobRecords

    Output:
      - structured reasoning report
    """
    if not jobs:
        raise ValueError("jobs must not be empty")

    if len(jobs) > 10:
        raise ValueError("analyze_job_matches expects at most 10 jobs")

    key = _cache_key(cv, jobs)
    cached = _load_cache(key)
    if cached is not None:
        return cached

    messages = _build_llm_messages(cv, jobs)
    raw_response = call_reasoning_llm(messages)
    parsed_report = _parse_llm_response(raw_response)
    final_report = _postprocess_report(parsed_report, cv)

    _save_cache(key, final_report)
    return final_report


def report_to_pretty_json(report: ReasoningReport) -> str:
    """Helper for printing readable output in tests."""
    return report.model_dump_json(indent=2)