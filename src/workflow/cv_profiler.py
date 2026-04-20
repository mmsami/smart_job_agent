"""
cv_profiler.py — Step 2 of CV pipeline: raw text → structured CVProfile.

Two-step design (more reliable than one-shot LLM extraction):
  Step 1 (LLM):    Extract raw structured facts from CV text — jobs with dates,
                   education, skills, etc. No computation, just extraction.
  Step 2 (Python): Compute years_experience from job dates, classify experience_level,
                   normalize casing, deduplicate, validate logic.

Notes:
  - temperature=0 for deterministic LLM output.
  - Content-based retry: retries if jobs are missing or all job titles are placeholders.
  - Caches by hash of raw text (diskcache). Re-parsing same text = no API call.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from diskcache import Cache
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.workflow.models import CVProfile

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

_client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemma-4-31b-it"
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CURRENT_YEAR = datetime.now().year
MIN_YEAR = 1950
LOGIC_VERSION = "v10"  # bump when Python logic changes to invalidate stale cache

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache" / "cv_profiler"
_cache = Cache(str(CACHE_DIR))

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ── Step 1: LLM extraction ───────────────────────────────────────────────────


def _load_system_prompt() -> str:
    return (PROMPTS_DIR / "cv_profiler.md").read_text()


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _is_bad_output(data: dict) -> bool:
    """Return True if LLM output is missing critical fields — trigger retry.

    Two-stage check:
    1. Absolute vacuum: if all primary lists are empty, the LLM failed to extract anything.
       Handles fresh graduates, career changers, and non-traditional CVs — any populated
       section (education, certifications, languages, etc.) counts as a successful parse.
    2. Placeholder jobs: if jobs were extracted, at least one must have a real title.
    """
    if not isinstance(data, dict):
        return True

    # Stage 1: require at least one populated primary list
    critical_fields = ["jobs", "education", "skills", "certifications", "languages"]
    has_any_content = any(
        isinstance(data.get(f), list) and len(data.get(f, [])) > 0
        for f in critical_fields
    )
    if not has_any_content:
        return True

    # Stage 2: if jobs present, reject all-placeholder titles
    jobs = data.get("jobs", [])
    if isinstance(jobs, list) and len(jobs) > 0:
        valid_titles = [
            j for j in jobs
            if isinstance(j, dict)
            and str(j.get("title", "")).strip().lower() not in ("", "unknown", "n/a")
        ]
        if len(valid_titles) == 0:
            return True

    return False


def _call_llm(raw_text: str, system_prompt: str) -> dict:
    """Call Gemma 4 to extract raw CV facts. Retries on API error or bad output."""
    user_message = (
        f"Extract structured CV information from the following CV text:\n\n{raw_text}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # temperature=0 on first attempt; small nudge on retries to avoid same bad output
            temperature = 0.0 if attempt == 1 else 0.1
            response = _client.models.generate_content(
                model=MODEL_NAME,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            data = json.loads((response.text or "").strip())

            if not isinstance(data, dict):
                raise ValueError(
                    "LLM returned a list or scalar, expected a JSON object"
                )

            if _is_bad_output(data):
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Attempt {attempt}/{MAX_RETRIES}: bad output (empty primary sections or all-placeholder jobs), retrying..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                logger.warning(
                    f"All {MAX_RETRIES} attempts returned bad output — proceeding with incomplete data"
                )

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: invalid JSON — {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: API error — {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(
        f"CV profiling failed after {MAX_RETRIES} attempts — LLM returned unusable output"
    )


# ── Step 2: Python computation & normalization ───────────────────────────────


def _is_present(value) -> bool:
    """Return True if value means 'current job' (present/current/now/ongoing/None)."""
    if not value:
        return True
    return bool(re.search(r"\b(present|current|now|ongoing)\b", str(value).lower()))


def _safe_year(value, prefer_last: bool = False) -> Optional[int]:
    """Safely parse a year from LLM output. Handles ints, 'Jan 2018', '2018-2020', None.

    prefer_last=True returns the last year found in a range (use for end_year).
    """
    if value is None:
        return None
    matches = re.findall(r"\b(?:19|20)\d{2}\b", str(value))
    if not matches:
        return None
    return int(matches[-1]) if prefer_last else int(matches[0])


def _clean_jobs(jobs: list) -> list[dict]:
    """Defensive cleaning — drop malformed entries, deduplicate by (title, start, end)."""
    seen = set()
    cleaned = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        title = str(j.get("title", "")).strip()
        if not title:
            continue
        company = str(j.get("company", "")).strip().lower()
        key = (title.lower(), company, j.get("start_year"), j.get("end_year"))
        if key not in seen:
            seen.add(key)
            cleaned.append(j)
    return cleaned


def _compute_years_experience(jobs: list[dict]) -> int:
    """Compute total years experience using interval merging (handles overlapping roles)."""
    intervals = []
    for job in jobs:
        start = _safe_year(job.get("start_year"))
        end_raw = job.get("end_year")
        end = CURRENT_YEAR if _is_present(end_raw) else _safe_year(end_raw, prefer_last=True)

        if not start:
            logger.warning(
                f"  Job skipped — missing start year: {job.get('title', 'unknown title')}"
            )
            continue
        if end is None:
            logger.warning(
                f"  Job skipped — invalid end year: {job.get('title', 'unknown title')}"
            )
            continue
        if start > CURRENT_YEAR:
            logger.warning(f"  Future start year {start} for '{job.get('title', 'unknown')}' — clamping to {CURRENT_YEAR}")
        if end > CURRENT_YEAR:
            logger.warning(f"  Future end year {end} for '{job.get('title', 'unknown')}' — clamping to {CURRENT_YEAR}")
        start = max(MIN_YEAR, min(start, CURRENT_YEAR))
        end = max(MIN_YEAR, min(end, CURRENT_YEAR))
        if end >= start:
            intervals.append((start, end if end > start else start + 1))

    if not intervals:
        return 0

    # merge overlapping intervals to avoid double-counting parallel roles
    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return max(0, sum(end - start for start, end in merged))


def _classify_experience_level(years: int) -> Literal["entry", "mid", "senior"]:
    """Deterministic rule-based classification."""
    if years <= 2:
        return "entry"
    elif years <= 7:
        return "mid"
    else:
        return "senior"


# Known acronyms/tech terms that should not be title-cased
_KEEP_CASE = {
    "AWS",
    "GCP",
    "iOS",
    "macOS",
    "SQL",
    "NoSQL",
    "API",
    "APIs",
    "REST",
    "CI/CD",
    "DevOps",
    "ETL",
    "NLP",
    "ML",
    "AI",
    "HR",
    "CRM",
    "KPI",
    "ROI",
    "B2B",
    "B2C",
    "SaaS",
    "UI",
    "UX",
    "JSON",
    "HTML",
    "CSS",
    "PHP",
    "JS",
    "TS",
    "C#",
    "C++",
    "SEO",
    "SEM",
    "SRE",
    "QA",
    "SDET",
    "FPGA",
    "ASIC",
    "PaaS",
    "IaaS",
    "IT",
    "SAP",
    "UAT",
    "UML",
    "MQTT",
    "HFM",
    "ERP",
    "SOX",
    "GAAP",
    "VAT",
    "KYC",
    "AML",
    "IFRS",
    "CPA",
    "CFA",
    "MBA",
    "SCRUM",
    "OKR",
}
_KEEP_CASE_LOWER = {k.lower() for k in _KEEP_CASE}
_KEEP_CASE_LOOKUP = {k.lower(): k for k in _KEEP_CASE}  # O(1) lookup vs O(n) next() scan


def _smart_title(text: str) -> str:
    """Title-case a phrase but preserve known acronyms."""
    words = []
    for word in text.strip().split():
        words.append(_KEEP_CASE_LOOKUP.get(word.lower(), word.capitalize()))
    return " ".join(words)


def _normalize_list(items) -> list[str]:
    """Deduplicate and smart-title-case a list of strings (preserves acronyms like AWS, SQL)."""
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return []
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = _smart_title(item)
        key = cleaned.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _normalize_education(raw: Optional[str]) -> Optional[Literal["bachelor", "master", "phd"]]:
    """Map any education string to strict enum: bachelor | master | phd | None.
    Uses word boundaries to avoid false matches (e.g. 'taskmaster' → not 'master')."""
    if not raw:
        return None
    edu = raw.lower().strip()
    if re.search(r"\b(phd|ph\.d|doctorate|doctor)\b", edu):
        return "phd"
    if re.search(r"\b(master|masters|msc|mba|meng|m\.sc|m\.eng)\b", edu):
        return "master"
    if re.search(r"\b(bachelor|bachelors|bsc|ba|be|btech|b\.sc|b\.tech)\b", edu):
        return "bachelor"
    return None


def _build_profile(raw: dict) -> CVProfile:
    """Step 2: compute derived fields and build validated CVProfile."""
    raw_jobs = raw.get("jobs")
    jobs_list = raw_jobs if isinstance(raw_jobs, list) else []
    jobs = _clean_jobs(jobs_list)
    education = raw.get("education") if isinstance(raw.get("education"), list) else []

    years = _compute_years_experience(jobs)
    level = _classify_experience_level(years)

    # Extract best education level from education list
    edu_priority = {"phd": 3, "master": 2, "bachelor": 1}
    best_edu = None
    best_field = None
    best_score = 0
    for edu in education or []:
        raw_title = edu.get("degree") or edu.get("title") or ""
        normalized = _normalize_education(raw_title)
        score = edu_priority.get(normalized or "", 0)
        if score > best_score:
            best_score = score
            best_edu = normalized
            best_field = _smart_title(str(edu.get("field") or "")) or None

    # contact is extracted for logging/debugging only — intentionally not stored in CVProfile (privacy)
    contact = raw.get("contact") or {}
    if contact:
        present = [k for k, v in contact.items() if v]
        if present:
            logger.info(f"  Contact fields found: {present}")

    # Sanitize current_location: ensure None or valid string
    current_location = raw.get("current_location")
    if current_location and current_location not in ["None", "null", ""]:
        current_location = str(current_location).strip()
    else:
        current_location = None

    return CVProfile(
        skills=_normalize_list(raw.get("skills") or []),
        experience_level=level,
        years_experience=years,
        current_location=current_location,
        education_level=best_edu,
        field_of_study=best_field,
        certifications=_normalize_list(raw.get("certifications") or []),
        languages=_normalize_list(raw.get("languages") or []),
        job_titles_held=_normalize_list(
            [j.get("title", "") for j in jobs if j.get("title")]
        ),
        industries=_normalize_list(raw.get("industries") or []),
        domain_keywords=_normalize_list(raw.get("domain_keywords") or [])[:15],
        tools=_normalize_list(raw.get("tools") or []),
    )


# ── Public API ───────────────────────────────────────────────────────────────


def profile_cv(raw_text: str, use_cache: bool = True) -> CVProfile:
    """
    Parse raw CV text into a structured CVProfile.

    Step 1: LLM extracts raw facts (jobs with dates, education, skills).
    Step 2: Python computes years_experience, experience_level, normalizes.

    Args:
        raw_text: Raw extracted text from cv_reader.
        use_cache: If True, returns cached result for same text (by content hash).

    Returns:
        Validated CVProfile Pydantic object.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text is empty — cv_reader may have failed to extract text")

    cache_key = f"cv_profile_{LOGIC_VERSION}_{MODEL_NAME}_{_text_hash(raw_text)}"

    if use_cache and cache_key in _cache:
        logger.info("Cache hit — returning cached CVProfile")
        return CVProfile.model_validate(_cache[cache_key])

    logger.info(f"Profiling CV ({len(raw_text)} chars)...")
    system_prompt = _load_system_prompt()

    raw = _call_llm(raw_text, system_prompt)
    if not raw:
        logger.error("  LLM returned no usable output — profile will be incomplete")
    jobs_raw = raw.get("jobs")
    skills_raw = raw.get("skills")
    num_jobs = len(jobs_raw) if isinstance(jobs_raw, list) else "?"
    num_skills = len(skills_raw) if isinstance(skills_raw, list) else "?"
    logger.info(f"  Raw extracted: {num_jobs} jobs, {num_skills} skills")

    profile = _build_profile(raw)
    logger.info(
        f"  Profile: {profile.years_experience} yrs → {profile.experience_level}, edu={profile.education_level}"
    )

    if use_cache:
        _cache[cache_key] = profile.model_dump()

    return profile
