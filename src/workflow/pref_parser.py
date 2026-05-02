"""
pref_parser.py — Parse freeform job preference text into JobSearchPreferences.
Also handles natural language CVProfile edits via apply_profile_edit().

Same Gemma + OpenRouter fallback pattern as cv_profiler. No caching — preferences
are cheap to re-parse and session-specific.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from src.workflow.models import CVProfile, JobSearchPreferences

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

MODEL_NAME = "gemma-4-31b-it"
_GEMMA_OPENROUTER_MODEL = "google/gemma-4-31b-it"
_GEMMA_TIMEOUT = 30

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _openrouter_client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=30)


def _call_gemma(prompt: str) -> dict:
    if not _client:
        raise ValueError("GOOGLE_API_KEY not set")

    def _do():
        response = _client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) > 1:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
        return json.loads(raw)

    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_do)
    try:
        return future.result(timeout=_GEMMA_TIMEOUT)
    except _FuturesTimeout:
        ex.shutdown(wait=False)
        raise TimeoutError("Gemma timed out")


def _call_openrouter(prompt: str) -> dict:
    client = _openrouter_client()
    response = client.chat.completions.create(
        model=_GEMMA_OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
    return json.loads(raw)


def _call(prompt: str) -> dict:
    try:
        return _call_gemma(prompt)
    except Exception as e:
        logger.warning(f"[pref-parser] Gemma failed ({e}) — falling back to OpenRouter")
        return _call_openrouter(prompt)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_preferences(text: str) -> JobSearchPreferences:
    """Parse freeform job preference text into a JobSearchPreferences object."""
    template = (PROMPTS_DIR / "pref_parser.md").read_text(encoding="utf-8")
    prompt = template.replace("{{USER_INPUT}}", text.strip())

    data = _call(prompt)

    data.setdefault("target_location", "United States")
    data.setdefault("work_type", "full-time")
    data.setdefault("employment_type", "full-time")
    data.setdefault("willing_to_relocate", False)
    data.setdefault("remote_preference", "flexible")
    data.setdefault("target_roles", [])
    data.setdefault("industry_preference", [])

    return JobSearchPreferences(**data)


def refine_preferences(existing: JobSearchPreferences, correction: str) -> JobSearchPreferences:
    """Merge a correction into existing preferences — preserves unchanged fields."""
    current = existing.model_dump_json(indent=2)
    prompt = f"""You are updating a user's job search preferences. Apply only what the user specifies — keep all other fields exactly as they are.

Current preferences:
{current}

User correction: {correction.strip()}

Return ONLY the updated preferences as valid JSON with the exact same schema and field names. Do not reset fields the user did not mention."""

    data = _call(prompt)
    # Fill any missing keys from existing to be safe
    existing_dict = existing.model_dump()
    for k, v in existing_dict.items():
        data.setdefault(k, v)
    return JobSearchPreferences(**data)


def apply_profile_edit(profile: CVProfile, instruction: str) -> CVProfile:
    """Apply a natural language edit instruction to a CVProfile."""
    current = profile.model_dump_json(indent=2)
    prompt = f"""You are editing a job candidate's profile. Apply the user's instruction to the JSON below.

Current profile:
{current}

Instruction: {instruction.strip()}

Return ONLY the updated profile as valid JSON with the exact same schema and field names. Do not add or remove fields."""

    data = _call(prompt)
    return CVProfile.model_validate(data)
