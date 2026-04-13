"""
Shared document schema for all job data sources.

Both fetch_arbeitnow.py and build_vector_store.py produce/consume JobDocument.
This is the single source of truth for field names, types, and nullability.

Field mapping:
    Kaggle field                → JobDocument field
    ─────────────────────────────────────────────────
    job_id                      → job_id
    title                       → title
    company_name                → company
    description                 → description  (HTML already stripped by parse_kaggle.py)
    location                    → location
    formatted_experience_level  → experience_level
    formatted_work_type         → work_type
    min_salary                  → min_salary
    max_salary                  → max_salary
    application_url             → url
    skill_labels                → skill_labels
    source = "kaggle"           → source

    Arbeitnow field             → JobDocument field
    ─────────────────────────────────────────────────
    slug                        → job_id
    company_name                → company
    title                       → title
    description                 → description  (strip HTML before storing)
    location                    → location
    remote                      → work_type    ("Remote" if True, else job_types[0])
    job_types[0]                → work_type
    tags (join ", ")            → skill_labels
    url                         → url
    (absent)                    → experience_level = None
    (absent)                    → min_salary = None
    (absent)                    → max_salary = None
    source = "arbeitnow"        → source
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator


class JobDocument(BaseModel):
    """Unified job posting schema used across all pipeline stages."""

    # ── Identity ───────────────────────────────────────────────────────
    job_id: str
    title: str
    company: str

    # ── Content (used for embedding) ───────────────────────────────────
    description: str  # plain text, HTML stripped
    skill_labels: Optional[str] = None  # comma-separated string or None

    # ── Metadata (stored but not embedded) ────────────────────────────
    location: Optional[str] = None
    experience_level: Optional[str] = None  # e.g. "Entry level", "Mid-Senior level"
    work_type: Optional[str] = None         # e.g. "Full-time", "Remote"
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    url: Optional[str] = None
    source: str  # "kaggle" or "arbeitnow"

    @field_validator("job_id", "title", "company", "description", mode="before")
    @classmethod
    def must_be_non_empty(cls, v: object) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Required string field must be non-empty")
        return v.strip()

    @field_validator("source", mode="before")
    @classmethod
    def valid_source(cls, v: object) -> str:
        if v not in ("kaggle", "arbeitnow"):
            raise ValueError(f"source must be 'kaggle' or 'arbeitnow', got {v!r}")
        return v

    def to_metadata(self) -> dict:
        """
        Return the metadata dict stored alongside the FAISS vector.
        All fields except description and skill_labels (those go in page_content).
        """
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "experience_level": self.experience_level,
            "work_type": self.work_type,
            "min_salary": self.min_salary,
            "max_salary": self.max_salary,
            "url": self.url,
            "source": self.source,
        }

    def to_page_content_prefix(self) -> str:
        """
        Signal-boosting prefix prepended to each chunk before embedding.
        Format: "{title} at {company}. {skill_labels}. "
        Omits skill_labels segment if null.
        """
        skills_part = f"{self.skill_labels}. " if self.skill_labels else ""
        return f"{self.title} at {self.company}. {skills_part}"
