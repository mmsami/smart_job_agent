"""
Workflow output schemas shared across retrieval methods (FAISS, BM25).

Two separate inputs feed into job search:
  - CVProfile: factual data extracted from the CV (who the person is)
  - JobSearchPreferences: what the user tells us they want (user input, not from CV)

JobRecord represents a single job from the search results,
with metadata for evaluation and display.
"""

from typing import Optional

from pydantic import BaseModel


class JobRecord(BaseModel):
    """Single job from search results (FAISS or BM25)."""

    job_id: str
    title: str
    company: str
    description: str
    location: Optional[str] = None
    experience_level: Optional[str] = None
    work_type: Optional[str] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    url: Optional[str] = None
    skill_labels: Optional[str] = None
    source: str  # "kaggle" or "arbeitnow"
    score: float  # relevance score (BM25 or cosine similarity)


class CVProfile(BaseModel):
    """
    Factual data extracted from the CV. No preferences — just who the person is.
    Populated by cv_parser.py (LLM step).
    """

    skills: list[str]                        # technical + domain skills
    experience_level: str                    # "entry" | "mid" | "senior" (derived from years)
    years_experience: Optional[int] = None   # numeric, calculated from career dates
    current_location: str                    # where CV says person is based
    education_level: Optional[str] = None   # "bachelor" | "master" | "phd"
    field_of_study: Optional[str] = None    # "Accounting", "Finance", "Computer Science"
    certifications: list[str] = []           # CPA, PMP, MBA, etc.
    languages: list[str] = []               # Korean, English, Chinese
    job_titles_held: list[str] = []          # past titles (Accounting Manager, etc.)
    industries: list[str] = []              # domains worked in (Renewable Energy, Finance)
    domain_keywords: list[str] = []         # domain-specific terms: GAAP, IFRS, SOX, reconciliation, audit
    tools: list[str] = []                   # specific software tools: NetSuite, Oracle, QuickBooks, Bloomberg


class JobSearchPreferences(BaseModel):
    """
    What the user wants — provided by user input, NOT extracted from CV.
    A CV says nothing about where someone wants to work or what type of job they want.
    Populated via CLI prompt / form in main.py.
    """

    target_location: str                     # where they want to work
    work_type: str                           # "full-time" | "part-time" | "remote" | "hybrid"
    employment_type: str = "full-time"       # "full-time" | "part-time" | "contract" | "any"
    willing_to_relocate: bool = False        # open to moving?
    target_roles: list[str] = []            # specific roles they're targeting (optional)
    industry_preference: list[str] = []     # industries user wants to work in
    remote_preference: str = "flexible"     # "remote" | "hybrid" | "onsite" | "flexible"
