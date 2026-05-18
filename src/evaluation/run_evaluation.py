"""
run_evaluation.py — The Scientific Evaluation Harness.

Executes the full experimental matrix:
  Personas (10) x Retrieval Methods (5+1 baseline) x Reasoning Models (3)

Experiment A: Compare retrieval strategies (BM25 vs FAISS, raw vs parsed, MiniLM vs MPNet).
Experiment B: Compare reasoning LLMs (Gemini 2.5 Pro, DeepSeek V4 Flash, Claude Sonnet 4.6)
              on the same reranked top-10, isolating the effect of the reasoning model.

Reranking is always Gemma (fixed across all runs), so Experiment B measures
reasoning quality independently of retrieval and reranking.

Results are saved as JSON + Markdown files for human Precision@10 labeling.
"""

import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Prevent tokenizer parallelism from conflicting with FAISS threads (Mac segfault fix)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# sentence_transformers must be imported before faiss to avoid Mac segfault
import faiss as faiss_lib

from src.evaluation.baseline_bm25 import BM25Retriever
from src.workflow.cv_profiler import profile_cv
from src.workflow.cv_reader import extract_text_from_pdf
from src.workflow.job_search import (
    index,
    job_descriptions,
    job_metadata,
    job_texts,
    retrieve_jobs,
    search_jobs,
)
from src.workflow.models import CVProfile, JobRecord, JobSearchPreferences
from src.workflow.reasoning import analyze_job_matches
from src.workflow.reranker import rerank_jobs
from src.workflow.retrieval_filters import passes_seniority_filter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

Provider = Literal["gemini", "deepseek", "claude"]

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalConfig:
    """All tunable constants for the evaluation run, in one place."""

    resumes_dir: Path = Path(__file__).parent.parent.parent / "data" / "resumes"
    output_dir: Path = Path(__file__).parent.parent.parent / "evaluation" / "results"
    vector_store_dir: Path = (
        Path(__file__).parent.parent.parent / "data" / "vector_store"
    )

    retrieval_k: int = 20  # jobs fetched per retrieval method
    rerank_k: int = 10  # jobs kept after reranking (passed to reasoning)
    persona_workers: int = 4  # parallel personas — bounded to avoid overwhelming APIs

    # Experiment B: which LLM reasons about job fit?
    models: tuple[Provider, ...] = ("gemini", "deepseek", "claude")

    # Experiment A: which retrieval strategy finds the best candidates?
    # Keys are human-readable labels; values are internal method IDs.
    methods: dict[str, str] = field(
        default_factory=lambda: {
            "BM25_RAW": "bm25_raw",
            "BM25_PARSED": "bm25_parsed",
            "FAISS_RAW": "faiss_raw",
            "FAISS_PARSED": "faiss_parsed",
            "FAISS_PARSED_MPNET": "faiss_parsed_mpnet",
        }
    )

    # Fixed for all personas — isolates the retrieval/model effect, not preferences
    default_preferences: JobSearchPreferences = field(
        default_factory=lambda: JobSearchPreferences(
            target_location="United States",
            work_type="full-time",
            employment_type="full-time",
            willing_to_relocate=True,
            remote_preference="flexible",
            target_roles=[],
            industry_preference=[],
        )
    )

    @property
    def all_method_names(self) -> list[str]:
        """All methods including the pre-rerank baseline derived from FAISS_PARSED."""
        return list(self.methods.keys()) + ["FAISS_PARSED_NORERANK"]


CONFIG = EvalConfig()
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)


# ── Lazy Model Cache ──────────────────────────────────────────────────────────


class ModelCache:
    """
    Loads heavy ML models once and holds them for the process lifetime.
    Thread-safe for read access after initial warm-up in the main thread.
    """

    def __init__(self):
        self._minilm = None
        self._mpnet = None
        self._mpnet_assets = None  # (index, texts, metadata, descriptions)

    def minilm(self):
        """Small, fast embedding model used for FAISS_RAW and FAISS_PARSED."""
        if self._minilm is None:
            from sentence_transformers import SentenceTransformer

            self._minilm = SentenceTransformer("all-MiniLM-L6-v2")
        return self._minilm

    def mpnet(self):
        """Larger, more accurate embedding model used for FAISS_PARSED_MPNET."""
        if self._mpnet is None:
            from sentence_transformers import SentenceTransformer

            self._mpnet = SentenceTransformer("all-mpnet-base-v2")
        return self._mpnet

    def mpnet_assets(self):
        """
        Loads the pre-built MPNet FAISS index + docstore from disk.
        Raises FileNotFoundError if the index hasn't been built yet.
        """
        if self._mpnet_assets is not None:
            return self._mpnet_assets

        idx_path = CONFIG.vector_store_dir / "faiss_mpnet.index"
        doc_path = CONFIG.vector_store_dir / "docstore_mpnet.json"
        desc_path = CONFIG.vector_store_dir / "job_descriptions_mpnet.json"

        if not idx_path.exists():
            raise FileNotFoundError(
                f"MPNet index not found at {idx_path}. "
                "Run: python -m src.data_pipeline.build_vector_store_mpnet"
            )

        mpnet_index = faiss_lib.read_index(str(idx_path), faiss_lib.IO_FLAG_MMAP)
        with open(doc_path, encoding="utf-8") as f:
            docstore = json.load(f)
        with open(desc_path, encoding="utf-8") as f:
            descriptions = json.load(f)

        self._mpnet_assets = (
            mpnet_index,
            [d["page_content"] for d in docstore],
            [d["metadata"] for d in docstore],
            descriptions,
        )
        logger.info(f"Loaded MPNet index: {mpnet_index.ntotal:,} vectors")
        return self._mpnet_assets


_models = ModelCache()


# ── Retrieval Strategies ──────────────────────────────────────────────────────


def _raw_profile_from_text(raw_text: str) -> CVProfile:
    """
    Wraps raw CV text into a minimal CVProfile.
    Used for 'raw' retrieval variants that skip structured parsing.
    """
    return CVProfile(
        skills=[raw_text],
        experience_level="mid",
        current_location=None,
        industries=[],
        domain_keywords=[],
        tools=[],
        languages=[],
        certifications=[],
        job_titles_held=[],
    )


def _build_job_records(
    raw_results: list[dict],
    descriptions: dict,
    cv_level: str,
    limit: int,
) -> list[JobRecord]:
    """
    Converts raw FAISS search results into typed JobRecord objects.
    Applies deduplication and seniority filtering along the way.
    """
    records: list[JobRecord] = []
    seen: set[str] = set()

    for r in raw_results:
        if r.get("source") != "kaggle":
            continue

        job_id = str(r.get("job_id", ""))
        url = r.get("url") or r.get("application_url") or ""

        # Skip if we've already added this job (by ID or URL)
        if job_id in seen or (url and url in seen):
            continue

        exp_level = (r.get("experience_level") or "").lower()
        title = (r.get("title") or "").lower()
        if not passes_seniority_filter(exp_level, title, cv_level):
            continue

        seen.add(job_id)
        if url:
            seen.add(url)

        records.append(
            JobRecord(
                job_id=job_id,
                title=str(r.get("title", "")),
                company=str(r.get("company", "")),
                description=descriptions.get(job_id, ""),
                location=r.get("location"),
                experience_level=r.get("experience_level"),
                work_type=r.get("work_type"),
                min_salary=float(r["min_salary"])
                if r.get("min_salary") is not None
                else None,
                max_salary=float(r["max_salary"])
                if r.get("max_salary") is not None
                else None,
                url=r.get("url"),
                skill_labels=r.get("skill_labels"),
                source=str(r.get("source", "kaggle")),
                score=float(r["score"]),
            )
        )

        if len(records) >= limit:
            break

    return records


def retrieve_bm25_raw(
    raw_text: str, prefs: JobSearchPreferences, bm25: BM25Retriever
) -> list[JobRecord]:
    return bm25.search(_raw_profile_from_text(raw_text), prefs, k=CONFIG.retrieval_k)


def retrieve_bm25_parsed(
    profile: CVProfile, prefs: JobSearchPreferences, bm25: BM25Retriever
) -> list[JobRecord]:
    return bm25.search(profile, prefs, k=CONFIG.retrieval_k)


def retrieve_faiss_parsed(
    profile: CVProfile, prefs: JobSearchPreferences
) -> list[JobRecord]:
    return retrieve_jobs(profile, prefs, top_k=CONFIG.retrieval_k)


def retrieve_faiss_raw(raw_text: str, profile: CVProfile) -> list[JobRecord]:
    vec = _models.minilm().encode([raw_text], convert_to_numpy=True).astype("float32")
    faiss_lib.normalize_L2(vec)
    raw_results = search_jobs(
        vec, index, job_texts, job_metadata, top_k=CONFIG.retrieval_k + 40
    )
    return _build_job_records(
        raw_results,
        job_descriptions,
        (profile.experience_level or "").lower(),
        CONFIG.retrieval_k,
    )


def retrieve_faiss_mpnet(
    profile: CVProfile, prefs: JobSearchPreferences
) -> list[JobRecord]:
    from src.workflow.job_search import serialize_cv_profile, serialize_preferences

    mpnet_index, mpnet_texts, mpnet_metadata, mpnet_descriptions = (
        _models.mpnet_assets()
    )
    query = f"Candidate profile: {serialize_cv_profile(profile)}. Job preferences: {serialize_preferences(prefs)}"
    vec = _models.mpnet().encode([query], convert_to_numpy=True).astype("float32")
    faiss_lib.normalize_L2(vec)
    raw_results = search_jobs(
        vec, mpnet_index, mpnet_texts, mpnet_metadata, top_k=CONFIG.retrieval_k + 40
    )
    return _build_job_records(
        raw_results,
        mpnet_descriptions,
        (profile.experience_level or "").lower(),
        CONFIG.retrieval_k,
    )


# Dispatch table: maps method ID → retrieval function call
# Adding a new retrieval strategy means adding one entry here.
def perform_retrieval(
    method_id: str,
    raw_text: str,
    profile: CVProfile,
    prefs: JobSearchPreferences,
    bm25: BM25Retriever,
) -> list[JobRecord]:
    dispatch = {
        "bm25_raw": lambda: retrieve_bm25_raw(raw_text, prefs, bm25),
        "bm25_parsed": lambda: retrieve_bm25_parsed(profile, prefs, bm25),
        "faiss_parsed": lambda: retrieve_faiss_parsed(profile, prefs),
        "faiss_raw": lambda: retrieve_faiss_raw(raw_text, profile),
        "faiss_parsed_mpnet": lambda: retrieve_faiss_mpnet(profile, prefs),
    }
    if method_id not in dispatch:
        raise ValueError(
            f"Unknown retrieval method: '{method_id}'. Valid: {list(dispatch)}"
        )
    return dispatch[method_id]()


# ── Result Persistence ────────────────────────────────────────────────────────


def save_json_result(
    path: Path,
    persona_id: str,
    method_name: str,
    model: Provider,
    jobs: list[JobRecord],
    report: Any,
) -> None:
    result = {
        "persona": persona_id,
        "method": method_name,
        "model": model,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [j.model_dump() for j in jobs],
        "reasoning": report.model_dump(),
    }
    path.write_text(json.dumps(result, indent=2))


def save_md_labeling_sheet(
    path: Path,
    persona_id: str,
    method_name: str,
    model: str,
    jobs: list[JobRecord],
    report: Any,
    profile: CVProfile | None = None,
) -> None:
    """Writes a Markdown file for a human evaluator to score each result."""

    def _escape(text: str) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", "")

    def _job_url(job: JobRecord) -> str | None:
        if job.url:
            return job.url
        if job.source == "kaggle" and job.job_id:
            return f"https://www.linkedin.com/jobs/view/{job.job_id}"
        return None

    job_explanations = {e.job_id: e for e in report.job_explanations}
    lines: list[str] = []

    # ── Header ──
    lines += [f"# Labeling Sheet: {persona_id} | {method_name} | {model}", ""]

    # ── Candidate summary ──
    if profile:
        skills_preview = ", ".join(profile.skills[:12]) + (
            "…" if len(profile.skills) > 12 else ""
        )
        edu = (
            f"{profile.education_level} in {profile.field_of_study}"
            if profile.education_level
            else "not specified"
        )
        lines += [
            "## Candidate Profile",
            "",
            f"**Level:** {profile.experience_level} ({profile.years_experience} yrs)  |  **Education:** {edu}",
            f"**Industries:** {', '.join(profile.industries) or 'not specified'}",
            f"**Key Skills:** {skills_preview}",
            "",
            "---",
            "",
        ]

    # ── Scoring rubric ──
    lines += [
        "**Relevance rubric:** correct domain + within ±1 seniority level = 1, else 0",
        "**Quality rubric:** 1=wrong/hallucinated  3=generic but correct  5=specific with evidence",
        "",
    ]

    # ── Label table (evaluator fills this in) ──
    lines += [
        "## Label Table",
        "",
        "Fill this in after reading the job details below.",
        "",
        "| # | Relevant (0/1) | Quality (1-5) | Title | Company |",
        "|---|----------------|---------------|-------|---------|",
    ]
    for i, job in enumerate(jobs, 1):
        lines.append(f"| {i} | | | {_escape(job.title)} | {_escape(job.company)} |")

    lines += ["", "---", "", "## Job Details", ""]

    # ── Per-job detail blocks ──
    for i, job in enumerate(jobs, 1):
        exp = job_explanations.get(job.job_id)
        url = _job_url(job)
        desc = (
            (job.description[:800] + "…")
            if len(job.description) > 800
            else job.description
        )

        salary = ""
        if job.min_salary or job.max_salary:
            lo = f"${job.min_salary:,.0f}" if job.min_salary else "?"
            hi = f"${job.max_salary:,.0f}" if job.max_salary else "?"
            salary = f" | Salary: {lo}–{hi}"

        lines += [
            f"### {i}. {job.title} @ {job.company}",
            f"**Level:** {job.experience_level or 'not specified'} | **Location:** {job.location or 'not specified'}{salary}"
            f" | {f'[View posting]({url})' if url else 'N/A'}",
            "",
            "**Description:**",
            desc,
            "",
            f"**Match reason:** {exp.match_reason if exp else '—'}",
            f"**Missing skills:** {', '.join(exp.missing_skills) if exp and exp.missing_skills else 'None'}",
            "",
            "---",
            "",
        ]

    # ── Career roadmap ──
    lines += [
        "## Career Roadmap (from reasoning)",
        "",
        f"**Top missing skills:** {', '.join(report.overall_missing_skills) or 'None'}",
        "",
        f"**Recommendation:** {report.recommendation}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Reasoning & Saving One Combination ───────────────────────────────────────


def run_and_save_one_combo(
    persona_dir: Path,
    persona_id: str,
    method_name: str,
    model: Provider,
    profile: CVProfile,
    top_10: list[JobRecord],
) -> str:
    """
    Runs one reasoning model on the top-10 jobs and writes JSON + MD.
    Returns a short status string for logging.
    """
    json_path = persona_dir / f"{method_name}_{model}.json"
    md_path = persona_dir / f"{method_name}_{model}.md"

    if json_path.exists() and md_path.exists():
        return f"skipped {model} (already exists)"

    report = analyze_job_matches(
        cv=profile, jobs=top_10, provider=model, use_cache=True
    )

    save_json_result(json_path, persona_id, method_name, model, top_10, report)
    save_md_labeling_sheet(
        md_path, persona_id, method_name, model, top_10, report, profile=profile
    )

    return f"done {model}"


def run_all_models_for_method(
    method_name: str,
    top_10: list[JobRecord],
    persona_dir: Path,
    persona_id: str,
    profile: CVProfile,
    counts: dict,
) -> None:
    """Fires all 3 reasoning models in parallel for one (method, top-10) pair."""
    with ThreadPoolExecutor(max_workers=len(CONFIG.models)) as executor:
        futures = {
            executor.submit(
                run_and_save_one_combo,
                persona_dir,
                persona_id,
                method_name,
                m,
                profile,
                top_10,
            ): m
            for m in CONFIG.models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                status = future.result()
                logger.info(f"    {status}")
                counts["skipped" if "skipped" in status else "done"] += 1
            except Exception as e:
                logger.exception(f"    Reasoning failed for {model}: {e}")
                counts["failed"] += 1


# ── Per-Persona Orchestration ─────────────────────────────────────────────────


def run_persona(pdf_path: Path, bm25: BM25Retriever) -> dict:
    """
    Runs the full method × model matrix for one resume.
    Returns a counts dict: {done, skipped, failed}.
    """
    persona_id = pdf_path.stem
    persona_dir = CONFIG.output_dir / persona_id
    persona_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}

    logger.info(f"\nProcessing Persona: {persona_id}")

    raw_text = extract_text_from_pdf(pdf_path, use_cache=True)
    profile = profile_cv(raw_text, use_cache=True)

    for method_name, method_id in CONFIG.methods.items():
        logger.info(f"  [{persona_id}] Method: {method_name}")

        # ── Retrieval ──
        try:
            top_20 = perform_retrieval(
                method_id, raw_text, profile, CONFIG.default_preferences, bm25
            )
        except Exception as e:
            logger.exception(
                f"  [{persona_id}] Retrieval failed for {method_name}: {e}"
            )
            counts["failed"] += len(CONFIG.models)
            continue

        # ── H3 baseline: save top-20 slice before reranking (FAISS_PARSED only) ──
        if method_name == "FAISS_PARSED":
            logger.info(f"  [{persona_id}] Saving H3 baseline: FAISS_PARSED_NORERANK")
            run_all_models_for_method(
                "FAISS_PARSED_NORERANK",
                top_20[: CONFIG.rerank_k],
                persona_dir,
                persona_id,
                profile,
                counts,
            )

        # ── Reranking ──
        try:
            top_10 = rerank_jobs(
                cv=profile,
                preferences=CONFIG.default_preferences,
                jobs=top_20,
                use_cache=True,
            )
        except Exception as e:
            logger.exception(
                f"  [{persona_id}] Reranking failed for {method_name}: {e}"
            )
            counts["failed"] += len(CONFIG.models)
            continue

        if len(top_10) < CONFIG.rerank_k:
            logger.warning(
                f"  [{persona_id}] Only {len(top_10)} jobs after reranking (expected {CONFIG.rerank_k})"
            )

        # ── Reasoning (all 3 models in parallel) ──
        run_all_models_for_method(
            method_name, top_10, persona_dir, persona_id, profile, counts
        )

    return counts


# ── Diagnostics ───────────────────────────────────────────────────────────────


def log_missing_combos(pdf_files: list[Path]) -> None:
    """Groups and logs any (persona, method, model) combinations missing from disk."""
    missing = [
        f"{pdf.stem}/{method}_{model}"
        for pdf in pdf_files
        for method in CONFIG.all_method_names
        for model in CONFIG.models
        if not (CONFIG.output_dir / pdf.stem / f"{method}_{model}.json").exists()
        or not (CONFIG.output_dir / pdf.stem / f"{method}_{model}.md").exists()
    ]

    if not missing:
        return

    logger.info(f"  Missing combos ({len(missing)}):")
    if len(missing) <= 5:
        for m in missing:
            logger.info(f"    - {m}")
        return

    by_method: Counter = Counter()
    by_model: Counter = Counter()
    for m in missing:
        _, combo = m.split("/", 1)
        method, model = combo.rsplit("_", 1)
        by_method[method] += 1
        by_model[model] += 1
    logger.info(
        "    By method: " + ", ".join(f"{k}×{v}" for k, v in by_method.most_common())
    )
    logger.info(
        "    By model:  " + ", ".join(f"{k}×{v}" for k, v in by_model.most_common())
    )


# ── Entry Point ───────────────────────────────────────────────────────────────


def run_evaluation() -> None:
    logger.info("=" * 60)
    logger.info("Starting Job Market Agent Experimental Suite")
    logger.info("=" * 60)

    # Warm up models in the main thread before spawning workers — avoids race conditions
    _models.minilm()
    try:
        _models.mpnet_assets()
        _models.mpnet()
    except FileNotFoundError as e:
        logger.warning(f"MPNet index unavailable — FAISS_PARSED_MPNET will fail: {e}")

    bm25 = BM25Retriever()

    pdf_files = sorted(CONFIG.resumes_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF resumes found in {CONFIG.resumes_dir}")
        return

    # +1 accounts for FAISS_PARSED_NORERANK (the H3 pre-rerank baseline)
    total = len(pdf_files) * (len(CONFIG.methods) + 1) * len(CONFIG.models)
    totals = {"done": 0, "skipped": 0, "failed": 0}

    logger.info(
        f"Found {len(pdf_files)} personas — running {CONFIG.persona_workers} in parallel."
    )

    with ThreadPoolExecutor(max_workers=CONFIG.persona_workers) as executor:
        futures = {
            executor.submit(run_persona, pdf, bm25): pdf.stem for pdf in pdf_files
        }
        for future in as_completed(futures):
            persona_id = futures[future]
            try:
                for k, v in future.result().items():
                    totals[k] += v
            except Exception as e:
                logger.exception(f"Persona {persona_id} failed unexpectedly: {e}")
                totals["failed"] += (len(CONFIG.methods) + 1) * len(CONFIG.models)

    # Count unresolved from filesystem — authoritative, not based on in-memory arithmetic
    unresolved = sum(
        1
        for pdf in pdf_files
        for method in CONFIG.all_method_names
        for model in CONFIG.models
        if not (CONFIG.output_dir / pdf.stem / f"{method}_{model}.json").exists()
        or not (CONFIG.output_dir / pdf.stem / f"{method}_{model}.md").exists()
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Experiments complete. Results saved to: {CONFIG.output_dir}")
    logger.info(f"  ✓ Done:    {totals['done']}/{total}")
    logger.info(f"  ↩ Skipped: {totals['skipped']}/{total}  (already existed)")
    logger.info(f"  ✗ Failed:  {totals['failed']}/{total}  (this run)")
    logger.info(f"  ◉ Unresolved on disk: {unresolved}/{total}")
    if unresolved > 0:
        logger.info("  ⚠ Re-run this script to retry failed combinations.")
        log_missing_combos(pdf_files)
    else:
        logger.info("  All combinations complete — ready for labeling.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_evaluation()
