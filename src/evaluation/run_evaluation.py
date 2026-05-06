"""
run_evaluation.py — The Scientific Evaluation Harness.

This script executes the full experimental matrix:
  Personas (10) x Retrieval Methods (4) x Reasoning Models (3)

Experiment A: Compare 4 retrieval strategies (BM25 vs FAISS, raw vs parsed).
Experiment B: Compare 3 reasoning LLMs (Gemma, DeepSeek, Claude) on the same
              reranked top-10, isolating the effect of the reasoning model.

Reranking is always Gemma (fixed across all runs). This allows Experiment B to
measure reasoning quality independently of retrieval and reranking.

Results are saved as JSON files for human Precision@10 labeling.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import faiss as faiss_lib
from dotenv import load_dotenv

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

load_dotenv()

# Loaded once before the parallel loop — avoids race condition across persona threads
_EMBED_MODEL = None
_MPNET_MODEL = None
_MPNET_ASSETS = None  # (index, job_texts, job_metadata, job_descriptions)


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _get_mpnet_model():
    global _MPNET_MODEL
    if _MPNET_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MPNET_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _MPNET_MODEL


def _get_mpnet_assets():
    global _MPNET_ASSETS
    if _MPNET_ASSETS is not None:
        return _MPNET_ASSETS

    data_dir = Path(__file__).parent.parent.parent / "data" / "vector_store"
    index_path = data_dir / "faiss_mpnet.index"
    docstore_path = data_dir / "docstore_mpnet.json"
    descriptions_path = data_dir / "job_descriptions_mpnet.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"MPNet index not found at {index_path}. "
            "Run: python -m src.data_pipeline.build_vector_store_mpnet"
        )

    mpnet_index = faiss_lib.read_index(str(index_path), faiss_lib.IO_FLAG_MMAP)
    with open(docstore_path, encoding="utf-8") as f:
        docstore = json.load(f)
    with open(descriptions_path, encoding="utf-8") as f:
        mpnet_descriptions = json.load(f)

    mpnet_texts = [d["page_content"] for d in docstore]
    mpnet_metadata = [d["metadata"] for d in docstore]

    _MPNET_ASSETS = (mpnet_index, mpnet_texts, mpnet_metadata, mpnet_descriptions)
    logger.info(f"Loaded MPNet index: {mpnet_index.ntotal:,} vectors")
    return _MPNET_ASSETS


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

RESUMES_DIR = Path(__file__).parent.parent.parent / "data" / "resumes"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_K = 20  # jobs fetched per retrieval method
RERANK_K = 10  # jobs kept after reranking (passed to reasoning)

Provider = Literal["gemma", "deepseek", "claude"]

# Experiment B: 3 reasoning LLMs (reranker is always Gemma)
MODELS: list[Provider] = ["gemma", "deepseek", "claude"]

# Experiment A: retrieval/query combinations
# FAISS_PARSED_MPNET added per professor feedback: test stronger embedding model
METHODS = {
    "BM25_RAW": "bm25_raw",
    "BM25_PARSED": "bm25_parsed",
    "FAISS_RAW": "faiss_raw",
    "FAISS_PARSED": "faiss_parsed",
    "FAISS_PARSED_MPNET": "faiss_parsed_mpnet",
}

# Fixed preferences for all personas — isolates retrieval/model effect
DEFAULT_PREFS = JobSearchPreferences(
    target_location="United States",
    work_type="full-time",
    employment_type="full-time",
    willing_to_relocate=True,
    remote_preference="flexible",
    target_roles=[],
    industry_preference=[],
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_raw_profile(raw_text: str) -> CVProfile:
    """
    Creates a CVProfile with all raw text in skills field.
    Used for 'Raw' retrieval tests — bypasses cv_profiler parsing.
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


def perform_retrieval(
    method: str,
    raw_text: str,
    profile: CVProfile,
    prefs: JobSearchPreferences,
    bm25: BM25Retriever,
) -> list[JobRecord]:
    """Handles the 4 different retrieval combinations."""

    if method == "bm25_raw":
        raw_prof = get_raw_profile(raw_text)
        return bm25.search(raw_prof, prefs, k=RETRIEVAL_K)

    elif method == "bm25_parsed":
        return bm25.search(profile, prefs, k=RETRIEVAL_K)

    elif method == "faiss_raw":
        raw_vec = _get_embed_model().encode([raw_text], convert_to_numpy=True).astype("float32")
        faiss_lib.normalize_L2(raw_vec)
        raw = search_jobs(raw_vec, index, job_texts, job_metadata, top_k=RETRIEVAL_K + 40)
        records: list[JobRecord] = []
        seen: set[str] = set()
        for r in raw:
            if r.get("source") != "kaggle":
                continue
            job_id = str(r.get("job_id", ""))
            if job_id in seen:
                continue
            seen.add(job_id)
            records.append(JobRecord(
                job_id=job_id,
                title=str(r.get("title", "")),
                company=str(r.get("company", "")),
                description=job_descriptions.get(job_id, ""),
                location=r.get("location"),
                experience_level=r.get("experience_level"),
                work_type=r.get("work_type"),
                min_salary=float(r["min_salary"]) if r.get("min_salary") is not None else None,
                max_salary=float(r["max_salary"]) if r.get("max_salary") is not None else None,
                url=r.get("url"),
                skill_labels=r.get("skill_labels"),
                source=str(r.get("source", "kaggle")),
                score=float(r["score"]),
            ))
            if len(records) >= RETRIEVAL_K:
                break
        return records

    elif method == "faiss_parsed":
        return retrieve_jobs(profile, prefs, top_k=RETRIEVAL_K)

    elif method == "faiss_parsed_mpnet":
        mpnet_index, mpnet_texts, mpnet_metadata, mpnet_descriptions = _get_mpnet_assets()
        mpnet_model = _get_mpnet_model()
        from src.workflow.job_search import serialize_cv_profile, serialize_preferences
        combined = f"Candidate profile: {serialize_cv_profile(profile)}. Job preferences: {serialize_preferences(prefs)}"
        vec = mpnet_model.encode([combined], convert_to_numpy=True).astype("float32")
        faiss_lib.normalize_L2(vec)
        raw = search_jobs(vec, mpnet_index, mpnet_texts, mpnet_metadata, top_k=RETRIEVAL_K + 40)
        records: list[JobRecord] = []
        seen: set[str] = set()
        for r in raw:
            if r.get("source") != "kaggle":
                continue
            job_id = str(r.get("job_id", ""))
            if job_id in seen:
                continue
            seen.add(job_id)
            records.append(JobRecord(
                job_id=job_id,
                title=str(r.get("title", "")),
                company=str(r.get("company", "")),
                description=mpnet_descriptions.get(job_id, ""),
                location=r.get("location"),
                experience_level=r.get("experience_level"),
                work_type=r.get("work_type"),
                min_salary=float(r["min_salary"]) if r.get("min_salary") is not None else None,
                max_salary=float(r["max_salary"]) if r.get("max_salary") is not None else None,
                url=r.get("url"),
                skill_labels=r.get("skill_labels"),
                source=str(r.get("source", "kaggle")),
                score=float(r["score"]),
            ))
            if len(records) >= RETRIEVAL_K:
                break
        return records

    else:
        raise ValueError(f"Unknown method: {method}")


# ── MD Preview ───────────────────────────────────────────────────────────────


def save_md_preview(
    path: Path,
    persona_id: str,
    method_name: str,
    model: str,
    jobs: list[JobRecord],
    report,
    profile: CVProfile | None = None,
) -> None:
    """Write a human-readable labeling sheet alongside the JSON."""

    def _md(text: str) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", "")

    def _job_url(job: JobRecord) -> str | None:
        if job.url:
            return job.url
        if job.source == "kaggle" and job.job_id:
            return f"https://www.linkedin.com/jobs/view/{job.job_id}"
        return None

    job_map = {e.job_id: e for e in report.job_explanations}

    lines = [
        f"# Labeling Sheet: {persona_id} | {method_name} | {model}",
        "",
    ]

    if profile:
        skills_str = ", ".join(profile.skills[:12]) + ("…" if len(profile.skills) > 12 else "")
        industries_str = ", ".join(profile.industries) or "not specified"
        edu = f"{profile.education_level} in {profile.field_of_study}" if profile.education_level else "not specified"
        lines += [
            "## Candidate Profile",
            "",
            f"**Level:** {profile.experience_level} ({profile.years_experience} yrs)  |  **Education:** {edu}",
            f"**Industries:** {industries_str}",
            f"**Key Skills:** {skills_str}",
            "",
            "---",
            "",
        ]

    lines += [
        "**Relevance rubric:** correct domain + within ±1 seniority level = 1, else 0",
        "**Quality rubric:** 1=wrong/hallucinated  3=generic but correct  5=specific with evidence",
        "",
        "## Label Table",
        "",
        "Fill this in after reading the job details below.",
        "",
        "| # | Relevant (0/1) | Quality (1-5) | Title | Company |",
        "|---|----------------|---------------|-------|---------|",
    ]

    for i, job in enumerate(jobs, 1):
        lines.append(f"| {i} | | | {_md(job.title)} | {_md(job.company)} |")

    lines += [
        "",
        "---",
        "",
        "## Job Details",
        "",
    ]

    for i, job in enumerate(jobs, 1):
        exp = job_map.get(job.job_id)
        url = _job_url(job)
        url_str = f"[View posting]({url})" if url else "N/A"
        level = job.experience_level or "not specified"
        location = job.location or "not specified"
        salary = ""
        if job.min_salary or job.max_salary:
            lo = f"${job.min_salary:,.0f}" if job.min_salary else "?"
            hi = f"${job.max_salary:,.0f}" if job.max_salary else "?"
            salary = f" | Salary: {lo}–{hi}"

        desc = (job.description[:800] + "…") if len(job.description) > 800 else job.description

        match_reason = exp.match_reason if exp else "—"
        missing = ", ".join(exp.missing_skills) if exp and exp.missing_skills else "None"

        lines += [
            f"### {i}. {job.title} @ {job.company}",
            f"**Level:** {level} | **Location:** {location}{salary} | {url_str}",
            "",
            "**Description:**",
            desc,
            "",
            f"**Match reason:** {match_reason}",
            f"**Missing skills:** {missing}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Career Roadmap (from reasoning)",
        "",
        f"**Top missing skills:** {', '.join(report.overall_missing_skills) or 'None'}",
        "",
        f"**Recommendation:** {report.recommendation}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Combo Saver ──────────────────────────────────────────────────────────────


def _save_combo(
    persona_dir: Path,
    persona_id: str,
    method_name: str,
    model: Provider,  # 3. Update hint from str to Provider
    parsed_profile: Any,
    top_10: list,
) -> str:
    """Run one reasoning model on top_10 and write JSON + MD. Returns status string."""
    json_path = persona_dir / f"{method_name}_{model}.json"
    md_path = persona_dir / f"{method_name}_{model}.md"

    if json_path.exists() and md_path.exists():
        return f"skipped {model} (already exists)"

    report = analyze_job_matches(
        cv=parsed_profile,
        jobs=top_10,
        provider=model,  # The linter now knows 'model' is a valid Provider
        use_cache=True,
    )

    result_data = {
        "persona": persona_id,
        "method": method_name,
        "model": model,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [j.model_dump() for j in top_10],
        "reasoning": report.model_dump(),
    }

    json_path.write_text(json.dumps(result_data, indent=2))
    save_md_preview(md_path, persona_id, method_name, model, top_10, report, profile=parsed_profile)
    return f"done {model}"


def _run_models_parallel(
    method_name: str,
    top_10: list,
    persona_dir: Path,
    persona_id: str,
    parsed_profile,
    counts: dict,
) -> None:
    """Fire all 3 reasoning models in parallel for one method × top_10."""
    with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        futures = {
            executor.submit(
                _save_combo,
                persona_dir,
                persona_id,
                method_name,
                m,
                parsed_profile,
                top_10,
            ): m
            for m in MODELS
        }
        for future in as_completed(futures):
            m = futures[future]
            try:
                result = future.result()
                logger.info(f"    {result}")
                if "skipped" in result:
                    counts["skipped"] += 1
                else:
                    counts["done"] += 1
            except Exception as e:
                logger.exception(f"    Reasoning failed for {m}: {e}")
                counts["failed"] += 1


# ── Per-Persona Worker ────────────────────────────────────────────────────────

PERSONA_WORKERS = 4  # parallel personas — bounded to avoid overwhelming APIs


def _run_persona(pdf_path: Path, bm25: BM25Retriever) -> dict:
    """Run the full method × model matrix for one persona. Returns counts dict."""
    persona_id = pdf_path.stem
    persona_dir = OUTPUT_DIR / persona_id
    persona_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}

    logger.info(f"\nProcessing Persona: {persona_id}")

    raw_text = extract_text_from_pdf(pdf_path, use_cache=True)
    parsed_profile = profile_cv(raw_text, use_cache=True)

    for method_name, method_id in METHODS.items():
        logger.info(f"  [{persona_id}] Method: {method_name}")

        try:
            top_20 = perform_retrieval(
                method_id, raw_text, parsed_profile, DEFAULT_PREFS, bm25
            )
        except Exception as e:
            logger.exception(f"  [{persona_id}] Retrieval failed for {method_name}: {e}")
            counts["failed"] += len(MODELS)
            continue

        # H3: pre-rerank baseline for FAISS_PARSED only
        if method_name == "FAISS_PARSED":
            logger.info(f"  [{persona_id}] Saving H3 baseline: FAISS_PARSED_NORERANK")
            _run_models_parallel(
                "FAISS_PARSED_NORERANK", top_20[:RERANK_K],
                persona_dir, persona_id, parsed_profile, counts,
            )

        try:
            top_10 = rerank_jobs(
                cv=parsed_profile,
                preferences=DEFAULT_PREFS,
                jobs=top_20,
                use_cache=True,
            )
        except Exception as e:
            logger.exception(f"  [{persona_id}] Reranking failed for {method_name}: {e}")
            counts["failed"] += len(MODELS)
            continue

        if len(top_10) < RERANK_K:
            logger.warning(
                f"  [{persona_id}] Only {len(top_10)} jobs after reranking (expected {RERANK_K})"
            )

        _run_models_parallel(
            method_name, top_10, persona_dir, persona_id, parsed_profile, counts
        )

    return counts


# ── Main Experiment Loop ──────────────────────────────────────────────────────


def run_evaluation():
    logger.info("=" * 60)
    logger.info("Starting Job Market Agent Experimental Suite")
    logger.info("=" * 60)

    # Pre-load models before forking threads — avoids race on lazy init
    _get_embed_model()
    try:
        _get_mpnet_assets()
        _get_mpnet_model()
    except FileNotFoundError as e:
        logger.warning(f"MPNet index unavailable — FAISS_PARSED_MPNET will fail: {e}")

    bm25_retriever = BM25Retriever()

    pdf_files = sorted(RESUMES_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF resumes found in data/resumes/")
        return

    # +1 for FAISS_PARSED_NORERANK (H3 pre-rerank baseline, derived from FAISS_PARSED)
    # METHODS already includes FAISS_PARSED_MPNET (6 methods total)
    total = len(pdf_files) * (len(METHODS) + 1) * len(MODELS)
    logger.info(f"Found {len(pdf_files)} personas — running {PERSONA_WORKERS} in parallel.")

    totals = {"done": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=PERSONA_WORKERS) as executor:
        futures = {
            executor.submit(_run_persona, pdf_path, bm25_retriever): pdf_path.stem
            for pdf_path in pdf_files
        }
        for future in as_completed(futures):
            persona_id = futures[future]
            try:
                counts = future.result()
                for k in totals:
                    totals[k] += counts[k]
            except Exception as e:
                logger.exception(f"Persona {persona_id} failed unexpectedly: {e}")
                totals["failed"] += (len(METHODS) + 1) * len(MODELS)

    missing = total - totals["done"] - totals["skipped"] - totals["failed"]
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Experiments complete. Results saved to: {OUTPUT_DIR}")
    logger.info(f"  ✓ Done:    {totals['done']}/{total}")
    logger.info(f"  ↩ Skipped: {totals['skipped']}/{total}  (already existed)")
    logger.info(f"  ✗ Failed:  {totals['failed']}/{total}  (re-run to retry)")
    if totals["failed"] > 0 or missing > 0:
        logger.info("  ⚠ Re-run this script to retry failed combinations.")
    else:
        logger.info("  All combinations complete — ready for labeling.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_evaluation()
