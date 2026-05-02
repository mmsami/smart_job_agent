# Dev Log: Pipeline Wiring (Step 2e)
**Date:** 2026-05-01
**Objective:** Wire all four pipeline components (cv_reader → cv_profiler → job_search → reranker → reasoning) into a single interactive CLI entry point.

---

## Scope

**File:** `src/main.py`
**Purpose:** End-to-end product demo. Takes a PDF CV from the user, collects preferences, runs the full pipeline, and displays tiered results in a Rich CLI.

---

## 1. Design Decisions

### Parallel Onboarding
The CV profiling (PDF → CVProfile) and preference collection both happen at startup. Rather than blocking the user while the vision LLM processes the CV, we run profiling in a `ThreadPoolExecutor` background thread while the user answers preference prompts in the foreground.

**Why:** cv_reader + cv_profiler together take ~15–30 seconds (vision LLM + Gemma). The user has ~5 preference questions to answer. Parallelising saves 15–20 seconds of perceived wait time.

### Tiered Output
Top 3 results get rich `Panel` displays with full match reason and missing skills. Ranks 4–10 get a compact `Table`. Career roadmap (missing skills) shown at the end.

**Why:** The top 3 are what the user cares most about. Showing 10 rich panels would be overwhelming and hide the signal.

### Verification Step
After profiling, the agent shows the extracted `CVProfile` and asks the user to confirm or edit key fields (experience_level, skills, industries, current_location).

**Why:** CV extraction isn't perfect. Giving the user a correction step prevents downstream errors from propagating through retrieval and reranking.

### Audit Trail
Session saved to `iterations/results_<cv_name>_<timestamp>.md` after each run. Contains CVProfile, preferences, all job matches with scores, match reasons, missing skills, and career roadmap.

**Why:** Needed for professor evaluation of the product. Also useful for debugging.

---

## 2. Linting Fixes Applied

During review, three linting issues were found and fixed:

| Issue | Fix |
|-------|-----|
| `report_to_pretty_json` imported from `reasoning` but never used | Removed from import |
| `return profile` at end of `verify_profile` was unreachable (after `while True` that only exits via `return`) | Removed dead code |
| `sys.path.insert(0, ...)` appeared after the local imports it was meant to support | Moved before local imports |

---

## 3. Architecture Notes

```
run_agent()
  ├── ThreadPoolExecutor
  │     ├── BG: extract_text_from_pdf() → profile_cv()  ← parallel
  │     └── FG: collect_preferences()                   ← parallel
  ├── verify_profile()           ← user correction step
  ├── retrieve_jobs()            ← FAISS top-20
  ├── rerank_jobs()              ← Gemma top-10
  ├── analyze_job_matches()      ← Gemma reasoning
  ├── Rich CLI output (tiered)
  └── save_full_audit_trail()    ← iterations/results_*.md
```

---

## Known Limitations

- Single file mode only (no batch). Batch processing is handled by `run_evaluation.py`.
- Preferences are collected interactively — not suitable for automated runs.
- If CV profiling fails (e.g. blank PDF, API timeout), the pipeline aborts early with an error message.

---

## 4. Intelligence Upgrade (2026-05-02)

The original CLI collected preferences via a rigid multi-question form. This section documents the rewrite to freeform natural language throughout.

### pref_parser.py — new module

`src/workflow/pref_parser.py` added with three public functions:

```python
parse_preferences(text: str) -> JobSearchPreferences
refine_preferences(existing: JobSearchPreferences, correction: str) -> JobSearchPreferences
apply_profile_edit(profile: CVProfile, instruction: str) -> CVProfile
```

Same Gemma + OpenRouter fallback pattern as `cv_profiler`. No caching (session-specific, cheap to re-parse).

`src/prompts/pref_parser.md` — prompt template with schema, rules, and 3 examples for Gemma.

Key design decision for `refine_preferences`: passes the full current prefs JSON + the user's delta to Gemma in one call, instructing it to preserve unchanged fields. Safer than merging two parsed objects because Gemma can interpret intent ("swap fintech for healthtech") rather than append blindly.

### collect_preferences() — freeform NL

User types a single natural language sentence describing their search. Parsed via `parse_preferences()`. Shows "Understood as" panel with the parsed result. On correction, calls `refine_preferences(prefs, text)` — not `parse_preferences()` again. Calling parse fresh on a correction wipes all previously stated fields.

### verify_profile() — NL profile editing

After CV profiling, user sees parsed `CVProfile` and can say "change experience to senior" or "add Kubernetes to skills". Routed to `apply_profile_edit()` which passes the full profile JSON + instruction to Gemma and returns the updated profile.

### Refinement loop

After results display, user can refine preferences and re-run the pipeline. Each iteration uses `refine_preferences(prefs, refine_text)` — not a fresh parse — so prior preferences are preserved unless the user explicitly changes them. `console.rule()` divides iterations visually.

### _pick_cv() — numbered CV list

Lists all PDFs in `data/resumes/` with numbers before entering the selection loop. List prints once — not on every invalid input (which was the previous behaviour). Falls back to manual path entry if user selects 0 or the directory is empty.

### Apply links for Kaggle jobs

The Kaggle LinkedIn dataset contains LinkedIn job IDs. LinkedIn posting URLs follow a fixed pattern:

```
https://www.linkedin.com/jobs/view/{job_id}
```

`_job_url(job)` constructs this when `job.source == "kaggle"` and `job.url` is not set. Used in both the top-3 panels and the compact table. Table cells show a shortened URL (truncated at 30 chars) to fit the column.

### Street address filter in cv_profiler

Some CVs list a street address in the location field. Added a regex filter in `_build_profile`:

```python
_street_pattern = re.compile(
    r"^\d+\s|\b(st\.|ave\.|blvd\.|dr\.|rd\.|lane|way|court|ct\.|place|pl\.|suite|#)\b",
    re.IGNORECASE,
)
current_location = None if _street_pattern.search(loc) else loc
```

Triggers on leading digits (e.g. "123 Main St") or common street suffix keywords. Returns `None` so FAISS doesn't embed a home address as a location signal.

### Log suppression

Noisy SDK loggers (httpx, httpcore, google.auth, google.genai, google.generativeai, src.workflow) suppressed to WARNING level in `main.py` only. Evaluation scripts keep full logging — suppression is demo-context-only.

### save_full_audit_trail — description + apply URL

Each job in the saved `iterations/results_*.md` now includes:
- Apply URL (LinkedIn link for Kaggle jobs)
- First 600 characters of the job description

Previously the audit trail only contained title, company, score, match reason, and missing skills — not enough for the user to act on the results after the session ends.

---

## 5. Audit Report Quality Fixes (2026-05-02)

Issues found by reviewing all `results_cv1_*.md` test runs:

### Score ordering bug

Jobs in the audit trail were written in reranker output order, not score order. The reranker returns a ranked list but the scores assigned by the reasoning model can disagree with that ranking. Result: a job with score 55 could appear above a job with score 85 in the saved file.

**Fix:** `save_full_audit_trail` now sorts `job_explanations` by score descending before writing, using a `score_map = {j.job_id: j.score for j in jobs}` lookup.

### Empty career roadmap

`overall_missing_skills` was blank in 5 of 6 test runs. When all retrieved jobs are entry-level marketing roles that match the candidate perfectly, the reasoning model returned an empty list — technically correct but useless for the user.

**Fix:** Added an explicit rule to `src/prompts/reasoning.md`: `overall_missing_skills` must never be empty. Even for strong matches, the model must identify 2–3 concrete skills that appear across job descriptions but are absent from the CV.

### Empty descriptions in results (data bug)

All job descriptions in several runs showed blank. Root cause: `to_metadata()` in `schemas.py` intentionally excludes `description` (it goes into the embedded `page_content`), but `retrieve_jobs` tried to read it from metadata and always got `""`.

**Fix:** Build scripts now save a `job_descriptions_{model}.json` lookup (`{job_id: description}`). `job_search.py` loads this at startup and populates the `description` field in `JobRecord` from the lookup keyed by `job_id`.

### Duplicate jobs flooding results

GIG USA and Millennium Recruiting had the same generic title ("Entry Level Openings: Fast-Paced Marketing Team") posted many times with different job IDs. An entry-level marketing CV would get 8–9 copies of the same posting in top-10.

**Fix:** `load_kaggle` in both build scripts now deduplicates by `(title.lower(), company.lower())`, keeping the first occurrence in CSV order. Requires an index rebuild.

### Street address in current_location (cache)

`current_location` showed "123 Anywhere St., Any City" across all runs even after the street address regex filter was added to `cv_profiler`. The filter was correct but the cached profile predated it.

**Fix:** Clear the cv_profiler cache before the next run: `rm -rf project/.cache/cv_profiler`

### Chunk-level duplicate jobs in FAISS results (post-rebuild fix)

After the index rebuild, `verify_rebuild.py` still flagged 1 duplicate in top-20. Root cause: each job is split into multiple embedding chunks. Even with title+company dedup in `load_kaggle`, multiple chunks from the same job can independently score highly and both surface in `retrieve_jobs`.

**Fix:** `retrieve_jobs` in `job_search.py` now deduplicates by `job_id` at query time using a `seen_job_ids` set, keeping the highest-scoring chunk per job. `fetch_k` bumped from `top_k + 20` to `top_k + 40` to absorb both source filtering and chunk dedup overhead. No index rebuild needed — runtime fix only. Confirmed 8/8 on `verify_rebuild.py`.

### Design limitations (documented, not fixed)

- **Weak match ranked #1:** The reranker (Gemma) ranks jobs before the reasoning model sees them. If the reranker makes an error, the reasoning model flags it as "weak match" but cannot demote it. These are independent models making independent judgments. Documented as a pipeline limitation for the report.
- **Generic reasoning when description is missing:** When a job has no description, the reasoning model falls back to seniority matching only, producing copy-paste match reasons. Fixed by populating descriptions correctly (see above).
