# Dev Log: Evaluation Harness
**Date:** 2026-05-01
**Objective:** Build a batch evaluation harness that runs the full experimental matrix and generates human-labelable output for Precision@10 scoring.

---

## Scope

**Files:** `src/evaluation/run_evaluation.py`, `src/evaluation/score_results.py`
**Purpose:** Scientific evaluation of retrieval and reasoning quality across 10 personas × 4 retrieval methods × 3 reasoning models.

---

## 1. Experimental Design

### Experiment A — Retrieval Comparison (4 methods)

| Method | Retriever | Query representation |
|--------|-----------|---------------------|
| `BM25_RAW` | BM25 | Raw CV text (dumped into `skills` field) |
| `BM25_PARSED` | BM25 | Structured CVProfile fields |
| `FAISS_RAW` | FAISS | Raw CV text embedded as-is |
| `FAISS_PARSED` | FAISS | Structured CVProfile + preferences embedded |

**Controlled variable:** `DEFAULT_PREFS` is fixed for all personas — isolates retrieval effect.

**Why raw vs parsed:** Tests whether LLM parsing of the CV (cv_profiler) adds signal over raw text for retrieval. If FAISS_PARSED beats FAISS_RAW, the structured profile is adding value.

### Experiment B — Reasoning Model Comparison (3 models)

Reranking is fixed (Gemma always). The top-10 reranked jobs are passed to each reasoning model independently.

| Model | Provider |
|-------|----------|
| `gemma` | Google AI Studio (free) |
| `deepseek` | OpenRouter |
| `claude` | OpenRouter |

**Why fixed reranking:** Isolates the reasoning effect. If reranker also varied, any quality difference could come from either reranking or reasoning — unattributable.

**Limitation (documented for report):** Experiment B measures reasoning quality *conditioned on Gemma-ranked candidates*. A frozen human-curated top-10 benchmark would give purer isolation, but is out of scope.

### Output volume
10 personas × 4 methods × 3 models = **120 combinations** → 120 JSON + 120 MD files.

---

## 2. Initial Build — Critical Bug Found

**Bug:** The model loop `for model in MODELS` iterated correctly, but `rerank_jobs()` was called inside the loop *without* a `model` parameter. All 3 models ran the same Gemma reranker — Experiment B was broken before it started.

**Root cause:** `rerank_jobs()` has no model param (Gemma-only). The loop variable `model` was declared but never passed anywhere. The script silently produced 3 identical outputs per method.

**Fix:** Restructured the loop — reranking moved *outside* the model loop (runs once per method), `analyze_job_matches(provider=model)` moved *inside* (runs 3×). This is the correct Experiment B design.

```python
# BEFORE (broken):
for model in MODELS:
    top_10 = rerank_jobs(...)          # Gemma always, model ignored
    report = analyze_job_matches(...)  # model never passed

# AFTER (correct):
top_10 = rerank_jobs(...)              # once per method
for model in MODELS:
    report = analyze_job_matches(..., provider=model)  # 3x per method
```

---

## 3. MD Labeling Sheet

Added `save_md_preview()` to generate a human-readable table alongside each JSON. Teammates fill in two columns:

| Column | Used for | Who fills |
|--------|----------|-----------|
| `Relevant (0/1)` | Experiment A P@10 | One teammate per persona (domain expert) |
| `Quality (1-5)` | Experiment B reasoning quality | All three model MDs per method |

**Key point:** For P@10, teammates only need to fill `Relevant` in the `_gemma.md` file per method — jobs are identical across all 3 models for the same retrieval method (same `top_10`).

---

## 4. Score Results Script

`score_results.py` reads all labeled MDs and outputs `evaluation/results/report.md`:

- **Experiment A table:** P@10 per retrieval method × persona + row averages
- **Experiment B table:** Average quality (1–5) per model × method

Parsing logic: splits each MD table row by `|`, extracts column 2 (Relevant) and column 3 (Quality), skips header/separator rows. Only rows with valid values (`0`/`1` or `1`–`5`) are counted.

---

## 5. Code Review Fixes — Round 1

Three code reviews received. Valid fixes applied:

| Issue | Fix | Review source |
|-------|-----|---------------|
| `SentenceTransformer` loaded inside `perform_retrieval` — reloaded every call | Extracted to `_get_embed_model()` singleton, loaded once | Review 1 |
| No resume support — crashed experiment restarts from zero | Added `if json_path.exists() and md_path.exists(): continue` | Review 1 |
| Pipe characters in job titles broke MD table | Added `_md()` helper: `.replace("|", "\\|").replace("\n", " ")` | Review 1 |
| `print()` and `logger` mixed | All `print()` replaced with `logger.info()` | Review 1 |
| No timestamp in output JSON | Added `run_timestamp: datetime.now(timezone.utc).isoformat()` | Review 1 |

---

## 6. Code Review Fixes — Round 2

| Issue | Fix | Review source |
|-------|-----|---------------|
| `datetime.utcnow()` deprecated in Python 3.12 | Changed to `datetime.now(timezone.utc)` | Review 2 |
| `k=20` hardcoded in 4 places | Defined `RETRIEVAL_K = 20` and `RERANK_K = 10` constants | Review 2 |
| No warning when reranker returns < 10 jobs | Added `logger.warning` when `len(top_10) < RERANK_K` | Review 2 |
| `logger.error` loses traceback on failures | Changed to `logger.exception` for all 3 except blocks | Review 2 |

---

## 7. Code Review Fixes — Round 3

| Issue | Fix | Review source |
|-------|-----|---------------|
| `import faiss` inside function — missing dep discovered late | Moved to top-level import | Review 3 |
| `_md()` didn't strip `\r` — Windows line endings could break tables | Added `.replace("\r", "")` | Review 3 |

**Rejected feedback (with reasons):**

| Claim | Reason rejected |
|-------|----------------|
| Gemma reasoning provider broken | 39/39 tests pass. Reviewer referenced a separate review we haven't seen. |
| FAISS raw vs parsed use different embedders | Both use `all-MiniLM-L6-v2`. Two instances ≠ different models. |
| FAISS index normalization unknown | Index is built with `faiss.normalize_L2` — confirmed in build script and README. |
| `DEFAULT_PREFS` creates bias | Intentional experimental control. Fixed prefs isolates retrieval effect. |
| Rate limiting will crash at 120 calls | `reasoning.py` already has 3-attempt exponential backoff. |
| `skills=[raw_text]` is wrong | Intentional ablation design for BM25_RAW. BM25 tokenizes the query string. |
| FAISS `score` field mismatch | `search_jobs` wrapper handles mapping internally. 25/25 tests confirm. |
| Global prefs bias Experiment B | Prefs are intentionally constant to isolate model effect. |

---

## 8. Final Architecture

```
run_evaluation()
  ├── BM25Retriever (initialised once)
  ├── for each persona PDF:
  │     ├── extract_text_from_pdf()   ← cached
  │     ├── profile_cv()              ← cached
  │     └── for each method (4):
  │           ├── perform_retrieval() → top_20
  │           ├── rerank_jobs()       → top_10  (Gemma, once per method)
  │           └── for each model (3):
  │                 ├── skip if JSON+MD exist
  │                 ├── analyze_job_matches(provider=model)
  │                 ├── save JSON  (results + reasoning + timestamp)
  │                 └── save MD   (labeling sheet with Relevant + Quality columns)

score_results()
  ├── for each persona dir:
  │     ├── Experiment A: parse Relevant from _gemma.md per method → P@10
  │     └── Experiment B: parse Quality from all 3 model MDs per method → avg
  └── write evaluation/results/report.md
```

---

## Known Limitations

- Sequential retrieval + reranking — 150 combinations but reasoning runs 3× in parallel per method. For 10 personas ~1–2 hrs. Caching makes reruns near-instant.
- Labeling instructions in MD are brief. A full annotation rubric with edge case examples should be distributed separately before labeling starts.
- Experiment B is conditioned on Gemma reranking — not pure reasoning quality isolation. Documented as a limitation for the report.

---

## 9. H3 Baseline + Statistical Tests + LLM Judge (2026-05-02)

### Changes

**`run_evaluation.py` — H3 baseline added**

After FAISS_PARSED retrieval (before reranking), the script now saves `top_20[:10]` as `FAISS_PARSED_NORERANK`. This gives us the pre-rerank top-10 by raw similarity score — the control condition for H3.

Extracted the `_run_model` closure into two module-level functions:
- `_save_combo(persona_dir, persona_id, method_name, model, parsed_profile, top_10)` — runs one reasoning model and writes JSON + MD
- `_run_models_parallel(...)` — fires all 3 models via `ThreadPoolExecutor`

This fixes a Python closure capture issue and makes the parallel logic reusable.

Total combinations updated: 10 × (4 methods + 1 NORERANK) × 3 models = **150**.

**`score_results.py` — statistical tests added**

- `FAISS_PARSED_NORERANK` added to METHODS list
- `bootstrap_ci()` — 10k bootstrap samples, 95% CI shown next to each method average (requires ≥2 labeled personas)
- `wilcoxon_p()` — Wilcoxon signed-rank test via scipy
- H3 section: FAISS_PARSED vs NORERANK delta + p-value
- Experiment B section: Wilcoxon for gemma vs deepseek and gemma vs claude

**`llm_judge.py` — new file (H4)**

Reads human Relevant labels from `*_gemma.md`. Calls `claude-sonnet-4-6` via OpenRouter (temperature=0, JSON-only) for each job. Computes Cohen's Kappa per persona/method. Outputs `evaluation/results/llm_judge_report.md` with kappa interpretation guide. Supports `--method` and `--persona` flags.

Cost: ~$0.30 for 4 personas × 5 methods × 10 jobs.

---

### 10. Gemma API Reliability Fixes (2026-05-02)

**Problem:** `genai.Client.models.generate_content()` hangs indefinitely — Google AI Studio free tier drops connections under load. The SDK has no native timeout parameter.

**Attempts that failed:**
- `request_options=types.RequestOptions(timeout=90)` → `types` has no `RequestOptions`
- `request_options={"timeout": 90}` → `generate_content()` doesn't accept `request_options`
- `with ThreadPoolExecutor(max_workers=1) as ex: future.result(timeout=90)` → `with` block calls `shutdown(wait=True)` on exit, blocking until the thread finishes regardless of timeout

**Working fix:** Remove the `with` statement, use `ex.shutdown(wait=False)` on timeout so the caller gets `TimeoutError` immediately while the background thread is abandoned:

```python
ex = ThreadPoolExecutor(max_workers=1)
future = ex.submit(_do_call)
try:
    return future.result(timeout=60)
except TimeoutError:
    ex.shutdown(wait=False)
    raise TimeoutError("Gemma timed out after 60s")
```

Applied to both `reasoning.py` (`_call_gemma`) and `reranker.py` (`_call_llm`).

**OpenRouter fallback added to reasoning.py:**

When Google AI Studio times out, `_call_llm` now falls back to `google/gemma-4-31b-it` via OpenRouter (which has a 60s timeout already set on the client):

```python
except TimeoutError:
    logger.warning("[gemma] Google AI Studio timed out — falling back to OpenRouter")
    return _call_openrouter(..., model_override="google/gemma-4-31b-it")
```

**Markdown fence stripping added to `_call_gemma`:**

Despite `response_mime_type="application/json"`, Gemma sometimes wraps output in ` ```json...``` ` fences. Added the same stripping logic already present in `_call_openrouter`.

**Why this matters:** These fixes are not just for evaluation — they harden the production path in `main.py` too. Any long Gemma call in the interactive demo will now timeout and retry cleanly instead of hanging the CLI.

---

## Updated Architecture (as of 2026-05-02)

```
run_evaluation()
  ├── BM25Retriever (initialised once)
  ├── for each persona PDF:
  │     ├── extract_text_from_pdf()   ← cached
  │     ├── profile_cv()              ← cached
  │     └── for each method (4):
  │           ├── perform_retrieval() → top_20
  │           ├── [FAISS_PARSED only] save top_20[:10] as FAISS_PARSED_NORERANK (H3 baseline)
  │           ├── rerank_jobs()       → top_10  (Gemma + OpenRouter fallback, once per method)
  │           └── _run_models_parallel() → 3 parallel reasoning calls
  │                 ├── skip if JSON+MD exist
  │                 ├── _save_combo(provider=model)  [Gemma falls back to OpenRouter on timeout]
  │                 └── save JSON + MD

score_results()
  ├── for each persona dir:
  │     ├── Experiment A: P@10 from _gemma.md + 95% bootstrap CI
  │     ├── H3: FAISS_PARSED vs NORERANK delta + Wilcoxon p
  │     └── Experiment B: avg quality per model + Wilcoxon (gemma vs deepseek/claude)
  └── write evaluation/results/report.md

llm_judge()
  ├── for each persona × method:
  │     ├── parse human labels from _gemma.md
  │     ├── call claude-sonnet-4-6 (0/1) for each job
  │     └── compute Cohen's Kappa
  └── write evaluation/results/llm_judge_report.md
```

---

## 11. Polish + Parallelization (2026-05-02)

### run_evaluation.py — persona-level parallelism

Extracted `_run_persona(pdf_path, bm25)` — full method × model loop for one persona.
Outer loop replaced with `ThreadPoolExecutor(max_workers=4)` using `as_completed`.

Personas are fully independent — no shared mutable state. Previously sequential across
10 personas even though `_run_models_parallel` was already parallelizing the 3 reasoning
models per method. Now both levels are parallel.

Embed model pre-loaded before the executor starts to avoid a race on lazy init.
Counts aggregated from each persona's return value, not a shared dict.
Log lines include `[persona_id]` prefix so interleaved output is readable.
`PERSONA_WORKERS = 4` constant at top — tune down if hitting API rate limits.

```
run_evaluation()
  ├── _get_embed_model()  ← pre-loaded once before forking
  ├── BM25Retriever       ← shared, read-only during search
  └── ThreadPoolExecutor(max_workers=4):
        └── _run_persona() × 10 personas in parallel
              └── for each method (4+1):
                    ├── perform_retrieval() → top_20
                    ├── [FAISS_PARSED] save NORERANK baseline
                    ├── rerank_jobs() → top_10
                    └── _run_models_parallel() → 3 reasoning models in parallel
```

### tests/evaluation/ — new test suite

`test_score_results.py` — 24 tests, no external deps:
- `parse_md_labels`: valid table, partial fill, quality-only, out-of-range quality, missing file
- `precision_at_k`: perfect/zero/mixed, uses-first-k, too-few → None
- `bootstrap_ci`: bounds contain mean, uniform, single value → None, seeded (deterministic)
- `wilcoxon_p`: valid, too few, mismatched lengths, identical lists, clearly different → p < 0.05

`test_llm_judge.py` — 16 tests, API mocked:
- `_parse_human_labels`: basic, skips unlabeled, empty
- `cohen_kappa`: perfect, disagreement, mixed, empty, mismatched, all-same-class
- `judge_job`: relevant, not relevant, noisy JSON, malformed → None, exception → None, 3 retries → None

40/40 passing.

### documentation

`evaluation_automation.txt` — plain prose explaining all 4 hypotheses, method groups,
4-step automation flow, statistical approach, labeling rubric. For professor/teammates.

`README.md` — full rewrite. Cut from 362 to ~170 lines. Evaluation section collapsed to
4 commands + pointer to evaluation_automation.txt.

---

## 12. Labeling Sheet + Audit Trail Improvements (2026-05-02)

### Problem

Teammates annotating the MD labeling sheets had no context. They saw a table of job titles and companies but no candidate profile, no job descriptions, and no apply links — making it difficult to judge relevance accurately without running the pipeline themselves.

### save_md_preview — evaluator context added

Added `profile: CVProfile | None = None` parameter. When provided, a candidate profile summary is inserted at the top of each sheet before the label table:

```
## Candidate Profile
Level: senior (8 yrs)  |  Education: BSc in Computer Science
Industries: Fintech, Banking
Key Skills: Python, SQL, Machine Learning, ...
```

Below the label table, a **Job Details** section now includes for each job:
- Level + location + salary range
- Apply URL (LinkedIn link for Kaggle jobs)
- First 800 characters of the job description
- Match reason (from reasoning output)
- Missing skills

This gives annotators everything needed to judge relevance without opening a browser or re-running the pipeline.

`_save_combo` updated to pass `parsed_profile` to `save_md_preview` — the profile is already in scope from `_run_persona`.

### Why this matters for annotation quality

Precision@10 is computed from human labels. Mislabelled jobs inflate or deflate P@10, which directly affects which hypothesis we accept. Adding context doesn't bias the label — the annotator still decides. It just prevents wrong labels from lack of information.

---

## 13. Data Quality Fixes — Index Rebuild Required (2026-05-02)

Issues found from reviewing test run results. All require rebuilding the FAISS index.

### Description always empty in results

`to_metadata()` in `schemas.py` intentionally omits `description` (it's in the embedded `page_content`). `retrieve_jobs` tried to read it from metadata and always got `""`. Affected every job returned by FAISS — descriptions were never populated in `JobRecord`.

**Fix:** Both build scripts now save `job_descriptions_{model}.json` — a `{job_id: description}` dict written alongside the index. `job_search.py` loads it at startup and uses `job_descriptions.get(job_id, "")` in `retrieve_jobs`. The docstore stays lean (no per-chunk description duplication).

**Impact on evaluation:** The labeling sheets and all `_save_combo` outputs now include real job descriptions. Annotators can read the JD to judge relevance instead of guessing from title alone.

### Duplicate postings flooding top-10

Staffing agencies (GIG USA, Millennium Recruiting) post the same generic job title repeatedly with different job IDs. An entry-level marketing CV returned 8–9 copies of "Entry Level Openings: Fast-Paced Marketing Team" in the top-10 — useless for evaluation diversity.

**Fix:** `load_kaggle` in both build scripts deduplicates by `(title.lower(), company.lower())` before chunking. First occurrence (lowest CSV row) is kept. Removed count printed at build time.

**Impact on evaluation:** P@10 becomes meaningful. Previously, 8 identical postings could inflate P@10 artifically — a candidate who fits one "Entry Level" role technically fits all 8 copies, but that's not 8 distinct relevant jobs.

### Rebuild command

```bash
rm -rf project/.cache/cv_profiler      # clear stale street address in profile cache
python -m src.data_pipeline.build_vector_store_minilm
```

---

## 14. First Full Evaluation Run — Bugs Found & Fixed (2026-05-02)

First real execution of `run_evaluation.py` against 4 personas (cv1, cv2, cv3, Perosona_Finance). Three bugs found and fixed. All 60 combinations now complete with clean descriptions.

### Bug 1 — `.numpy()` on already-numpy array (FAISS_RAW crash)

`SentenceTransformer.encode()` returns `numpy.ndarray` directly. The code called `.numpy()` on it — a PyTorch tensor method — causing `AttributeError` on every FAISS_RAW run.

**Fix:** Changed to `encode([raw_text], convert_to_numpy=True).astype("float32")`.

**Why missed:** FAISS_RAW is an eval-only code path not exercised by any prior test or `main.py` flow. Cache also masked it on restarts.

### Bug 2 — Wrong key for description in FAISS_RAW path

`search_jobs()` returns `job_description` (from `page_content`). The `perform_retrieval` FAISS_RAW branch read `r.get("description", "")` — wrong key — so all descriptions were empty strings.

**Fix:** Changed to `r.get("job_description", r.get("description", ""))`.

**Why this wasn't enough:** `page_content` in the docstore is the embedded text (title + truncated snippet), not the full description. Root cause was different.

### Bug 3 — FAISS_RAW not using `job_descriptions` lookup (root cause)

`retrieve_jobs` (FAISS_PARSED) was fixed in the rebuild to use `job_descriptions.get(job_id, "")` — the dedicated `{job_id: description}` dict. `perform_retrieval`'s FAISS_RAW branch was never updated to do the same.

**Fix:** Imported `job_descriptions` from `job_search` and used `job_descriptions.get(str(r.get("job_id", "")), "")` — identical to how `retrieve_jobs` works.

### Bug 4 — Reranker cache returned stale empty-description results

After fixing Bug 3, descriptions were still empty in saved JSON. Root cause: `rerank_jobs(use_cache=True)` had cached the `top_10` from the broken run. The cache key hashes `job_id + score` only — not description — so the fix didn't invalidate the cache.

**Fix:** Deleted `.cache/reranker/` to force cache rebuild.

**Lesson:** Cache key should include description or at least a data-version tag if the description source changes. For now, manual cache clear is the recovery path.

### New test file: `test_run_evaluation.py` (12 tests)

Added to prevent regression on Bugs 2–4. Tests use mocks — no real FAISS/API calls needed.

| Test | What it guards |
|------|---------------|
| `test_faiss_raw_description_comes_from_lookup_not_job_texts` | Bug 3: must use `job_descriptions` dict, not `page_content` |
| `test_faiss_raw_descriptions_non_empty` | Bug 2+3: description field populated |
| `test_faiss_raw_missing_job_id_in_lookup_returns_empty_string` | Graceful fallback |
| `test_bm25_raw_returns_records_with_descriptions` | BM25 path stays clean |
| `test_bm25_parsed_returns_records_with_descriptions` | BM25 path stays clean |
| `test_bm25_raw_passes_raw_profile_to_bm25` | Bug 2 analogue for BM25 |
| `test_faiss_parsed_delegates_to_retrieve_jobs` | Delegation not broken |
| `test_unknown_method_raises_value_error` | Fail fast on typo |
| + 4 profile/fixture tests | `get_raw_profile` contract |

All 12 passing.

### Final state

```
60/60 combinations complete (0 failed)
All 60 JSON files: 10/10 jobs with descriptions ≥ 983 chars
Ready for human labeling (Precision@10)
```

---

## 15. Bugs Found During Human Labeling (2026-05-09)

Both bugs were caught by Julia during the first real labeling pass on a tech intern persona. Neither appeared in automated tests because tests mock retrieval output — they don't exercise the full pipeline with real data.

---

### Bug 1 — BM25 missing load-time dedup

**Symptom:** Julia saw duplicate job postings in BM25 result sheets.

**Root cause:** FAISS build scripts deduplicate by `(title.lower(), company.lower())` at index build time, reducing 123,849 → ~96,728 docs. BM25 `_load_and_index()` loaded the full raw CSV without dedup — only query-time dedup existed (the `seen_title_company` check in `search()`). Query-time dedup catches exact matches within a single result set but near-duplicates (minor spelling/casing variation) in the 123,849-row corpus could both score highly and pass the check.

**Fix:** Added load-time dedup in `_load_and_index()` after all sources are loaded, before building the BM25 corpus — mirrors FAISS exactly.

**Why missed:** Code reviews examined BM25 and FAISS separately. The query-time dedup in BM25 looked correct in isolation. The gap only surfaces by comparing BM25's load path against FAISS's build path side-by-side — a cross-component audit we never did.

---

### Bug 2 — FAISS seniority filter missing from run_evaluation.py

**Symptom:** Julia flagged FAISS_PARSED_MPNET returning many senior-level jobs for the intern tech CV.

**Root cause:** `BM25Retriever.search()` applies `_passes_seniority_filter()` before returning results. The shared FAISS processing loop in `run_evaluation.py` (lines 218–249) applied only `source == kaggle` + job_id dedup — no seniority filter. All 5 FAISS methods were affected, not just MPNet. The filter existed in BM25 but was never ported to the FAISS path because FAISS calls `search_jobs()` directly (the raw vector search) instead of `retrieve_jobs()`, and `retrieve_jobs()` also has no seniority filter.

**Fix:** Added seniority hard filter inline in the shared FAISS processing loop — identical logic to `BM25Retriever._passes_seniority_filter()`:
- Senior CV: excludes `entry level`, `associate`, `internship` exp levels + junior/intern title keywords
- Entry CV: excludes `director`, `executive`, `c-suite` exp levels + VP/director/chief title keywords

**Why missed:** BM25 and FAISS retrieval paths were reviewed as separate components. The seniority filter was confirmed present in BM25. Nobody cross-checked whether the FAISS path in `run_evaluation.py` applied the same post-processing. Tests mock the retrieval layer and never exercise real `experience_level` metadata.

---

### Impact

All result files for the affected persona are invalid — both bugs affect the output. Delete the persona's result folder, re-run `run_evaluation.py` for that persona, send fresh files to Julia for re-labeling.

**Lesson:** Integration-point audits matter more than per-file reviews. For any shared post-processing (dedup, filtering), explicitly verify every retrieval path applies it — not just the one that was built first.
