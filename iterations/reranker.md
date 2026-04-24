# Dev Log: Reranker Prompt (reranker.md)
**Date:** 2026-04-24
**Objective:** Design and validate the LLM reranker prompt that takes 20 FAISS-retrieved jobs and returns the top 10 best matches for a candidate, with scores and reasoning.

---

## Scope

**Input:** CVProfile + JobSearchPreferences + 20 JobRecords from upstream FAISS retrieval
**Output:** Ranked list of top 10 jobs, each with a score (0–100) and 1–3 sentence reasoning
**Model:** To be determined (same LLM stack as cv_profiler — Gemma 4 31B or equivalent)
**Position in pipeline:** Stage 3 — after CV extraction (Stage 1) and FAISS retrieval (Stage 2)

---

## Why a Reranker Step

FAISS retrieval is fast but shallow — it matches embedding similarity, not semantic fit. A Finance candidate searching "accounting manager" may retrieve software engineering jobs because both mention "management" and "reporting." The reranker adds a reasoning layer that enforces:
- Seniority match (entry/mid/senior)
- Domain alignment (Finance vs. Tech vs. Hospitality)
- Role realism (a Chef CV should not rank a CFO role highly)
- Candidate-stated preferences (target location, remote, work type)

The retriever maximizes recall; the reranker maximizes precision.

---

## 1. Initial Design — Teammate Contribution (2026-04-24)

Teammate (Hamid) wrote the first version of the prompt. Structure and scoring intent were sound.

### What was good
- Five-dimension scoring rubric (skills, experience, role/domain, location, red-flag penalty) maps cleanly to CVProfile fields
- 20→10 reduction matches the pipeline design
- `job_id + score + reasoning` output structure is machine-parseable
- Red-flag penalty (0 to -15) is the right mechanism for penalizing obvious mismatches
- "Return strict JSON only" instruction is critical for downstream parsing
- Anti-gaming rules ("do not simply repeat the search score") are good guardrails

### Gaps found

**Gap 1 — Missing `JobSearchPreferences` as input (critical)**
The prompt received only `CVProfile + JobRecords`. But `JobSearchPreferences` is a separate object in the schema with `target_location`, `work_type`, `remote_preference`, `willing_to_relocate`, `target_roles`, and `industry_preference`.

Dimension 4 (location/work arrangement, 10 pts) was implicitly scoring against `CVProfile.current_location` — where the candidate *is*, not where they *want* to work. A candidate based in Berlin targeting remote US jobs would be penalized 10 pts incorrectly on every job.

Fix: added `JobSearchPreferences` as a third input. Updated dimension 4 to explicitly reference `target_location`, `work_type`, `remote_preference`, `willing_to_relocate`.

**Gap 2 — `skill_labels` field not mentioned**
`JobRecord` has a `skill_labels` field (structured keyword tags from the job source). Dimension 1 only referenced job description text. The `skill_labels` field is a higher-signal source for skills matching than free-form description parsing.

Fix: added "Use both the job description and the `skill_labels` field if present" to dimension 1.

**Gap 3 — Location red-flag condition incomplete**
The red-flag penalty listed "major location mismatch when role appears location-bound" but did not account for `willing_to_relocate`. A candidate willing to relocate should not be penalized for location mismatch.

Fix: tightened to "major location mismatch when role appears location-bound *and candidate is not willing to relocate*."

**Gap 4 — Unclosed JSON code block**
The output example was missing the closing triple-backtick. Minor formatting bug but could confuse some LLMs that parse the prompt literally.

Fix: added closing fence.

---

## 2. Iteration 1 — Prompt Update (2026-04-24)

Changes applied to `src/prompts/reranker.md`:

| Area | Change |
|------|--------|
| Input spec | Added `JobSearchPreferences` as input 2; JobRecords moved to input 3 |
| Dimension 1 | Added "`skill_labels` field if present" to skills evidence sources |
| Dimension 4 | Rewrote to explicitly use `target_location`, `work_type`, `remote_preference`, `willing_to_relocate` from `JobSearchPreferences` |
| Red-flag rule | Added "and candidate is not willing to relocate" condition |
| Important rules | Added `JobSearchPreferences` to evidence sources sentence |
| Output block | Closed missing ` ``` ` fence |

**Intentionally not changed:**
- Scoring weights (40/25/20/10/−15) — reasonable distribution, no evidence to change without test data
- Salary fields (`min_salary`/`max_salary` on JobRecord) — no salary preference exists in `JobSearchPreferences`, so scoring salary would require an input that does not exist

---

## 3. LLM-as-Judge Evaluation (2026-04-24)

Before implementing the reranker in code, ran the prompt against 3 different LLMs using the mock senior finance persona (`mock_cv_senior_finance` + `mock_preferences_senior_finance` + 10 mock JobRecords k001–k010). Evaluated each output using a structured judge prompt across 6 dimensions (selection quality, ranking order, score calibration, reasoning grounding, preference adherence, red-flag enforcement).

### Test setup
- CVProfile: `mock_cv_senior_finance` — 12yr senior finance, CPA+MBA, NYC, SAP/SQL/GAAP/FP&A
- JobSearchPreferences: target NYC, hybrid, full-time, not willing to relocate
- Jobs: 10 records — 4 finance (k001/k004/k008/k010), 3 backend tech (k002/k005/k009), 2 HR (k003/k006), 1 junior frontend (k007)
- Expected: top 4 = finance jobs, bottom 6 = tech/HR with very low scores (<20)

### Results

| Output | Total | Grade | Strongest | Weakest |
|--------|-------|-------|-----------|---------|
| LLM A | 78/100 | C+ | Clean domain penalization | Bottom-band scores too high (HR jobs got 15–18) |
| LLM B | 80/100 | B- | Preference usage (explicitly cited willing_to_relocate) | Ranking inversion: k008 above k004; score inflation at extremes (98/0) |
| LLM C | 91/100 | A | Score calibration + explicit red-flag reasoning | Minor: k001 above k010 (debatable) |

### Findings that required prompt changes

**Finding 1 — No score ceiling for domain mismatches**
LLM A gave k003 (HR Coordinator) a 18 and k006 a 15 for a senior finance candidate. The rubric allows this because each dimension bottoms out at 0, and the red-flag cap is only -15. A complete domain mismatch can still accumulate spurious partial scores from vague dimension overlap.

**Finding 2 — Reasoning rarely cited JobSearchPreferences**
LLM A scored 6/10 on preference adherence — only 2 of 10 reasoning strings referenced preferences at all. The instruction "use evidence from CVProfile, JobSearchPreferences, and job fields" appears in the rules section but is easy to ignore when writing reasoning strings. Needs enforcement at the output level.

**Finding 3 — Over-qualification not addressed**
LLM B ranked k008 (Director of Finance, 15yr required) above k004 (Finance Manager, 8-10yr required). Candidate has 12 years — under-qualified for k008, over-qualified for k004. The prompt only described seniority mismatch going one direction (candidate too junior). Over-qualification was a blind spot.

---

## 4. Iteration 2 — Prompt Update from Judge Findings (2026-04-24)

Changes applied to `src/prompts/reranker.md`:

| Area | Finding | Change |
|------|---------|--------|
| Scoring rubric intro | Finding 1 | Added domain mismatch cap: "If a job is in a completely unrelated domain, the total score must not exceed 20 regardless of other factors" |
| Dimension 2 | Finding 3 | Added over-qualification note: candidate significantly more senior than role requires may be penalized 5–10 pts within dimension 2 score |
| Output requirements | Finding 2 | Added: "When location, work type, or relocation is a factor, reasoning must explicitly reference the relevant JobSearchPreferences field" |

**Intentionally not changed:**
- Scoring weights — LLM C got A grade with existing weights; no evidence to change them
- Reasoning length ("1–3 sentences") — all three outputs were appropriately concise

---

## 5. Iteration 3 — External Feedback Review (2026-04-24)

External reviewer identified 7 issues. Evaluated each against observed test data.

### Applied (5 fixes)

| Issue | Fix |
|-------|-----|
| Rubric sums to 95, not 100 — no job can ever reach 100 | Bumped Experience 25→30 (Skills 40 + Experience 30 + Role 20 + Location 10 = 100). Updated dimension 2 ranges accordingly (26-30 / 15-25 / 0-14) |
| Over-qualification penalty placement ambiguous (dim 2 vs red-flag) | Clarified: penalty applies within dimension 2 score only. Added "Do not apply a separate red-flag penalty for this" |
| Domain mismatch cap + "highly unrelated domain" red flag = double penalty | Removed "highly unrelated domain" from red-flag list — domain cap already handles it |
| No tie-breaking rule | Added: "break ties by higher skill alignment first, then by original search score" |
| "20 jobs" hard-coded in description contradicts flexible return rule | Removed hard-coded "20" from task description and input spec |

### Skipped (2 items)

| Issue | Reason |
|-------|--------|
| Missing JSON schemas for CVProfile / JobSearchPreferences / JobRecord | LLM receives actual JSON at runtime and reads field names directly. All 3 test outputs used correct field names without inline schemas. Adding schemas would bloat the prompt with no observed benefit |
| Reasoning should cite rubric components | Would make reasoning verbose and harder to read downstream. Current "1–3 sentence factual explanation" is sufficient |

---

## 6. Iteration 4 — Second External Feedback Review (2026-04-24)

Second external reviewer identified 6 issues. Schema and field-level verification done before applying any changes.

### Schema check findings
- `JobRecord` has no `years_experience` field — only `experience_level`. Dimension 2 must infer required years from `description` text. Added explicit note to the prompt.
- `JobRecord` has no `local_only` flag — relocation logic must reference `description` text, not a structured field. Handled in fix 3 wording.

### Applied (4 fixes)

| Issue | Fix |
|-------|-----|
| Model may inflate bad job scores to reach count of 10 | Added rule: "Low scores are acceptable for bottom-ranked jobs — do not inflate a score to justify inclusion" |
| "Seniority mismatch" in dim 5 could still catch over-qualification despite dim 2 warning | Tightened dim 5 wording: "under-qualification only — over-qualification is already handled in dimension 2, do not penalize it again here" |
| Willing-to-relocate but job doesn't specify — model guesses | Added neutral score guidance (4-7 range) for relocation cases; clarified model must infer local-only restriction from `description` text, not assume it |
| Domain mismatch cap positioned at top of rubric — confusing calculation order | Moved to a "Final score calculation" section after all 5 dimensions as a post-calculation override |

### Also fixed (schema-driven)
- Dimension 2: added note that `JobRecord` has no `years_experience` field — model must infer required years from job `description` text

### Skipped (2 items)

| Issue | Reason |
|-------|--------|
| LLM bad at mental math — add scratchpad | Adds tokens and complexity. Rubric is guidance not mandated arithmetic; scores are holistic approximations by design |
| 50-word reasoning limit + escape quotes instruction | Over-engineering. Modern LLMs handle JSON string escaping reliably; arbitrary word count adds friction |

---

## 7. Iteration 5 — Third External Feedback Review (2026-04-24)

Third reviewer identified 9 issues. All assessed against schema and existing prompt before applying.

### Applied (5 fixes)

| Issue | Fix |
|-------|-----|
| "Infer years from description" too loose — model may hallucinate precision | Constrained: "only infer if explicitly stated or strongly implied (e.g., '5+ years', 'senior', 'junior', 'lead'). Otherwise rely on experience_level" |
| Reasoning with 5 dimensions and 1-3 sentences leads to random dimension selection | Added: "focus on the 1–2 most impactful factors" to output requirements |
| Implicit/unclear job location — no default specified | Added: "if job location constraints are unclear, assume flexibility unless explicitly stated otherwise" |
| Subjective language in reasoning ("great fit", "strong candidate") | Added: "avoid subjective language — use concrete evidence only" to output requirements |
| Domain cap vs red-flag precedence not explicit | Added: "apply the domain cap after all scoring and penalties — do not additionally penalize for domain mismatch in the red-flag section" |

### Skipped (4 items)

| Issue | Reason |
|-------|--------|
| "Exactly 10" forces poor matches | Already handled in iteration 4: "low scores acceptable, do not inflate" |
| Tie-breaking rewording | Negligible improvement over existing wording |
| Diversity across roles | Not our use case — 471k diverse jobs in FAISS, unlikely to surface 10 near-identical roles |
| Missing fields handling | CVProfile is Pydantic-validated upstream before reaching reranker. JobRecord optional fields are typed Optional in schema. Handled at schema level |

---

## 8. Iteration 6 — Fourth External Feedback Review (2026-04-24)

Reviewer pushed again for inline schemas for CVProfile, JobSearchPreferences, and JobRecord.

**Assessed and partially rejected.** Reviewer's proposed schema contained two type errors:
- `remote_preference` described as boolean — actual type is `str` (`"remote" | "hybrid" | "onsite" | "flexible"`)
- `skill_labels` described as list of strings — actual type is `Optional[str]` (comma-separated string)

Adding the reviewer's schema verbatim would introduce wrong type information — a regression. All three test LLMs in iteration 3 used correct field names from actual JSON at runtime without any inline schemas.

**One valid narrow fix:** `skill_labels` is explicitly referenced in dimension 1 but its format was unspecified. Added "(comma-separated string)" to the dimension 1 description.

| Change | Applied |
|--------|---------|
| Clarify `skill_labels` format in dimension 1 | ✅ Added "(comma-separated string)" |
| Full inline schema block | ❌ Skipped — reviewer's schema had wrong types; LLMs read actual JSON at runtime |

---

## 9. Implementation — reranker.py (2026-04-24)

Prompt iteration complete. Moved to code.

### Design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM call count | 1 batch call for all 20 jobs | 20 separate calls would exhaust quota and add ~40s latency |
| API | google-genai SDK, `gemma-4-31b-it` | Same stack as cv_profiler — consistent, already validated |
| Response schema | `_RerankResponse` Pydantic model passed as `response_schema` | Enforces `reranked_jobs[].job_id/score/reasoning` at API level, reduces retry rate |
| Caching | diskcache, key = MD5(cv + prefs + job_ids), 1 GB cap | Same pattern as cv_profiler. Static dataset — descriptions don't change between runs |
| Description truncation | 2,500 chars per job | ~50k chars total for 20 jobs, well within context limit |
| Output sort | Explicit `sort(key=score, reverse=True)` after LLM output | LLM order not guaranteed even with response_schema |
| "Lost in the Middle" fix | Applied to reranker OUTPUT (top-10 list) | Fix is for reasoning step (Step 3), not for this LLM. Best first, second-best last |
| Injection guard | Prepended to system prompt before rubric | Job descriptions are untrusted — prevents "Rate me #1" attacks |

### Validation checks in `_validate_output`

- `reranked_jobs` key present and non-empty
- `len >= min(10, len(input_jobs))` — rejects incomplete LLM output
- Duplicate `job_id` check
- Numeric type check on `score` — `float("high")` raises TypeError not ValueError, bypasses retry without explicit check
- All `job_id` values must exist in input set

### Code review findings (3 external reviews)

**Review 1 — two "critical" issues, both wrong:**
- Claimed `google-genai` SDK doesn't support Gemma → False. AI Studio serves Gemma 4 via this SDK. cv_profiler uses the identical pattern and is validated working.
- Claimed Lost-in-Middle fix is misplaced → False. Reviewer didn't know Step 3 (reasoning) exists. Fix is for that LLM, not this one.

**Review 2 — same Lost-in-Middle misread, but reviewer self-corrected:**
- "Applying to output only helps if another LLM reads it later" — correct, and that's exactly what happens. Their own 4-step summary described the current implementation.
- `response_schema` suggestion: valid. Applied.

**Review 3 — best review, actionable:**
- Sort output by score explicitly → applied
- Min item count validation with `min(10, len(jobs))` threshold → applied
- Duplicate job_id check → applied
- Numeric score type check → applied
- Cache size limit (1 GB) → applied
- MD5 → SHA256, cache key content expansion, timeout, sentence-boundary truncation → all skipped (over-scoped for static dataset research project)

### Test coverage

30/30 tests passing (`tests/workflow/test_reranker.py`). Covers:
- Truncation (short/long/exact)
- User message construction
- Validate output: valid, sorted, missing key, too few results, unknown job_id, duplicate job_id, non-numeric score, non-dict, missing score
- Lost-in-Middle: order, single/two/ten jobs, preserves all ids
- Input hash: deterministic, differs on different input
- rerank_jobs: returns JobRecord list, scores updated, ids from input, Lost-in-Middle applied, retry on bad JSON, empty input raises, cache hit skips LLM, single LLM call

---

## 10. Integration Test + Full Suite (2026-04-24)

### Integration test (`tests/workflow/test_reranker_integration.py`)

Added 10 behavioral invariant tests that run the **real Gemma 4 LLM** on mock personas.
Run with: `pytest -m integration tests/workflow/test_reranker_integration.py -v`

Invariants tested (all pass — 10/10):
- Returns exactly 10 results
- No unknown or duplicate job_ids
- Scores valid range [0–100]
- Best job is first (Lost-in-Middle fix applied)
- Finance jobs (k001/k004/k008/k010) each score ≤ 20 (domain cap enforced)
- HR jobs (k003/k006) each score ≤ 20 (domain cap enforced)
- All tech jobs outscore all domain-capped jobs
- k002 (Stripe Python/AWS) in top 3 for mid-tech candidate
- k009 (Amazon mid-level) outscores k007 (junior frontend) — seniority logic

**Cache behaviour:** first run = real API call (~3,206 input tokens, 607 output, 3,813 total). All subsequent runs: instant from diskcache. Teammates pay quota once on first run.

**Purpose:** gate check before manual labeling — if this passes, LLM is behaving sensibly on real data.

### Full suite (`pytest tests/`)

Running the full suite after reranker work revealed a pre-existing bug in `baseline_bm25.py`:
- `url` and `experience_level` fields received pandas `nan` (float) instead of `None` when CSV values were missing
- `nan` is truthy, so `(nan or "")` returned `nan` not `""` — caused `AttributeError: 'float' object has no attribute 'lower'`
- Fix: added `_nan_to_none()` helper at module level in `baseline_bm25.py`, applied to all optional fields at `JobRecord` construction and in `_passes_seniority_filter`

**Lesson:** run full suite (`pytest tests/`) before declaring any step complete — not just the step's own tests.

Full suite result after fix: **65/65 passing**.

### LangSmith tracing

Added `wrappers.wrap_gemini()` from LangSmith SDK to `_client` at module init. All `generate_content` calls now traced automatically under project `ie_686`. Trace verified live — shows model, tokens, temperature, input/output, component tag.

Pattern for `reasoning.py`:
```python
_client = wrappers.wrap_gemini(
    genai.Client(api_key=GOOGLE_API_KEY),
    tracing_extra={"tags": ["reasoning", "gemma"], "metadata": {"component": "reasoning", "model": MODEL_NAME}},
)
```

---

## Known Limitations / Open Items

1. **`skill_labels` sparsity unknown:** Not all JobRecords have `skill_labels` populated (depends on data source — arbeitnow provides it, kaggle may not). The prompt handles this with "if present" but the practical impact on scoring is unknown until we run against real data.

2. **Salary dimension omitted:** `JobRecord` has `min_salary`/`max_salary` but `JobSearchPreferences` has no salary target field. If salary becomes a user input in future, dimension 4 could expand to cover compensation fit.

3. **Score calibration variance across runs:** LLM C's A-grade output still ranked k001 above k010 (debatable). Same job/CV pair may score differently across runs or models. Evaluation against labeled personas will quantify this.

4. **Only 10 test jobs used:** Mock data has 10 records, not 20. The prompt is designed for 20→10 selection. Full selection quality cannot be evaluated until real FAISS retrieval runs against the full dataset.
