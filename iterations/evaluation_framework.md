# Evaluation Framework — Dev Log

## decisions made

### method groups expanded to 6
added FAISS_PARSED_MPNET as 6th retrieval method per professor feedback requesting
a stronger embedding model for comparison. all-mpnet-base-v2 has a 384-token limit
vs 256 for MiniLM — better at capturing long-form context in job descriptions.

MPNet index is built separately (build_vector_store_mpnet.py) and loaded lazily in
run_evaluation.py. if index not present, script warns and continues — recoverable
by building index and re-running (existing files are skipped, only missing ones generated).

### FAISS_RAW source filter fixed
FAISS_RAW was returning Kaggle + Arbeitnow results (mixed pool) while FAISS_PARSED
filtered to Kaggle-only. this made H2 (raw vs parsed) comparison inconsistent.
fixed: FAISS_RAW now also filters to source="kaggle" with +40 over-fetch to absorb
filtering. all 4 FAISS methods now use the same job pool — H2 is apples-to-apples.

### H1 uses reranked results (deliberate)
H1 compares FAISS_PARSED vs BM25_PARSED — both reranked by Gemma. reranking is
identical for both so any P@10 difference reflects retrieval quality feeding into
a constant reranking step. this tests the real system, not an artificial raw baseline.
a BM25_PARSED_NORERANK equivalent doesn't exist and isn't worth adding.
report note: "reranking is fixed (Gemma) across all methods — P@10 differences
reflect retrieval quality feeding into a constant reranking step."

### H1 and H2 Wilcoxon tests added to score_results.py
docstring promised Wilcoxon for H1/H2/H3 but only H3 was computed. added:
  H1:  FAISS_PARSED vs BM25_PARSED   (same parsed query, different retrieval method)
  H2a: BM25_PARSED vs BM25_RAW       (keyword search, parsed vs raw query)
  H2b: FAISS_PARSED vs FAISS_RAW     (semantic search, parsed vs raw query)
refactored into _hypothesis_block() helper — all 4 hypotheses use the same format.

### qualitative observations added (no extra labeling)
two observations computed automatically from existing JSON results:
  diversity    — unique companies, experience levels, title stems in top-10 per method
  preference   — % of top-10 jobs matching stated work_type preference (full-time)
framed as "beyond precision" in the report — not new hypotheses, no statistical tests.
addresses professor point about part-time/employment preference dimension.

### labeling workflow locked
60 files total (10 personas × 6 methods × 1 gemma file each).
split across 3 teammates, Julia coordinates.
git-based merge: each person labels their assigned personas, commits, pushes.
Julia runs score_results + llm_judge once all 3 push.
existing labels are never overwritten by re-runs — skip logic on json+md existence.

## what's pending
- delete test CVs (cv1, cv2, cv3, Perosona_Finance) before real eval — superseded; real eval ran on 10 personas (cv1-3 + Perosona_Finance are stale scratch folders, safe to delete)

## what's done
- 10 real CVs labeled across all 6 methods (60 _gemini.md files)
- MPNet index built and run
- Hamid's missing 10_tech_mid FAISS_RAW row filled — H2b now fully paired (was n=9)
- score_results.py + llm_judge.py re-run on complete data; results captured in `iterations/results_analysis.md`

---

## bug fixes (2026-05-07)

### score_results.py: _hypothesis_block silent data loss
`lines += [...]` inside `_hypothesis_block` caused Python to treat `lines` as a local
variable (assignment → local scope). H1/H2/H3 blocks were computed but silently discarded
— report showed "not enough data" even with labeled personas.
fix: changed `lines +=` to `lines.extend()` — method call doesn't trigger local scope.
note: Julia's workaround (`lines = []` at top of function) would have silenced the crash
but discarded all hypothesis output. correct fix is extend().

### score_results.py: Experiment B separator phantom column
`sep2 = "|--------|" + "--------|" * len(MODELS) + "|"` produced 6 pipes for a 4-column
table (Method + 3 models = 5 pipes needed). the trailing `"|"` was extra.
fix: removed trailing `"|"` — sep2 now ends with the last `"--------|"` repetition.

---

## post-eval changes (2026-05-15)

### .gitignore: JSON results no longer excluded
`evaluation/results/**/*.json` was gitignored as "large, auto-generated, not needed in repo".
this silently broke `llm_judge.py` on any teammate who pulled fresh — judge needs the JSON
to read each job's title/description for the LLM call. without it, every (persona, method)
combo hit `if not json_path.exists(): continue` and the report came out empty.
fix: removed the `evaluation/results/**/*.json` rule. JSONs now commit normally. they ARE
the audit trail of what the retriever surfaced for each labeled run; re-running the pipeline
would produce slightly different results due to LLM non-determinism, so the captured JSONs
are the canonical reference for any future judge or re-scoring work.

### llm_judge.py: prompt v1 (lenient) → v2 (strict, rubric-mirrored)
the initial judge prompt restated the rubric in two short bullets ("domain matches" + "±1 seniority"),
producing κ = 0.227 (fair) with no negative cells but many LLM P@10 = 1.0 cells — the judge
was effectively rubber-stamping.

v2 prompt rewritten to mirror `src/prompts/reasoning.md`:
- seniority mismatch framed as **primary disqualifier**, not one factor among many
- domain mismatch caps relevance regardless of keyword overlap
- structured reasoning fields required before verdict: `candidate_seniority`, `job_seniority`, `domain_match`
- default-to-not-relevant unless both criteria clearly met
- `max_tokens` raised from 16 to 150 to fit the structured output
- regex `r'"relevant"\s*:\s*([01])'` unchanged — still extracts the verdict from the JSON

result: κ = 0.074 overall (worse), 14/60 negative-κ cells. judge swung from over-approval to
over-rejection on borderline cases; clear-no cases (NORERANK, MPNet) stayed easier and retained
moderate-to-fair κ. v1 report preserved at `evaluation/results/llm_judge_report_baseline.md`
for the report comparison.

interpretation: not a failure to write up — a methodology finding. prompt engineering alone
shifts judge bias direction but does not align judgment with human raters on borderline cases.
this is stronger evidence for the "LLM-as-judge not a substitute for human labels" claim than
either single κ value would be.

### llm_judge.py: parallelism added
inner job loop (10 jobs per persona × method) wrapped in `ThreadPoolExecutor(max_workers=5)`.
preserves call ordering via `pool.map` so `llm_labels[i]` still pairs with `human_labels[i]`.
matches the concurrency cap `reasoning.py` settled on (Iteration 5) — 5 simultaneous outbound
calls is the practical OpenRouter rate-limit ceiling for sustained bursts.
also added `time.sleep(2 ** attempt)` exponential backoff to the existing 3-retry loop;
prior code retried instantly on failure which is useless against 429s.
total runtime: ~30-50 min sequential → ~5-6 min parallel. 600 calls, same cost.
