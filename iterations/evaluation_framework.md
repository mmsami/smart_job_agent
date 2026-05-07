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
- 10 real CVs needed in data/resumes/ (critical path)
- MPNet index build before full eval run (run_evaluation.py handles missing index gracefully)
- delete test CVs (cv1, cv2, cv3, Perosona_Finance) before real eval

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
