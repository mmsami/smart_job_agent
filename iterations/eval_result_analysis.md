# Results Analysis — Final Evaluation Run

**Date:** 2026-05-15
**Inputs:** `evaluation/results/report.md` (P@10 + H1-H3 Wilcoxon), `llm_judge_report.md` (κ, strict prompt), `llm_judge_report_baseline.md` (κ, lenient prompt), 180 JSON results across 10 personas × 6 methods × 3 reasoning models.

---

## 1. Headline result

The reranker is the only retrieval-pipeline component with a statistically detectable benefit at n=10. Parser, semantic-vs-keyword choice, and embedding model size show no significant effect. LLM-as-judge cannot be substituted for human raters at this scale — two prompt configurations produced opposite biases, neither aligning with human labels.

---

## 2. Precision@10 by method

| Method                  | P@10 | 95% CI       |
|-------------------------|------|--------------|
| FAISS_PARSED            | 0.53 | [0.45, 0.61] |
| FAISS_RAW               | 0.52 | [0.37, 0.66] |
| BM25_PARSED             | 0.50 | [0.39, 0.60] |
| BM25_RAW                | 0.48 | [0.31, 0.64] |
| FAISS_PARSED_NORERANK   | 0.36 | [0.29, 0.43] |
| FAISS_PARSED_MPNET      | 0.33 | [0.23, 0.44] |

Top 4 methods cluster within 5 percentage points. The two ablation variants (no reranker, MPNet embedding) sit clearly below.

---

## 3. Hypothesis tests

| Hypothesis | Comparison                                  | Delta   | Wilcoxon p | Verdict        |
|------------|---------------------------------------------|---------|------------|----------------|
| H1         | FAISS_PARSED vs BM25_PARSED                 | +0.030  | 0.3438     | null           |
| H2a        | BM25_PARSED vs BM25_RAW                     | +0.020  | 1.0000     | null           |
| H2b        | FAISS_PARSED vs FAISS_RAW                   | +0.010  | 0.9766     | null           |
| H3         | FAISS_PARSED vs FAISS_PARSED_NORERANK       | +0.170  | **0.0117** | **significant** |
| (descriptive) | MiniLM (FAISS_PARSED) vs MPNet           | +0.200  | n/a        | MiniLM wins    |

**H3 interpretation.** The reranker contributes +17 percentage points P@10 against a paired baseline. Effect is large relative to within-method variance.

**H1/H2 null interpretation.** Not "no effect" — "no detectable effect at n=10." H2 in particular hides a real but directionally inconsistent per-persona effect (see §6). Frame as inconclusive, not refutation.

---

## 4. LLM judge experiment (H4)

Two prompt configurations tested. Same Claude Sonnet 4.6 model, same human labels, same 600 jobs.

| Prompt              | Overall κ | Negative-κ cells | Behavior                                         |
|---------------------|-----------|-------------------|--------------------------------------------------|
| Lenient (initial)   | 0.227     | 0 / 60            | Over-approves; many cells at LLM P@10 = 1.0       |
| Strict (rubric)     | **0.074** | 14 / 60           | Over-rejects; flips on borderline cases           |

**Strict prompt construction.** Mirrors `src/prompts/reasoning.md` rubric: seniority mismatch framed as primary disqualifier, domain mismatch caps relevance, structured reasoning fields (`candidate_seniority`, `job_seniority`, `domain_match`) committed before verdict, default-to-not-relevant fallback. Increased `max_tokens` from 16 to 150 to fit structured output.

**Per-method κ comparison.**

| Method                  | κ lenient | κ strict  | Direction |
|-------------------------|-----------|-----------|-----------|
| BM25_RAW                | 0.111     | -0.053    | worse     |
| BM25_PARSED             | 0.222     | 0.184     | worse     |
| FAISS_RAW               | 0.121     | -0.041    | worse     |
| FAISS_PARSED            | 0.210     | -0.007    | worse     |
| FAISS_PARSED_NORERANK   | 0.379     | 0.208     | worse     |
| FAISS_PARSED_MPNET      | 0.320     | 0.153     | worse     |

**Observation.** Methods retrieving obviously-bad jobs (NORERANK, MPNet) have the highest κ in both runs — both humans and LLM easily reject these. Methods retrieving borderline jobs (FAISS_PARSED, BM25_PARSED) show κ collapse under the strict prompt. The judge can agree on clear cases but not on borderline cases — those require subjective tolerance the rubric does not pin down.

**Methodology finding.** Prompt engineering alone shifts the bias direction (lenient → strict) but does not align LLM judgment with human raters on borderline relevance decisions. Off-the-shelf LLM judges are not a substitute for human labels on this task.

---

## 5. Qualitative observations (no labeling required)

Computed from JSON top-10 results.

**Diversity.** All methods produce 9.2-9.7 unique companies per top-10 (max 10) — no retriever clustering on single employers. Title stem diversity 7.7-9.1.

**Seniority spread.** 1.4-2.4 unique experience levels per top-10. Reranker constrains seniority correctly; tight spread is desired, not a defect.

**Preference satisfaction.** Fraction of top-10 with `work_type = full-time` (the persona preference):

| Method                  | % full-time |
|-------------------------|-------------|
| BM25_RAW                | 83%         |
| FAISS_RAW               | 83%         |
| FAISS_PARSED            | 82%         |
| BM25_PARSED             | 81%         |
| FAISS_PARSED_NORERANK   | 78%         |
| FAISS_PARSED_MPNET      | 70%         |

Reranker version 4 points higher than NORERANK — reranker indirectly improves preference alignment. Kaggle `work_type` metadata coverage is incomplete; numbers directional only.

---

## 6. Per-persona signal

Borderline cases where method-level summary hides real per-persona effects.

| Persona          | Observation                                                                 |
|------------------|------------------------------------------------------------------------------|
| 04_tech_senior   | Parser rescued bad raw query: FAISS_RAW 0.10 → FAISS_PARSED 0.60.            |
| 03_tech_intern   | Parser broke good raw query: FAISS_RAW 0.70 → FAISS_PARSED 0.40.             |
| 10_tech_mid      | BM25 fails (0.10-0.20) but FAISS_RAW recovers (0.70). CV vocabulary mismatch.|
| 06_cook_senior   | All methods 0.60-0.80. Distinct domain → easy persona. Floors the variance.  |
| 09_hr_mid        | All methods 0.20-0.50. Hard persona. Possibly ambiguous CV signals.          |
| 07_cook_mid      | Strict LLM judge collapses here: human 0.80 → LLM 0.10 on BM25_RAW.          |

03_tech_intern and 04_tech_senior cancel directionally → contributes to H2 nulls. The effect is real, just not consistent in sign across personas.

---

## 7. Experiment B (reasoning quality)

Only BM25_PARSED labeled across all 3 reasoning models. Other 5 methods unlabeled.

| Method      | Gemini | DeepSeek | Claude |
|-------------|--------|----------|--------|
| BM25_PARSED | 4.20   | 4.60     | 4.60   |

DeepSeek and Claude tied; Gemini 0.40 points lower. Sample insufficient for Wilcoxon. Not blocking — H-B can be reported as preliminary directional evidence with a labeling-incomplete caveat.

---

## 8. What this means for the report

**Story to tell.**
1. Reranker validated as primary value driver (+17pp P@10, p=0.01).
2. Parser, FAISS-vs-BM25, and embedding-model choice show no detectable effect at n=10 — honest null results.
3. LLM-as-judge probed twice; both prompts miscalibrated against human labels in opposite directions. Methodology finding stronger than either single κ value.
4. Top-10 results are diverse and seniority-constrained; reranker indirectly improves preference alignment.

**What not to overclaim.**
- Don't say FAISS beats BM25. Don't say parsing helps. Both are null.
- Don't say κ=0.227 is a working judge. Don't say κ=0.074 is failure. Report both with interpretation.
- Don't say MPNet is bad — it's an honest negative result for this dataset's title-keyword-heavy retrieval pattern.

**Limitations to acknowledge.**
- n=10 underpowered for H1/H2.
- Single metric (P@10 at fixed cutoff). No nDCG, no MRR.
- Reasoning quality (Experiment B) labeled only on BM25_PARSED.
- LLM judge calibration tested with prompt engineering only; few-shot anchoring + inter-rater human κ left to future work.

---

## 9. Files of record

- `evaluation/results/report.md` — P@10 + H1-H3 Wilcoxon (canonical)
- `evaluation/results/llm_judge_report.md` — κ under strict prompt (latest)
- `evaluation/results/llm_judge_report_baseline.md` — κ under lenient prompt (snapshot)
- `evaluation/results/<persona>/<METHOD>_<model>.{md,json}` — per-run labels + raw output
