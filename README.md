# Smart Job Market Agent

Takes a CV, returns the top matching jobs with explanations and skill gap analysis.
Uses semantic search (FAISS + embeddings) compared against a BM25 keyword baseline.

---

## Setup

**Option A — venv (recommended)**
```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env
# fill in your keys (see table below)
```

**Option B — conda**
```bash
conda create -n smart_job_agent python=3.11
conda activate smart_job_agent

pip install -r requirements.txt
cp .env.example .env
```

> **Mac users:** Python 3.12+ causes segfaults with faiss-cpu + sentence-transformers. Use Python 3.11.

Then download the shared data files from Google Drive and place them at:

```
data/vector_store/faiss_minilm.index
data/vector_store/docstore_minilm.json
data/vector_store/job_descriptions_minilm.json
data/vector_store/faiss_mpnet.index
data/vector_store/docstore_mpnet.json
data/vector_store/job_descriptions_mpnet.json
data/kaggle_cleaned/postings_cleaned.csv
data/resumes/                        ← persona PDFs (one per evaluator)
```

Do not re-run the data pipeline scripts — the indexes are already built and shared via Google Drive.

---

## Environment variables

| Variable | Where to get it | Used by |
|----------|----------------|---------|
| `GOOGLE_API_KEY` | aistudio.google.com/apikey (free) | cv_reader, cv_profiler, reranker, reasoning |
| `OPENROUTER_API_KEY` | provided by university | reasoning (DeepSeek, Claude), LLM judge |
| `LANGSMITH_API_KEY` | smith.langchain.com (free) | tracing for reranker + reasoning |
| `LANGSMITH_PROJECT` | set to `job-market-agent` | tracing |

---

## Running the demo

```bash
cd project
python -m src.main
```

Prompts you for a CV path and job preferences, then runs the full pipeline and prints results.

---

## Project structure

```
project/
├── src/
│   ├── main.py                      # interactive CLI — run this
│   ├── data_pipeline/               # one-time build scripts (already run)
│   │   ├── parse_kaggle.py
│   │   ├── fetch_arbeitnow.py
│   │   ├── build_vector_store_minilm.py
│   │   ├── build_vector_store_mpnet.py
│   │   └── schemas.py
│   ├── workflow/                    # pipeline components
│   │   ├── models.py                # JobRecord, CVProfile, JobSearchPreferences
│   │   ├── mocks.py                 # test fixtures (3 personas, 10 jobs)
│   │   ├── cv_reader.py             # PDF → raw text (Gemini vision, cached)
│   │   ├── cv_profiler.py           # raw text → CVProfile (Gemini, cached)
│   │   ├── job_search.py            # CVProfile → top 20 jobs (FAISS cosine)
│   │   ├── reranker.py              # top 20 → top 10 (Gemma batch, cached)
│   │   └── reasoning.py             # top 10 → explanations + skill gaps (3 providers)
│   ├── evaluation/
│   │   ├── baseline_bm25.py         # BM25 keyword retriever
│   │   ├── run_evaluation.py        # generates full result matrix (4 personas × 5 methods × 3 models)
│   │   ├── score_results.py         # computes P@10, CI, Wilcoxon from labeled MDs
│   │   └── llm_judge.py             # Claude-as-judge → Cohen's Kappa (H4)
│   └── prompts/                     # edit these to tune LLM behavior
│       ├── cv_profiler.md
│       ├── reranker.md
│       └── reasoning.md
├── tests/
│   ├── workflow/
│   │   ├── test_mocks.py
│   │   ├── test_bm25.py
│   │   ├── test_cv_reader.py
│   │   ├── test_cv_profiler.py
│   │   ├── test_job_search.py
│   │   ├── test_reranker.py
│   │   ├── test_reranker_integration.py
│   │   └── test_reasoning.py
│   └── evaluation/
│       ├── test_run_evaluation.py   # perform_retrieval correctness (12 tests)
│       ├── test_score_results.py    # P@10, CI, Wilcoxon (24 tests)
│       └── test_llm_judge.py        # Cohen's Kappa + judge calls (16 tests)
├── data/
├── evaluation/
│   └── results/                     # output of run_evaluation.py
├── iterations/                      # dev notes per component
└── evaluation_automation.txt        # hypotheses, scoring, labeling rubric
```

---

## Running the evaluation

See `evaluation_automation.txt` for the full explanation of hypotheses, method groups,
labeling rubric, and statistical approach.

```bash
# step 1 — generate results (skips already-complete runs)
cd project
python -m src.evaluation.run_evaluation

# step 2 — label the MD files manually
# open evaluation/results/<persona>/<METHOD>_gemma.md
# fill Relevant (0/1) column — one model per method is enough for P@10
# fill Quality (1-5) in all 3 model MDs for explanation quality comparison

# step 3 — compute scores
python -m src.evaluation.score_results
# outputs evaluation/results/report.md

# step 4 — LLM judge
python -m src.evaluation.llm_judge
# outputs evaluation/results/llm_judge_report.md
```

### What run_evaluation.py produces

10 personas × 6 method variants × 3 reasoning models = **180 result files** (JSON + MD each).

| Method | Retriever | Query |
|--------|-----------|-------|
| `BM25_RAW` | BM25 | raw CV text |
| `BM25_PARSED` | BM25 | structured CVProfile |
| `FAISS_RAW` | FAISS | raw CV text embedded (MiniLM) |
| `FAISS_PARSED` | FAISS | CVProfile + preferences embedded (MiniLM) |
| `FAISS_PARSED_NORERANK` | FAISS | FAISS_PARSED top-10 before reranking (H3 baseline) |
| `FAISS_PARSED_MPNET` | FAISS | CVProfile + preferences embedded (MPNet) |

Reranking is always Gemma (fixed). Reasoning runs with gemma / deepseek / claude per method.
All runs skip if the output JSON + MD already exist — safe to restart after failure.

---

## Tests

```bash
cd project

# no external files needed
python -m pytest tests/workflow/test_mocks.py -v
python -m pytest tests/workflow/test_reranker.py -v
python -m pytest tests/workflow/test_reasoning.py -v
python -m pytest tests/evaluation/test_run_evaluation.py -v
python -m pytest tests/evaluation/test_score_results.py -v
python -m pytest tests/evaluation/test_llm_judge.py -v

# needs postings_cleaned.csv
python -m pytest tests/workflow/test_bm25.py -v

# needs vector store files
python -m pytest tests/workflow/test_job_search.py -v

# needs GOOGLE_API_KEY (cached after first run)
python -m pytest -m integration tests/workflow/test_reranker_integration.py -v

# needs GOOGLE_API_KEY + PDFs in data/resumes/
python -m tests.workflow.test_cv_profiler

# verify build pipeline works before full rebuild
python -m tests.data_pipeline.test_build_smoke
```

Total: **117 tests** across workflow + evaluation.

---

## Data

```
data/
├── kaggle_cleaned/
│   └── postings_cleaned.csv         123,849 rows (raw; dedup to ~96,728 unique title+company in build)
├── arbeitnow/
│   └── arbeitnow_jobs.json          957 jobs (frozen API snapshot)
├── vector_store/                    (shared via Google Drive)
│   ├── faiss_minilm.index           all-MiniLM-L6-v2, 384 dims
│   ├── docstore_minilm.json         chunk metadata (parallel to index)
│   ├── job_descriptions_minilm.json {job_id: full_description} lookup
│   ├── faiss_mpnet.index            all-mpnet-base-v2, 768 dims
│   ├── docstore_mpnet.json          chunk metadata (parallel to index)
│   └── job_descriptions_mpnet.json  {job_id: full_description} lookup
└── resumes/                         persona PDFs (one per evaluator)
```

`job_descriptions_*.json` files are required — without them, FAISS retrieval returns jobs with empty descriptions.

---

## Caching

All LLM components cache to disk under `.cache/`. Same input = zero API cost on re-run.

To force a fresh run after changing logic, clear the relevant cache:

```bash
rm -rf .cache/cv_reader .cache/cv_profiler .cache/reasoning .cache/reranker
```

Or bump the `LOGIC_VERSION` / `PROMPT_VERSION` constant inside the file.

| Prompt changed | Clear this | What re-runs |
|---------------|------------|-------------|
| `reasoning.md` | `.cache/reasoning` | 60 reasoning calls |
| `reranker.md` | `.cache/reranker` | reranking + 60 reasoning calls |
| `cv_profiler.md` | `.cache/cv_profiler` | everything downstream |

**Note:** If job descriptions change (index rebuild), clear `.cache/reranker` — the reranker cache key hashes job IDs + scores only, not descriptions.

---

## Mock data

```python
from src.workflow.mocks import (
    mock_cv_mid_tech,
    mock_cv_senior_finance,
    mock_cv_junior_hr,
    mock_preferences_mid_tech,
    mock_job_records,
)
```

Use these to develop and test components without the full stack.
