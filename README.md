# Smart Job Market Agent

A job matching system that takes a CV and returns the most relevant job postings with explanations and skill gap analysis. Uses semantic search (FAISS + embeddings) compared against a BM25 keyword baseline to demonstrate the value of semantic retrieval.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your environment file and add your API key
cp .env.example .env

# 3. Download shared data files from Google Drive (link in project docs):
#    - data/vector_store/faiss_minilm.index
#    - data/vector_store/docstore_minilm.json
#    - data/kaggle_cleaned/postings_cleaned.csv
```

---

## Project Structure

```
project/
├── src/
│   ├── data_pipeline/       # One-time data build scripts, not part of runtime pipeline
│   │   ├── parse_kaggle.py
│   │   ├── fetch_arbeitnow.py
│   │   ├── build_vector_store_minilm.py
│   │   └── schemas.py
│   ├── workflow/            # Core pipeline components
│   │   ├── models.py        # Shared data schemas (JobRecord, CVProfile, etc.)
│   │   ├── mocks.py         # Stable test data for development
│   │   ├── cv_reader.py     # PDF to raw text via vision LLM
│   │   ├── cv_profiler.py   # Raw text to structured CVProfile
│   │   ├── job_search.py    # Semantic retrieval via FAISS, returns top 20
│   │   ├── reranker.py      # LLM reranking, returns top 10
│   │   └── reasoning.py     # Match explanations and skill gap analysis
│   ├── evaluation/
│   │   ├── baseline_bm25.py # BM25 keyword baseline
│   │   ├── run_evaluation.py
│   │   └── error_analysis.py
│   ├── prompts/             # LLM prompt templates
│   │   ├── cv_profiler.md
│   │   ├── reranker.md
│   │   └── reasoning.md
│   └── main.py              # End-to-end pipeline
├── tests/
│   ├── data_pipeline/
│   │   └── test_retrieval.py
│   └── workflow/
│       ├── test_mocks.py
│       ├── test_bm25.py
│       ├── test_cv_reader.py
│       ├── test_cv_profiler.py
│       └── test_job_search.py
├── data/                    # See data structure section below
├── iterations/              # Development notes and iteration logs
└── requirements.txt
```

---

## Running the Tests

Run these to verify your setup before developing.

### 1. Mock data validation (no external files needed)
```bash
cd project
python -m pytest tests/workflow/test_mocks.py -v
```
Expected: **19/19 passed**

### 2. BM25 baseline (requires Kaggle cleaned CSV)
```bash
cd project
python -m tests.workflow.test_bm25
```
Expected: **8/8 tests passed** — loads 124k jobs and runs keyword retrieval (~30s).

If you see a `FileNotFoundError`, download `postings_cleaned.csv` from Google Drive and place it at `data/kaggle_cleaned/postings_cleaned.csv`.

### 3. FAISS retrieval (requires vector store files)
```bash
cd project
python -m pytest tests/workflow/test_job_search.py -v
```
Expected: **16/16 passed** — loads 471k vectors and runs semantic search (~21s).

If you see a `FileNotFoundError`, download the vector store files from Google Drive and place them at `data/vector_store/`.

### 4. Reranker (contract tests — no external files needed)
```bash
cd project
python -m pytest tests/workflow/test_reranker.py -v
```
Expected: **30/30 passed** — all mocked, instant.

### 5. Reranker integration test (requires `GOOGLE_API_KEY` in `.env`)
```bash
cd project
pytest -m integration tests/workflow/test_reranker_integration.py -v
```
Expected: **10/10 passed** — runs real Gemma 4 LLM on mock personas and checks behavioral invariants (domain cap, seniority ordering, top job selection). **First run makes a real API call (~3,800 tokens). All subsequent runs are instant from cache.**

Invariants checked:
- Finance/HR jobs score ≤ 20 for a tech candidate (domain cap enforced)
- Best tech job (Stripe Python/AWS) appears in top 3
- Mid-level AWS job outscores junior frontend job (seniority logic)
- All scores in [0, 100], no duplicates, no unknown job_ids

If 10/10 pass → reranker is working correctly on real data. Proceed to manual labeling.

### 6. CV Pipeline Testing (cv_reader + cv_profiler)
```bash
cd project
python -m tests.workflow.test_cv_profiler                    # scan all PDFs in data/resumes/
python -m tests.workflow.test_cv_profiler data/resumes/cv.pdf  # test single file
```

**Supported Formats:** PDF only. DOCX and other formats are not supported — convert to PDF first (e.g., LibreOffice).

**Results Location:**
- **Raw extracted text:** `iterations/cv_extraction_results.md` (Step 1: PDF → raw text via vision LLM)
- **Structured profile JSON:** `iterations/cv_profiler_results.md` (Step 2: raw text → CVProfile)

Check both files to compare extraction quality and profiling accuracy. Share these files with teammates for validation.

**Prompts Location:**
LLM prompts are decoupled from code for easier iteration:
- Vision extraction prompt: `src/prompts/cv_reader.md`
- Text profiling system prompt: `src/prompts/cv_profiler.md`

Edit these `.md` files directly to tweak extraction/profiling behavior without touching Python.

**Caching & Cache Clearing:**
Both `cv_reader` and `cv_profiler` use disk-based caching (content-keyed) to avoid redundant API calls:
- Same PDF file (even if renamed) → cache hit → zero vision API cost
- Same extracted text from different PDFs → cache hit → zero Gemma API cost

**After code fixes, clear the cache before re-testing:**
```bash
rm -rf .cache/cv_reader .cache/cv_profiler
python -m tests.workflow.test_cv_profiler data/resumes/cv.pdf  # fresh API calls
```

Alternatively, bump the version numbers to trigger cache invalidation automatically:
- `cv_reader.py`: change `PROMPT_VERSION = "v2"` (also invalidates cache if you modify `src/prompts/cv_reader.md`)
- `cv_profiler.py`: change `LOGIC_VERSION = "v5"` (also invalidates cache if you modify `src/prompts/cv_profiler.md`)

---

## Developing with Mock Data

The mock data in `src/workflow/mocks.py` provides stable test fixtures so each pipeline component can be developed and tested independently without needing the full stack running.

```python
from src.workflow.mocks import (
    mock_cv_mid_tech,           # CVProfile: 5yrs Python/React, San Francisco
    mock_cv_senior_finance,     # CVProfile: 12yrs SAP/CPA, New York
    mock_cv_junior_hr,          # CVProfile: 2yrs recruiting/SHRM-CP, Chicago
    mock_preferences_mid_tech,  # JobSearchPreferences: remote, full-time
    mock_job_records,           # list[JobRecord]: 10 realistic Kaggle samples
)
```

**Mock job records include edge cases:**
- `k004` — null salary
- `k005` — fully remote role
- `k006` — part-time work type
- `k009` — geographic mismatch (Seattle job for SF candidate)

See `iterations/mocks.md` for full mock data specification and usage examples.

---

## BM25 Baseline

Keyword-based retrieval using BM25 — the evaluation baseline that the semantic pipeline is measured against.

```python
from src.evaluation.baseline_bm25 import BM25Retriever
from src.workflow.models import CVProfile, JobSearchPreferences

retriever = BM25Retriever()
results = retriever.search(cv_profile, preferences, k=20, source="kaggle")
# Returns list[JobRecord] sorted by BM25 score
```

**Design notes:**
- Always use `source="kaggle"` — evaluation is Kaggle-only per project scope
- Location is not included as a BM25 query token (Kaggle is ~99% US jobs; location adds noise)
- Seniority hard filter applied post-retrieval (entry CVs exclude Director/VP; senior CVs exclude intern/entry)

---

## Loading the FAISS Index

The index is built with raw FAISS (not LangChain). Handled by `src/workflow/job_search.py`.

- `data/vector_store/faiss_minilm.index` — FAISS `IndexFlatIP`, L2-normalized vectors (cosine similarity)
- `data/vector_store/docstore_minilm.json` — parallel list of `{page_content, metadata}` dicts

**Model:** `all-MiniLM-L6-v2`, 384 dimensions, 256-token hard limit.
**Index stats:** 471,671 vectors from 124,806 jobs (v2 build, paragraph-based chunking).

**Important:** The docstore index order matches the FAISS index exactly — `job_texts[i]` and `job_metadata[i]` correspond to vector `i`. Verified by `assert len(job_texts) == index.ntotal` at load time.

---

## Data Schemas

### `src/data_pipeline/schemas.py` — `JobDocument`
Internal pipeline schema. Used by data build scripts only.

### `src/workflow/models.py` — `JobRecord`, `CVProfile`, `JobSearchPreferences`
Workflow-level schemas used by all pipeline components. `JobRecord` is `JobDocument` + `score`.

```python
from src.workflow.models import JobRecord, CVProfile, JobSearchPreferences
```

**`CVProfile`** — factual data extracted from the CV (who the person is):
skills, experience_level, years_experience, current_location, education_level, certifications, job_titles_held, industries, domain_keywords, tools

**`JobSearchPreferences`** — user-provided intent (what they want):
target_location, work_type, willing_to_relocate, target_roles, remote_preference

---

## Data Directory

```
data/
├── kaggle_raw/          ← Download from Kaggle (gitignored, ~500MB)
├── kaggle_cleaned/      ← parse_kaggle.py output (shared via Google Drive)
│   ├── postings_cleaned.csv
│   └── data_quality_report.txt
├── arbeitnow/
│   └── arbeitnow_jobs.json   ← frozen evaluation snapshot (957 jobs)
├── vector_store/        ← shared via Google Drive (gitignored)
│   ├── faiss_minilm.index      ← all-MiniLM-L6-v2 index
│   └── docstore_minilm.json    ← parallel docstore
└── resumes/             ← test CV files (PDF or txt)
```

---

## Data Pipeline (Phase 1 — Already Complete)

The Kaggle parsing and Arbeitnow fetch were run once and their outputs are frozen. The vector store was rebuilt with corrected token handling (v2) and is shared via Google Drive.

| Script | What it does | Output |
|--------|-------------|--------|
| `parse_kaggle.py` | Cleans Kaggle CSV, joins skills + industries | `postings_cleaned.csv` |
| `fetch_arbeitnow.py` | Fetches Arbeitnow API snapshot | `arbeitnow_jobs.json` |
| `build_vector_store_minilm.py` | Embeds all jobs, builds FAISS index | `faiss_minilm.index` + `docstore_minilm.json` |

For full pipeline details and iteration notes, see `iterations/` folder.
