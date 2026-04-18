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
#    - data/vector_store/faiss.index
#    - data/vector_store/docstore.json
#    - data/kaggle_cleaned/postings_cleaned.csv
```

---

## Project Structure

```
project/
├── src/
│   ├── data_pipeline/       # One-time data build scripts (Phase 1, already run)
│   │   ├── parse_kaggle.py
│   │   ├── fetch_arbeitnow.py
│   │   ├── build_vector_store.py
│   │   └── schemas.py
│   ├── workflow/            # Core pipeline components
│   │   ├── models.py        # Shared data schemas (JobRecord, CVProfile, etc.)
│   │   ├── mocks.py         # Stable test data for development
│   │   ├── cv_reader.py     # CV text extraction (PDF/txt) using PyMuPDF
│   │   ├── cv_profiler.py   # CV text → structured profile (LLM)
│   │   ├── job_search.py    # Semantic retrieval (FAISS) → top 20 jobs
│   │   ├── reranker.py      # LLM reranking → top 10 jobs
│   │   └── reasoning.py     # Match explanations + skill gap analysis (LLM)
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
│       └── test_bm25.py
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

### 3. FAISS index sanity check (requires vector store files)
```bash
cd project
python tests/data_pipeline/test_retrieval.py
```
Expected: Prints top-5 results per query. **Manual audit** — verify that results are in the correct domain (e.g., Python query → Python jobs, accounting query → finance jobs). No automated assertions by design.

If you see a `FileNotFoundError`, download the vector store from Google Drive and place files at `data/vector_store/`.

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

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

results = vectorstore.similarity_search_with_score("python developer aws", k=20)
```

**Index stats:** 192,514 vectors (124,806 jobs chunked), 384 dimensions, `all-MiniLM-L6-v2` embeddings.

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
├── vector_store/        ← shared via Google Drive (~844MB, gitignored)
│   ├── faiss.index
│   └── docstore.json
└── resumes/             ← test CV files (PDF or txt)
```

---

## Data Pipeline (Phase 1 — Already Complete)

The data pipeline was run once to build the shared index. The outputs are frozen and shared via Google Drive for evaluation consistency. **Do not re-run these scripts** — rebuilding would produce different embeddings and break cross-run comparisons.

| Script | What it does | Output |
|--------|-------------|--------|
| `parse_kaggle.py` | Cleans Kaggle CSV, joins skills + industries | `postings_cleaned.csv` |
| `fetch_arbeitnow.py` | Fetches Arbeitnow API snapshot | `arbeitnow_jobs.json` |
| `build_vector_store.py` | Embeds all jobs, builds FAISS index | `faiss.index` + `docstore.json` |

For full pipeline details and iteration notes, see `iterations/` folder.
