# Job Hunting Agent — Phase 1: Data Pipeline

## Overview

Phase 1 builds a FAISS vector index from two job posting sources (Kaggle + Arbeitnow) to enable semantic retrieval in downstream phases. All scripts read from and write to `data/`.

---

## Setup: Download Kaggle Raw Data

Before running the pipeline, you **must** download the raw Kaggle LinkedIn Job Postings dataset:

1. Go to [Kaggle LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
2. Download all files (postings.csv, job_skills.csv, job_industries.csv, etc.)
3. Place them in `data/kaggle_raw/`

The structure should look like:
```
data/kaggle_raw/
├── postings.csv
├── job_skills.csv
├── job_industries.csv
└── ... (other files from Kaggle)
```

**Note:** `data/kaggle_raw/` is gitignored. Each team member must download it independently.

---

## Phase 1 Pipeline

### Step 1: Parse Kaggle Dataset
**File:** `src/data_pipeline/parse_kaggle.py`  
**What it does:** 
- Reads raw files from `data/kaggle_raw/` (postings.csv, job_skills.csv, job_industries.csv)
- Joins postings with job_skills and job_industries tables
- Cleans descriptions (removes HTML, deduplicates, filters by min length)
- Outputs normalized JobDocument schema

**Prerequisites:** You must download and place raw Kaggle files in `data/kaggle_raw/` (see Setup section above)

**Run:**
```bash
cd project
python src/data_pipeline/parse_kaggle.py
```

**Output:**
- `data/kaggle_cleaned/postings_cleaned.csv` — cleaned postings (columns: job_id, title, company_name, description, location, formatted_experience_level, formatted_work_type, min_salary, max_salary, application_url, skill_labels, industries, source) [Shared via Google Drive]
- `data/kaggle_cleaned/data_quality_report.txt` — summary of rows dropped/kept

**Note:** This output is gitignored locally but will be rebuilt from raw data.

---

### Step 2: Fetch Arbeitnow API
**File:** `src/data_pipeline/fetch_arbeitnow.py`  
**What it does:**
- Paginates through Arbeitnow API endpoints
- Cleans HTML in descriptions
- Maps to JobDocument schema (sets salary/experience_level/skills to null if not available)
- Deduplicates by job slug

**Run:**
```bash
cd project
python src/data_pipeline/fetch_arbeitnow.py
```

**Output:**
- `data/arbeitnow/arbeitnow_jobs.json` — JSON list of JobDocument dicts (one per line for streaming)

**⚠️ Note:** `data/arbeitnow/arbeitnow_jobs.json` is the evaluation snapshot. Do not rebuild it. Arbeitnow API data changes over time; rebuilding would produce different results and break evaluation consistency across the team.

---

### Step 3: Download FAISS Vector Index

**⚠️ DO NOT BUILD THIS YOURSELF** — the vector store is shared via Google Drive.

**What:** The vector store snapshot (faiss.index + docstore.json, ~500 MB) was built during Phase 1 evaluation and is shared via Google Drive to ensure all team members work with the exact same embeddings.

**To get the files:**
1. Download the vector store from [Google Drive link]
2. Extract to `data/vector_store/`

**Why you shouldn't rebuild:**
- The vector store snapshot is the evaluation baseline
- Rebuilding produces different embeddings (Arbeitnow data changes over time)
- Different embeddings = different retrieval results = inconsistent team testing

**For reference — build script (DO NOT RUN):**
- `src/data_pipeline/build_vector_store.py` — only for documentation; already built and frozen

**Output:**
- `data/vector_store/faiss.index` — FAISS flat index (normalized for cosine distance)
- `data/vector_store/docstore.json` — parallel list of chunk metadata + page_content

---

### Step 4 (Optional): Test Retrieval
**File:** `src/data_pipeline/test_retrieval.py`  
**What it does:** Loads the FAISS index and runs basic retrieval smoke tests

**Run:**
```bash
cd project
python src/data_pipeline/test_retrieval.py
```

---

## Shared Schema

**File:** `project/src/data_pipeline/schemas.py`  
Defines `JobDocument` with fields:
- job_id, title, company_name, description, location
- formatted_experience_level, formatted_work_type, min_salary, max_salary
- application_url, skill_labels, industries, source

All files map input data to this schema before downstream processing.

---

## Data Directory Structure

```
data/
├── kaggle_raw/                       ← YOU DOWNLOAD from Kaggle (gitignored)
│   ├── postings.csv
│   ├── job_skills.csv
│   ├── job_industries.csv
│   └── ... (other Kaggle files)
├── kaggle_cleaned/                   ← OUTPUT from parse_kaggle.py (gitignored, rebuilds locally)
│   ├── postings_cleaned.csv
│   └── data_quality_report.txt
├── arbeitnow/
│   └── arbeitnow_jobs.json           ← evaluation snapshot (evaluation baseline)
└── vector_store/                     ← SHARED VIA GOOGLE DRIVE (~500 MB, too large for git)
    ├── faiss.index
    └── docstore.json
```
