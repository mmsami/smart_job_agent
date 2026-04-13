# DEV LOG: Data Cleaning & Fetching

## [Phase 1] Initial Setup & Kaggle
Sami already cleaned the Kaggle data (123k rows). Looks solid. 
Problem: we wanted more current data, so decided to pull from Arbeitnow API.

## [Iteration 1] Arbeitnow API Exploration
Did some test calls to `https://www.arbeitnow.com/api/job-board-api?page=1`.
- Found: `slug` (ID), `company_name`, `title`, `description` (it's all HTML, need to strip it), `tags` (array), `remote` (bool).
- Missing: No experience level or salary in the API. 

Headache: How to merge this with Kaggle? Kaggle has tons of fields Arbeitnow doesn't.
Decision: Create a shared schema. If a field is missing (like salary for Arbeitnow), just set it to `None`. Better to be honest about missing data than guess.

## [Iteration 2] Folder Mess
Folders were all over the place. Everything was just in `data/cleaned/`.
- `postings_cleaned.csv`
- `arbeitnow_jobs.json`
- some raw files mixed in.

Reorganized it to be cleaner:
- `data/kaggle_raw/` -> raw stuff
- `data/kaggle_cleaned/` -> processed CSV
- `data/arbeitnow/` -> API snapshot
- `data/vector_store/` -> FAISS index

Now it's actually clear what is input and what is output.

## [Iteration 3] Schema & Mapping
Tried mapping manually first, but it was a pain. 
Solution: Use Pydantic (`schemas.py`).
- `JobDocument` class handles the mapping for both sources.
- `tags` from Arbeitnow -> joined into a string for `skill_labels`.
- `remote` bool -> if True, then "Remote", else use `job_types[0]`.
- Added `source` field ("kaggle" vs "arbeitnow") so we can filter later.

Ran the fetcher: got 957 jobs from Arbeitnow. Not a huge amount, but good for EU/Germany coverage.

## [Iteration 4] Vector Store Build
Updated `build_vector_store.py` to pull from both sources.
- Loaded Kaggle CSV + Arbeitnow JSON.
- Paragraph chunking: split on `\n\n`. If a para is too long (>512 tokens), use a sliding window with 50 token overlap.
- Signal boosting: prepending `{title} at {company}. {skills}.` to every chunk. This is key so the embedding doesn't lose the context of the job title in long descriptions.
- FAISS setup: `IndexFlatIP` + `normalize_L2` for cosine similarity.

## [Final Checks]
- Arbeitnow: 957 jobs. Most fields okay, salary/exp are null.
- Kaggle: 123k jobs. mostly complete.
- Comparison: Both sources are in the same index now. lauest results are consistent.

## [Discussion] Current vs Historical Data
Kyoungmi brought up that Kaggle is 2023-2024 data. Should we only use Arbeitnow for the user?
- Discussion: If we only use Arbeitnow (957 jobs), the project is too "small". We need the 124k for the FAISS demo (scaling).
- Decision: Keep both in the index. Use all for evaluation (shows scale), but add a `source` filter for the demo (shows current jobs only). Best of both worlds.

## [Log] Data Sharing / Git
Kaggle data is too big for Git (~500MB).
Setup:
1. `data/kaggle_cleaned_sample/` -> 1000 rows in Git so prof can test.
2. Full data -> gitignored. build locally.
3. Vector store -> shared via Google Drive (saves everyone from running the embedding script).
