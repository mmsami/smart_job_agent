# Dev Log: Data Cleaning & Ingestion
**Date:** 2026-04-12 to 2026-04-15
**Objective:** Create a unified, high-quality dataset by merging a massive static Kaggle dataset with a current API snapshot from Arbeitnow.

---

## 1. The Data Sources
We are dealing with two fundamentally different data shapes:
1. **Kaggle LinkedIn Postings:** ~124k jobs. Static, high volume, US-centric. Rich supplementary tables for skills and industries.
2. **Arbeitnow API:** ~950 jobs. Dynamic, EU/Germany centric. Simple JSON structure, lacking experience levels and salaries.

---

## 2. Kaggle Pipeline (The Heavy Lifting)
The Kaggle dataset was too raw for direct indexing. We implemented a multi-stage cleaning pipeline in `parse_kaggle.py`.

### The Join Strategy
To add structure to the raw descriptions, we performed several joins using the `job_id` as the key:
- **Skill Labels:** Joined `job_skills.csv` → `skills.csv`. This transformed raw IDs into 35 broad professional categories (e.g., "Information Technology", "Sales"). **Coverage: 98.6%**.
- **Industry Classification:** Joined `job_industries.csv` → `industries.csv`. This added high-level industry labels. **Coverage: 98.8%**.

### Cleaning & Validation
- **HTML Stripping:** Used regex to remove all HTML tags from descriptions.
- **Junk Filtering:** Defined a "junk" row as one where the title is $<3$ characters AND the description is $<50$ characters. **Result: 0 rows dropped** (Kaggle's data was surprisingly clean).
- **Experience Validation:** Verified all `formatted_experience_level` values against our target schema to ensure no unexpected labels would break the filter later.

### Data Quality Report (Summary)
- **Total Rows:** 123,849
- **Full Quality Rows:** 122,040 (98.5%) — have good title, description, and skill labels.
- **Salary Sparsity:** 75.9% of rows have `null` salaries.
- **Experience Sparsity:** 23.7% of rows have `null` experience levels.
- **Median Description Length:** 3,419 characters.

---

## 3. Arbeitnow Integration (The "Current" Signal)
To ensure the agent is useful for users in Germany, we implemented `fetch_arbeitnow.py`.

### API Mapping Logic
Since Arbeitnow's JSON doesn't match Kaggle's CSV, we implemented a mapping layer in `schemas.py`:
| Arbeitnow Field | Shared Schema Field | Logic |
|-----------------|-------------------|-------|
| `slug` | `job_id` | Unique identifier |
| `company_name` | `company` | Direct map |
| `title` | `title` | Direct map |
| `description` | `description` | Strip HTML tags |
| `remote` (bool) | `work_type` | If True → "Remote", else use `job_types[0]` |
| `tags` (list) | `skill_labels` | Join list into a comma-separated string |
| `location` | `location` | Direct map |
| (missing) | `salary` / `exp` | Set to `null` |

**Final Count:** 957 jobs fetched and validated via Pydantic.

---

## 4. Architectural Decisions

### The "Current vs. Historical" Conflict
The team debated whether to discard the Kaggle data because it is historical. 
**Decision:** Keep both.
- **Why?** Using only Arbeitnow (957 jobs) makes the project a "toy" application. Using 125k jobs proves the system can **scale**.
- **The Solution:** Maintain a `source` metadata field. Use the full index for evaluation (proving scale), but allow the user to toggle a "Current Jobs Only" filter in the demo (filtering for `source == 'arbeitnow'`).

### Git & Portability Strategy
The full cleaned dataset is ~500MB, which is too large for GitHub.
1. **`data/kaggle_cleaned_sample/`**: Committed 1,000 rows to Git. This allows the professor to run the code immediately without downloading GBs of data.
2. **Full Index**: Shared the built FAISS index via Google Drive. This prevents teammates from having to run the 69-minute embedding script locally.

---

## 5. Final State
- **Unified Dataset:** 124,806 total jobs (Kaggle-only used in pipeline runs per professor feedback).
- **Shared Schema:** All jobs now follow the `JobDocument` Pydantic model.
- **Ready for Indexing:** Data is cleaned, HTML-free, and augmented with skill/industry labels.

---

## 6. Dataset Constraint: Partial Employment Preferences

**Professor feedback:** "Including criteria like part-time or partial employment preferences would also be an interesting dimension to consider."

**What we wanted to implement:** A fine-grained `employment_percentage` field (e.g., "20 hours/week", "3 days/week") in `JobSearchPreferences` to allow users to express partial work preferences precisely.

**Why we couldn't:** The Kaggle dataset's `formatted_work_type` field only contains broad categories (`Full-time`, `Part-time`, `Contract`, `Temporary`, `Internship`). There is no hours-per-week or days-per-week field on any job posting. Adding a preference the corpus cannot match against produces a dead filter — it would never exclude or prioritize any result.

**What we did instead:** Added `employment_type: "full-time" | "part-time" | "contract" | "any"` to `JobSearchPreferences`, which maps directly to the available `formatted_work_type` values in the dataset.

**For the report:** This is a documented limitation of the dataset, not the system. A richer job corpus (e.g., one that includes weekly hours) would enable finer-grained employment matching. Mention in the Known Limitations section.
