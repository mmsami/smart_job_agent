# Dev Log: Mock Data for Contract Validation
**Date:** 2026-04-18
**Updated:** 2026-04-18
**Objective:** Provide stable, predictable "fake" data to allow parallel development of the pipeline steps without waiting for cv_parser.py or FAISS index completion.

---

## 1. The Rationale

In a complex pipeline, Step B depends on the output of Step A. If Step A is being developed or tweaked, Step B is blocked. **Mocks solve this.**

By defining a fixed set of `CVProfile` and `JobRecord` objects in `src/workflow/mocks.py`, we create a "contract" between team members:
- Searcher (job_search.py + reranker.py) can develop without waiting for CV Parser
- Analyzer (reasoning.py) can develop without waiting for Searcher
- All schemas guaranteed to match models.py exactly via Pydantic validation

---

## 2. Mock Data Specification

### CV Profiles (3 Personas)

| Persona | Level | Years | Location | Skills (Sample) | Certs |
|---------|-------|-------|----------|-----------------|-------|
| **Senior Finance** | senior | 12 | New York, NY | SAP, SQL, R, FP&A, accounting, tax | CPA, MBA |
| **Mid-level Tech** | mid | 5 | San Francisco, CA | Python, React, TypeScript, AWS, Docker | — |
| **Junior HR** | entry | 2 | Chicago, IL | recruiting, payroll, onboarding, compliance | SHRM-CP |

### Job Search Preferences (Matching Each Persona)

| Persona | Target Location | Remote Preference | Work Type |
|---------|-----------------|-------------------|-----------|
| Senior Finance | New York, NY | hybrid | full-time |
| Mid-level Tech | San Francisco, CA | remote | full-time |
| Junior HR | Chicago, IL | hybrid | full-time |

### Job Records (10 Realistic Kaggle Samples)

All 10 jobs are from Kaggle (`source="kaggle"`), with varied seniority, domains, and edge cases:

| ID | Title | Company | Level | Location | Salary | Edge Case |
|----|-------|---------|-------|----------|--------|-----------|
| k001 | Senior Accountant - FP&A | Goldman Sachs | senior | New York, NY | 140k–180k | — |
| k002 | Senior Backend Engineer - Python | Stripe | mid | San Francisco, CA | 160k–220k | — |
| k003 | HR Coordinator | TechCorp Inc | entry | Chicago, IL | 45k–55k | — |
| k004 | Finance Manager | Accenture | senior | New York, NY | **null** | Null salary |
| k005 | Full-Stack Engineer (Remote) | GitLab | mid | **Remote** | 150k–200k | Remote |
| k006 | HR Specialist - Part Time | Health Clinic | entry | Chicago, IL | 32k–40k | **Part-time** |
| k007 | Junior Frontend Developer | Startup XYZ | entry | San Francisco, CA | 80k–100k | Entry-level |
| k008 | Director of Finance | JPMorgan Chase | senior | New York, NY | 220k–300k | Executive |
| k009 | Backend Engineer - AWS | Amazon | mid | Seattle, WA | 170k–240k | Geographic mismatch |
| k010 | Senior Accounting Manager | Deloitte | senior | New York, NY | 130k–170k | Hybrid role |

**Edge Cases Tested:**
- ✅ Null salary (k004)
- ✅ Remote work (k005)
- ✅ Part-time employment (k006)
- ✅ Entry-level roles (k003, k006, k007)
- ✅ Geographic mismatch (k009 Seattle for CA candidate)
- ✅ Seniority mismatch (k008 Director for mid-level candidate)

---

## 3. Usage Guide for Teammates

### For the "Searcher" (Implements `job_search.py` + `reranker.py`)

```python
from src.workflow.mocks import (
    mock_cv_mid_tech,
    mock_preferences_mid_tech,
    mock_job_records
)

# Step 1: Develop with mocks
cv = mock_cv_mid_tech  # CVProfile: 5 yrs Python, San Francisco
prefs = mock_preferences_mid_tech  # JobSearchPreferences: remote, full-time
jobs = mock_job_records  # 10 sample jobs

# Step 2: Build retriever to return top-20 jobs
# retriever.search(cv, prefs) → List[JobRecord] (ranked by relevance)

# Step 3: Build reranker to score top-20 → top-10
# reranker.rerank(cv, prefs, top_20) → List[JobRecord] (re-ranked by LLM)
```

**Expected behavior with mocks:**
- Input: Mid-level tech CV (Python, React, AWS, 5 yrs, CA)
- Output: Top results should include k002 (Stripe), k005 (GitLab remote), k009 (AWS)
- Seniority filter: Skip k007 (junior developer), d008 (director)

### For the "Analyzer" (Implements `reasoning.py`)

```python
from src.workflow.mocks import mock_cv_mid_tech, mock_job_records

# Step 1: Develop with mocks (no searcher/reranker needed)
cv = mock_cv_mid_tech  # CVProfile
top_10_jobs = mock_job_records[:10]  # Simulate reranker output

# Step 2: Build reasoning engine to explain matches + skill gaps
# report = reasoning.analyze(cv, top_10_jobs)
# Output: Structured explanation per job + missing skills summary
```

**Expected behavior with mocks:**
- For k002 (Stripe, Python), CV has Python → high match
- For k001 (Goldman Sachs, accounting), CV lacks accounting → skill gap
- Report should identify top 3 missing skills across all 10 jobs

---

## 4. Validation & Testing

### Automated Tests (`tests/workflow/test_mocks.py`)

All mocks pass Pydantic validation. Run tests:
```bash
pytest tests/workflow/test_mocks.py -v
```

**Test coverage:**
- ✅ All CVProfile objects load with required fields
- ✅ All JobSearchPreferences load correctly
- ✅ All JobRecord objects validate against schema
- ✅ Edge cases are present (null salary, part-time, entry-level, remote)
- ✅ Searcher usage pattern works (retriever.search(cv, prefs))
- ✅ Analyzer usage pattern works (analyzer.analyze(cv, jobs[:10]))
- ✅ Jobs span multiple domains (Tech, Finance, HR)

### Validation Output
```
✓ CVProfile mocks: 3 personas (senior, mid, entry)
✓ JobSearchPreferences mocks: 3 preferences (NY, SF, Chicago)
✓ JobRecord mocks: 10 jobs with 5 edge cases
✓ Pydantic validation: PASS
```

---

## 5. Key Design Decisions

### Why 3 Personas?
- Covers seniority spectrum: Entry (HR), Mid (Tech), Senior (Finance)
- Tests domain diversity: HR, Tech, Finance (not just tech-biased)
- Aligns with final 10-persona test set (tech/finance/hr split)

### Why 10 Job Records?
- Large enough to test ranking/filtering logic
- Small enough for manual inspection and debugging
- Includes edge cases (null salary, part-time, remote, seniority mismatch)

### Why All Kaggle Source?
- Evaluation uses Kaggle-only (`source="kaggle"`)
- Mocks should match evaluation data distribution
- Simpler for teammates to reason about

### Import Pattern
```python
from src.workflow.mocks import (
    mock_cv_senior_finance,
    mock_cv_mid_tech,
    mock_cv_junior_hr,
    mock_preferences_senior_finance,
    mock_preferences_mid_tech,
    mock_preferences_junior_hr,
    mock_job_records
)
```

All objects are module-level variables (no factory functions). Teammates can import and use directly.

---

## 6. FAISS Index Setup

### Where to Get It

The FAISS index is built by Sami in Phase 1 and shared via Google Drive.

**Setup steps for teammates:**
1. Download `faiss.index` + `docstore.json` from: **[Google Drive link — Sami to provide]**
2. Place in: `data/vector_store/` (create folder if missing)
3. Load in code:
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/vector_store", embeddings, allow_dangerous_deserialization=True)
```

### Index Contents
- **Total jobs:** 124,806 (Kaggle ~123,849 + Arbeitnow ~957)
- **Evaluation mode:** Use `source="kaggle"` filter (returns Kaggle-only)
- **Embedding model:** all-MiniLM-L6-v2 (384 dimensions)
- **Metadata per job:** job_id, title, company, location, experience_level, work_type, skills, source, etc.

---

## 7. Sample LLM Response Formats

### For Reranker Output

Input: top 20 jobs + CV  
Output: top 10 jobs with new relevance scores

```json
{
  "reranked_jobs": [
    {
      "job_id": "k002",
      "title": "Senior Backend Engineer - Python",
      "company": "Stripe",
      "score": 92.5,
      "reasoning": "CV has Python + AWS + Docker. Job requires all three. Remote matches preference."
    },
    {
      "job_id": "k005",
      "title": "Full-Stack Engineer (Remote)",
      "company": "GitLab",
      "score": 88.3,
      "reasoning": "Strong match: React + Node.js. Fully remote aligns with preference."
    },
    // ... 8 more
  ]
}
```

**Note:** Scores are LLM-assigned (0-100 relevance), NOT cosine similarity.

### For Reasoning Output

Input: CV + top 10 jobs  
Output: explanations per job + aggregated missing skills

```json
{
  "cv_summary": "Mid-level Python developer, 5 years experience, SF-based, wants remote",
  "job_explanations": [
    {
      "job_id": "k002",
      "title": "Senior Backend Engineer - Python",
      "company": "Stripe",
      "match_reason": "You have Python + AWS + Docker experience. This job explicitly requires all three. Remote-flexible matches your preference.",
      "extracted_facts": {
        "key_responsibilities": ["Build payment infrastructure", "Design microservices", "Lead technical decisions"],
        "requirements": ["5+ years Python", "AWS experience", "PostgreSQL", "REST APIs"]
      },
      "missing_skills": ["Go", "GraphQL", "gRPC"]
    },
    {
      "job_id": "k005",
      "title": "Full-Stack Engineer (Remote)",
      "company": "GitLab",
      "match_reason": "Strong fit: React + Node.js expertise. Job is 100% remote with async culture.",
      "extracted_facts": {
        "key_responsibilities": ["Build collaboration features", "Maintain platform stability"],
        "requirements": ["4-6 years experience", "React", "Node.js", "PostgreSQL"]
      },
      "missing_skills": ["TypeScript (nice-to-have)", "Kubernetes"]
    },
    // ... 8 more
  ],
  "overall_missing_skills": [
    "Go (appears in 5 jobs)",
    "Kubernetes (appears in 4 jobs)",
    "GraphQL (appears in 3 jobs)"
  ],
  "recommendation": "Strong backend candidate. Learning Go would unlock 5+ additional matches. Remote-first companies are ideal fit."
}
```

**Note:** This is ONE LLM call for all 10 jobs (not 10 separate calls).
