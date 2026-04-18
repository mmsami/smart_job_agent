"""
Test suite for BM25 baseline retrieval.

Tests:
    1. Basic smoke test — returns 20 results with scores
    2. Keyword trap — Java CV should not return JavaScript jobs in top 20
    3. Seniority gap — entry CV should not contain Director/VP roles
    4. Null metadata — jobs with missing experience_level still appear

Usage:
    cd project && python -m tests.workflow.test_bm25
    or
    python tests/workflow/test_bm25.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.workflow.models import CVProfile, JobSearchPreferences
from src.evaluation.baseline_bm25 import BM25Retriever

# ── Shared retriever (built once, reused across tests) ─────────────────
print("Loading BM25 index (shared across all tests)...")
retriever = BM25Retriever()
print()

# ── Shared preferences ─────────────────────────────────────────────────
default_prefs = JobSearchPreferences(
    target_location="United States",
    work_type="full-time",
    employment_type="full-time",
    willing_to_relocate=False,
    target_roles=[],
    industry_preference=[],
    remote_preference="flexible",
)

PASS = "PASS"
FAIL = "FAIL"
results_summary = []


def run_test(name, passed, detail=""):
    status = PASS if passed else FAIL
    results_summary.append((name, status, detail))
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# ══════════════════════════════════════════════════════════════════════
# Test 1 — Basic smoke test
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Test 1: Basic smoke test (entry Python dev, source=kaggle)")
print("=" * 70)

cv_entry_python = CVProfile(
    skills=["Python", "Django", "PostgreSQL", "React", "Git"],
    experience_level="entry",
    years_experience=1,
    current_location="San Francisco, CA",
    education_level="bachelor",
    field_of_study="Computer Science",
    certifications=[],
    languages=["English"],
    job_titles_held=["Junior Developer", "Intern"],
    industries=["Software", "Technology"],
    domain_keywords=["REST API", "agile", "CI/CD"],
    tools=["Docker", "GitHub"],
)

results = retriever.search(cv_entry_python, default_prefs, k=20, source="kaggle")

print(f"\nTop 20 results:\n")
for i, job in enumerate(results, 1):
    print(f"  {i:2d}. [{job.score:.2f}] {job.title} | {job.company} | exp={job.experience_level or 'null'}")

run_test("Returns exactly 20 results", len(results) == 20, f"got {len(results)}")
run_test("All results from kaggle", all(j.source == "kaggle" for j in results))
run_test("All results have scores > 0", all(j.score > 0 for j in results))
run_test("Results sorted descending by score", all(results[i].score >= results[i+1].score for i in range(len(results)-1)))


# ══════════════════════════════════════════════════════════════════════
# Test 2 — Keyword trap: Java CV should not return JavaScript jobs
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Test 2: Keyword trap (Java CV → no JavaScript-only jobs in top 20)")
print("=" * 70)

cv_java = CVProfile(
    skills=["Java", "Spring Boot", "Maven", "JUnit", "Hibernate"],
    experience_level="mid",
    years_experience=4,
    current_location="New York, NY",
    education_level="bachelor",
    field_of_study="Computer Science",
    certifications=[],
    languages=["English"],
    job_titles_held=["Java Developer", "Software Engineer"],
    industries=["Software", "Finance"],
    domain_keywords=["microservices", "REST API", "SQL"],
    tools=["IntelliJ", "Git", "Jenkins"],
)

java_results = retriever.search(cv_java, default_prefs, k=20, source="kaggle")

# JavaScript-only jobs: title contains "javascript" or "node.js" but NOT "java" alone
js_only_titles = [
    j.title for j in java_results
    if "javascript" in j.title.lower() or "node.js" in j.title.lower()
    if "java" not in j.title.lower().replace("javascript", "")
]

print(f"\nTop 10 titles returned for Java CV:")
for i, job in enumerate(java_results[:10], 1):
    print(f"  {i:2d}. {job.title}")

if js_only_titles:
    print(f"\n  JavaScript-only hits in top 20: {js_only_titles}")
else:
    print(f"\n  No JavaScript-only hits in top 20.")

run_test("No JavaScript-only jobs in top 20 for Java CV", len(js_only_titles) == 0,
         f"found: {js_only_titles}" if js_only_titles else "")


# ══════════════════════════════════════════════════════════════════════
# Test 3 — Seniority gap: entry CV should not return Director/VP roles
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Test 3: Seniority gap (entry CV → no Director/VP roles in top 20)")
print("=" * 70)

# Reuse cv_entry_python from Test 1
seniority_results = retriever.search(cv_entry_python, default_prefs, k=20, source="kaggle")

EXEC_KEYWORDS = {"director", "vp ", "vice president", "chief ", "head of", "c-level", "partner"}
exec_hits = [
    j.title for j in seniority_results
    if any(kw in j.title.lower() for kw in EXEC_KEYWORDS)
]

print(f"\nExperience levels in top 20:")
from collections import Counter
exp_counts = Counter(j.experience_level or "null" for j in seniority_results)
for level, count in exp_counts.most_common():
    print(f"  {level}: {count}")

if exec_hits:
    print(f"\n  Executive-level hits: {exec_hits}")
else:
    print(f"\n  No Director/VP/executive titles found.")

run_test("No Director/VP roles in top 20 for entry CV", len(exec_hits) == 0,
         f"found: {exec_hits}" if exec_hits else "")


# ══════════════════════════════════════════════════════════════════════
# Test 4 — Null metadata: jobs with missing experience_level still appear
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Test 4: Null metadata (jobs with missing experience_level still returned)")
print("=" * 70)

# Run a broad search — null-metadata jobs should not be excluded
null_results = retriever.search(cv_entry_python, default_prefs, k=20, source="kaggle")

null_exp_jobs = [j for j in null_results if j.experience_level is None]
print(f"\n  Jobs with null experience_level in top 20: {len(null_exp_jobs)}")
for j in null_exp_jobs[:3]:
    print(f"    - {j.title} | {j.company}")

# Pass if at least some null-metadata jobs appear (they exist in dataset)
# If 0, it could mean the filter wrongly excluded them OR the top results just happen to all have metadata
# We check the broader index has null-exp jobs, then verify they can appear
null_in_index = sum(1 for j in retriever.jobs if j.get("experience_level") is None and j["source"] == "kaggle")
print(f"  Null experience_level jobs in full Kaggle index: {null_in_index:,}")

run_test(
    "Null experience_level jobs exist in index (filter not excluding them at load)",
    null_in_index > 0,
    f"{null_in_index:,} jobs with null exp level in index"
)
run_test(
    "Null experience_level jobs can appear in results",
    len(null_exp_jobs) > 0,
    f"{len(null_exp_jobs)} found in top 20" if null_exp_jobs else "0 found — check seniority filter skip logic"
)


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
passed = sum(1 for _, s, _ in results_summary if s == PASS)
total = len(results_summary)
for name, status, detail in results_summary:
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
print()
print(f"  {passed}/{total} tests passed")
if passed < total:
    print("  SOME TESTS FAILED — review output above")
else:
    print("  ALL TESTS PASSED")
