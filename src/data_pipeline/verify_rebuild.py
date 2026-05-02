"""
verify_rebuild.py — Sanity check after rebuilding the FAISS index.

Checks:
  1. Descriptions lookup exists and is non-empty
  2. A sample search returns jobs with non-empty descriptions
  3. No duplicate title+company pairs in the top-20 results
  4. Spot-checks a known problematic job (GIG USA entry-level flood)

Usage:
    python -m src.data_pipeline.verify_rebuild
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
DESCRIPTIONS_PATH = VECTOR_DIR / "job_descriptions_minilm.json"
DOCSTORE_PATH = VECTOR_DIR / "docstore_minilm.json"

PASS = "✓"
FAIL = "✗"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def main():
    print("=" * 55)
    print("Index Rebuild Verification")
    print("=" * 55)
    results = []

    # 1. Descriptions file exists
    exists = DESCRIPTIONS_PATH.exists()
    results.append(check("job_descriptions_minilm.json exists", exists))
    if not exists:
        print("\nRebuild the index first: python -m src.data_pipeline.build_vector_store_minilm")
        sys.exit(1)

    with open(DESCRIPTIONS_PATH, encoding="utf-8") as f:
        descriptions: dict = json.load(f)

    results.append(check(
        "Descriptions lookup non-empty",
        len(descriptions) > 0,
        f"{len(descriptions):,} entries"
    ))

    empty_descs = sum(1 for v in descriptions.values() if not v.strip())
    results.append(check(
        "No empty descriptions in lookup",
        empty_descs == 0,
        f"{empty_descs} empty" if empty_descs else "all good"
    ))

    # 2. Docstore exists and has matching count
    if DOCSTORE_PATH.exists():
        with open(DOCSTORE_PATH, encoding="utf-8") as f:
            docstore = json.load(f)
        results.append(check(
            "Docstore loaded",
            len(docstore) > 0,
            f"{len(docstore):,} chunks"
        ))
    else:
        results.append(check("Docstore exists", False))

    # 3. Run a live search and check descriptions populate
    print("\n  Running live FAISS search...")
    try:
        from src.workflow.job_search import retrieve_jobs
        from src.workflow.models import CVProfile, JobSearchPreferences

        test_cv = CVProfile(
            skills=["Digital Marketing", "Brand Positioning"],
            experience_level="entry",
            years_experience=1,
            current_location=None,
            education_level="master",
            field_of_study="Strategic Marketing",
            industries=[],
            domain_keywords=[],
            tools=[],
            languages=[],
            certifications=[],
            job_titles_held=["Marketing Manager"],
        )
        test_prefs = JobSearchPreferences(
            target_location="United States",
            work_type="full-time",
            employment_type="full-time",
            willing_to_relocate=True,
            remote_preference="flexible",
            target_roles=[],
            industry_preference=[],
        )
        jobs = retrieve_jobs(test_cv, test_prefs, top_k=20)

        results.append(check("retrieve_jobs returned results", len(jobs) > 0, f"{len(jobs)} jobs"))

        empty_in_results = sum(1 for j in jobs if not j.description.strip())
        results.append(check(
            "All returned jobs have descriptions",
            empty_in_results == 0,
            f"{empty_in_results} empty" if empty_in_results else "all populated"
        ))

        # 4. Duplicate check
        seen: set = set()
        dupes = 0
        for j in jobs:
            key = (j.title.lower().strip(), j.company.lower().strip())
            if key in seen:
                dupes += 1
            seen.add(key)
        results.append(check(
            "No duplicate title+company in top-20",
            dupes == 0,
            f"{dupes} dupes found" if dupes else "all unique"
        ))

        # 5. GIG USA flood check — should appear at most once
        gig_usa_count = sum(
            1 for j in jobs
            if "entry level openings" in j.title.lower() and "gig usa" in j.company.lower()
        )
        results.append(check(
            "GIG USA entry-level flood resolved",
            gig_usa_count <= 1,
            f"found {gig_usa_count} copies"
        ))

        # Print sample result
        if jobs:
            sample = jobs[0]
            print(f"\n  Sample result: {sample.title} @ {sample.company}")
            preview = sample.description[:120].replace("\n", " ")
            print(f"  Description:   {preview}{'...' if len(sample.description) > 120 else ''}")

    except Exception as e:
        results.append(check("Live search", False, str(e)))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 55}")
    print(f"{'All checks passed' if passed == total else 'Some checks failed'}: {passed}/{total}")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
