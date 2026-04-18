"""
Fetch job postings from the Arbeitnow API and save as a JSON snapshot.

Paginates through all available pages, deduplicates by slug, maps to the
shared JobDocument schema, and writes the result to data/cleaned/arbeitnow_jobs.json.

Usage:
    python src/data_pipeline/fetch_arbeitnow.py

Output:
    data/arbeitnow/arbeitnow_jobs.json   — list of JobDocument dicts
"""

import json
import os
import re
import time

import requests

try:
    from src.data_pipeline.schemas import JobDocument
except ImportError:
    from schemas import JobDocument

# ── Config ─────────────────────────────────────────────────────────────
API_URL = "https://www.arbeitnow.com/api/job-board-api"
MAX_PAGES = 30          # safety cap (API has ~20–25 real pages)
REQUEST_DELAY = 1.0     # seconds between pages (be polite)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "arbeitnow", "arbeitnow_jobs.json")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# ── HTML stripping (same logic as parse_kaggle.py) ─────────────────────
def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Field mapping ──────────────────────────────────────────────────────
def map_to_document(raw: dict) -> JobDocument | None:
    """
    Map one Arbeitnow API job object to a JobDocument.
    Returns None if required fields are missing or invalid.
    """
    slug = (raw.get("slug") or "").strip()
    title = (raw.get("title") or "").strip()
    company = (raw.get("company_name") or "").strip()
    description_html = raw.get("description") or ""
    description = strip_html(description_html)

    if not slug or not title or not company or len(description) < 50:
        return None  # skip junk rows

    # work_type: prefer "Remote" if remote flag is True, else first job_type
    job_types = raw.get("job_types") or []
    if raw.get("remote"):
        work_type = "Remote"
    elif job_types:
        work_type = job_types[0]
    else:
        work_type = None

    # skill_labels: join tags list
    tags = raw.get("tags") or []
    skill_labels = ", ".join(tags) if tags else None

    try:
        return JobDocument(
            job_id=slug,
            title=title,
            company=company,
            description=description,
            skill_labels=skill_labels,
            location=raw.get("location") or None,
            experience_level=None,   # Arbeitnow does not provide this
            work_type=work_type,
            min_salary=None,         # Arbeitnow does not provide this
            max_salary=None,         # Arbeitnow does not provide this
            url=raw.get("url") or None,
            source="arbeitnow",
        )
    except Exception as e:
        print(f"  Validation error for slug={slug!r}: {e}")
        return None


# ── Pagination ─────────────────────────────────────────────────────────
def fetch_all() -> list[dict]:
    seen_slugs: set[str] = set()
    documents: list[dict] = []

    for page in range(1, MAX_PAGES + 1):
        print(f"Fetching page {page}...", end=" ", flush=True)
        try:
            resp = requests.get(API_URL, params={"page": page}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            break

        data = resp.json()
        jobs = data.get("data") or []

        if not jobs:
            print("empty — done.")
            break

        page_count = 0
        for raw in jobs:
            slug = (raw.get("slug") or "").strip()
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            doc = map_to_document(raw)
            if doc:
                documents.append(doc.model_dump())
                page_count += 1

        print(f"{page_count} added (total: {len(documents)})")

        # Check if there's a next page
        links = data.get("links") or {}
        if not links.get("next"):
            print("No next page — done.")
            break

        time.sleep(REQUEST_DELAY)

    return documents


def main():
    print("=" * 60)
    print("Arbeitnow API — Fetch Snapshot")
    print("=" * 60)

    documents = fetch_all()

    print(f"\nTotal valid documents: {len(documents):,}")
    print(f"Saving to {OUTPUT_PATH}...")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(documents):,} jobs.")

    # Quick stats
    work_types = {}
    for d in documents:
        wt = d.get("work_type") or "(null)"
        work_types[wt] = work_types.get(wt, 0) + 1
    print("\nWork type distribution:")
    for wt, cnt in sorted(work_types.items(), key=lambda x: -x[1]):
        print(f"  {wt:20s}: {cnt:>5,}")


if __name__ == "__main__":
    main()
