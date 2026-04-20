"""
test_cv_profiler.py — Controlled testing of CV profiling (raw text → CVProfile JSON).

Usage:
  python -m tests.workflow.test_cv_profiler                      # scan all + confirm
  python -m tests.workflow.test_cv_profiler data/resumes/cv.pdf  # test single file

Results logged to: project/iterations/cv_profiler_results.md

CACHING:
  cv_reader uses disk cache (keyed by PDF content hash) — no vision API calls on re-runs.
  cv_profiler uses disk cache (keyed by text hash) — no Gemma calls on re-runs.

  This means: same CV content (even if renamed) hits cache → zero API cost on re-test.

CACHE CLEARING (for development/debugging):
  After code fixes, you must clear the cache to test changes:

    rm -rf .cache/cv_reader .cache/cv_profiler

  Then re-run the test. Next run will make fresh API calls and show the fix.

  Alternatively, bump version numbers in cv_reader.py or cv_profiler.py:
    - Change PROMPT_VERSION = "v2" in cv_reader.py
    - Change LOGIC_VERSION = "v10" in cv_profiler.py
  This auto-invalidates old cache entries without deleting files.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

from src.workflow.cv_reader import extract_text_from_pdf
from src.workflow.cv_profiler import profile_cv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESUMES_DIR = Path(__file__).parent.parent.parent / "data" / "resumes"
RESULTS_FILE = Path(__file__).parent.parent.parent / "iterations" / "cv_profiler_results.md"


def _find_all_pdfs() -> list[Path]:
    if not RESUMES_DIR.exists():
        logger.warning(f"Directory not found: {RESUMES_DIR}")
        return []
    return sorted(RESUMES_DIR.glob("*.pdf"))


def _test_single_file(pdf_path: Path) -> dict:
    if not pdf_path.exists():
        return {"file": pdf_path.name, "status": "ERROR", "error": "File not found"}

    logger.info(f"\nProcessing: {pdf_path.name}")
    try:
        raw_text = extract_text_from_pdf(pdf_path, use_cache=True)
        profile = profile_cv(raw_text, use_cache=True)
        return {
            "file": pdf_path.name,
            "status": "OK",
            "profile": profile.model_dump(),
        }
    except Exception as e:
        return {"file": pdf_path.name, "status": "ERROR", "error": str(e)[:300]}


def _write_results(results: list[dict]):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = RESULTS_FILE.read_text() if RESULTS_FILE.exists() else ""
    timestamp = datetime.now().isoformat()
    section = f"## Test Run: {timestamp}\n\n"

    for r in results:
        section += f"### {r['file']}\n"
        section += f"**Status:** {r['status']}\n\n"
        if r["status"] == "OK":
            section += "```json\n"
            section += json.dumps(r["profile"], indent=2)
            section += "\n```\n\n"
        else:
            section += f"**Error:** {r.get('error')}\n\n"

    content = existing + section if existing else f"# CV Profiler Results\n\n{section}"
    RESULTS_FILE.write_text(content)
    logger.info(f"\n✓ Results appended to: {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Test cv_profiler on single or batch CVs")
    parser.add_argument("file", nargs="?", type=Path, help="Single PDF to test (optional)")
    args = parser.parse_args()

    results = []

    if args.file:
        results.append(_test_single_file(args.file))
    else:
        pdfs = _find_all_pdfs()
        if not pdfs:
            logger.warning("No PDFs found in data/resumes/")
            sys.exit(1)

        logger.info(f"Found {len(pdfs)} PDF(s):")
        for p in pdfs:
            logger.info(f"  - {p.name}")

        confirm = input(f"\nRun profiling on all {len(pdfs)} files? (y/n): ").strip().lower()
        if confirm != "y":
            logger.info("Cancelled.")
            sys.exit(0)

        logger.info("\nStarting profiling (cache enabled — cache hits skip API calls)...\n")
        for pdf_path in pdfs:
            results.append(_test_single_file(pdf_path))

    _write_results(results)
    logger.info(f"✓ Tested {len(results)} file(s)")


if __name__ == "__main__":
    main()
