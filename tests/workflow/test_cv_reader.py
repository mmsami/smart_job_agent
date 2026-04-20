"""
test_cv_reader.py — Controlled testing of CV text extraction with caching.

Usage:
  python -m tests.workflow.test_cv_reader                      # scan all + confirm
  python -m tests.workflow.test_cv_reader data/resumes/cv.pdf  # test single file

Results logged to: project/iterations/cv_extraction_results.md
Full extracted text appended per CV. Cache hits logged to avoid unnecessary API calls.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

from src.workflow.cv_reader import extract_text_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

RESUMES_DIR = Path(__file__).parent.parent.parent / "data" / "resumes"
RESULTS_FILE = Path(__file__).parent.parent.parent / "iterations" / "cv_extraction_results.md"


def _find_all_pdfs() -> list[Path]:
    """Find all PDF files in data/resumes/."""
    if not RESUMES_DIR.exists():
        logger.warning(f"Directory not found: {RESUMES_DIR}")
        return []
    return sorted(RESUMES_DIR.glob("*.pdf"))


def _test_single_file(pdf_path: Path) -> dict:
    """Test extraction on a single PDF. Returns result dict with full text."""
    if not pdf_path.exists():
        return {
            "file": pdf_path.name,
            "size_kb": 0,
            "status": "ERROR",
            "error": "File not found",
            "text_length": 0,
            "text": "",
        }

    size_kb = pdf_path.stat().st_size // 1024
    logger.info(f"Testing: {pdf_path.name} ({size_kb} KB)")

    try:
        text = extract_text_from_pdf(pdf_path, use_cache=True)
        return {
            "file": pdf_path.name,
            "size_kb": size_kb,
            "status": "OK",
            "text_length": len(text),
            "text": text,
        }
    except Exception as e:
        return {
            "file": pdf_path.name,
            "size_kb": size_kb,
            "status": "ERROR",
            "error": str(e)[:300],
            "text_length": 0,
            "text": "",
        }


def _write_results(results: list[dict]):
    """Append results to consolidated markdown file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Read existing content
    existing = ""
    if RESULTS_FILE.exists():
        existing = RESULTS_FILE.read_text()

    # Build new results section
    timestamp = datetime.now().isoformat()
    section = f"## Test Run: {timestamp}\n\n"

    for r in results:
        section += f"### {r['file']} ({r['size_kb']} KB)\n"
        section += f"**Status:** {r['status']}\n"

        if r["status"] == "OK":
            section += f"**Extracted:** {r['text_length']} chars\n\n"
            section += "#### Extracted Text\n\n"
            section += f"```\n{r['text']}\n```\n\n"
        else:
            section += f"**Error:** {r.get('error', 'Unknown error')}\n\n"

    # Append to existing or create new
    if existing:
        content = existing + section
    else:
        content = f"# CV Extraction Results\n\n{section}"

    RESULTS_FILE.write_text(content)
    logger.info(f"✓ Results appended to: {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Test cv_reader on single or batch CVs (uses cache to avoid redundant calls)"
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Single PDF to test (optional)",
    )
    args = parser.parse_args()

    results = []

    if args.file:
        # Test single file
        results.append(_test_single_file(args.file))
    else:
        # Scan all PDFs
        pdfs = _find_all_pdfs()
        if not pdfs:
            logger.warning("No PDFs found in data/resumes/")
            sys.exit(1)

        logger.info(f"Found {len(pdfs)} PDF(s):")
        for p in pdfs:
            logger.info(f"  - {p.name}")

        confirm = input(f"\nRun extraction on all {len(pdfs)} files? (y/n): ").strip().lower()
        if confirm != "y":
            logger.info("Cancelled.")
            sys.exit(0)

        logger.info("\nStarting extraction (cache enabled — cache hits skip API calls)...\n")
        for pdf_path in pdfs:
            results.append(_test_single_file(pdf_path))

    _write_results(results)
    logger.info(f"\n✓ Tested {len(results)} file(s)")


if __name__ == "__main__":
    main()
