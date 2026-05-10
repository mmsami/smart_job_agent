"""
validate_results.py — Pre-labeling integrity check.

Scans all result JSON files and flags any that have:
  - Fewer than 10 job explanations
  - Empty or placeholder match_reason ("" or "—")
  - Duplicate job_ids inside job_explanations
  - Blank title or company in any explanation
  - job_ids in reasoning that don't match result job_ids
  - Empty recommendation or cv_summary
  - Malformed / unreadable JSON

Run this after run_evaluation and before labeling starts.

Usage:
    python -m src.evaluation.validate_results
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"


def validate_results() -> None:
    json_files = sorted(OUTPUT_DIR.glob("*/*.json"))
    if not json_files:
        logger.info("No result files found.")
        return

    issues: list[str] = []

    for path in json_files:
        label = f"{path.parent.name}/{path.name}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(f"  [MALFORMED JSON] {label} — {e}")
            continue

        reasoning = data.get("reasoning", {})
        explanations = reasoning.get("job_explanations", [])

        if len(explanations) < 10:
            issues.append(f"  [INCOMPLETE] {label} — {len(explanations)}/10 explanations")
            continue

        empty = [
            e.get("job_id", "?")
            for e in explanations
            if not (e.get("match_reason") or "").strip()
            or (e.get("match_reason") or "").strip() == "—"
        ]
        if empty:
            issues.append(f"  [EMPTY MATCH_REASON] {label} — job_ids: {empty}")

        # Duplicate job_ids
        seen_ids: list[str] = [e.get("job_id", "") for e in explanations]
        dupes = {jid for jid in seen_ids if seen_ids.count(jid) > 1}
        if dupes:
            issues.append(f"  [DUPLICATE JOB_IDS] {label} — {dupes}")

        # Blank title or company
        blank_meta = [
            e.get("job_id", "?")
            for e in explanations
            if not (e.get("title") or "").strip() or not (e.get("company") or "").strip()
        ]
        if blank_meta:
            issues.append(f"  [BLANK TITLE/COMPANY] {label} — job_ids: {blank_meta}")

        # job_ids in reasoning don't match result job_ids
        result_ids = {str(r.get("job_id", "")) for r in data.get("results", [])}
        reasoning_ids = {e.get("job_id", "") for e in explanations}
        mismatch = reasoning_ids - result_ids
        if mismatch:
            issues.append(f"  [JOB_ID MISMATCH] {label} — reasoning has unknown ids: {mismatch}")

        # Empty top-level fields
        if not (reasoning.get("recommendation") or "").strip():
            issues.append(f"  [EMPTY RECOMMENDATION] {label}")
        if not (reasoning.get("cv_summary") or "").strip():
            issues.append(f"  [EMPTY CV_SUMMARY] {label}")

    logger.info("=" * 60)
    logger.info(f"Validated {len(json_files)} result files")
    if issues:
        logger.info(f"Issues found ({len(issues)}) — delete these files and re-run:")
        for issue in issues:
            logger.info(issue)
    else:
        logger.info("All files passed — safe to start labeling.")
    logger.info("=" * 60)


if __name__ == "__main__":
    validate_results()
