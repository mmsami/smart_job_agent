"""
score_results.py — Compute Precision@10 and Reasoning Quality from labeled MD sheets.

Usage:
    python -m src.evaluation.score_results

Reads labeled MD files from evaluation/results/<persona>/<METHOD>_<model>.md
Outputs a summary report to evaluation/results/report.md and prints to console.

Labeling instructions (fill in the MD files first):
  Experiment A — Relevant (0/1): Is this job relevant to the persona?
                 Relevant = correct domain + within ±1 seniority level.
                 Fill this column in any one model's MD per method (jobs are
                 identical across models for the same method).
  Experiment B — Quality (1-5): How good is the match explanation?
                 1 = wrong/hallucinated, 3 = generic but correct, 5 = specific and insightful.
                 Fill this column in ALL THREE model MDs to compare reasoning quality.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"
METHODS = ["BM25_RAW", "BM25_PARSED", "FAISS_RAW", "FAISS_PARSED", "FAISS_PARSED_NORERANK"]
MODELS = ["gemma", "deepseek", "claude"]
CANONICAL_MODEL = "gemma"  # P@10 uses this model's MD (jobs identical across models)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_md_labels(md_path: Path) -> tuple[list[int], list[int]]:
    """
    Parse a filled-in labeling sheet.
    Returns (relevant_labels, quality_ratings).
    Columns: | # | Relevant (0/1) | Quality (1-5) | Title | Company | ...
    Only rows with a valid value in the column are included.
    """
    relevant: list[int] = []
    quality: list[int] = []

    if not md_path.exists():
        return relevant, quality

    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # cells layout after split: ["", "#", "Relevant", "Quality", "Title", ...]
        if len(cells) < 5:
            continue
        # Skip header and separator rows
        if cells[1] in ("#", "") or "---" in cells[1]:
            continue

        rel_val = cells[2]
        qual_val = cells[3]

        if rel_val in ("0", "1"):
            relevant.append(int(rel_val))

        if qual_val.isdigit() and 1 <= int(qual_val) <= 5:
            quality.append(int(qual_val))

    return relevant, quality


def precision_at_k(labels: list[int], k: int = 10) -> float | None:
    if len(labels) < k:
        return None
    return round(sum(labels[:k]) / k, 2)


def bootstrap_ci(values: list[float], n_boot: int = 10_000, ci: float = 0.95) -> tuple[float, float] | None:
    """Return (lower, upper) 95% bootstrap CI for the mean."""
    if len(values) < 2:
        return None
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return (float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))))


def wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    """Wilcoxon signed-rank p-value for paired samples, or None if not applicable."""
    if len(a) != len(b) or len(a) < 4:
        return None
    try:
        result = wilcoxon(a, b)
        return float(result.pvalue)
    except ValueError:
        return None


# ── Aggregation ───────────────────────────────────────────────────────────────

def collect_results() -> tuple[dict, dict]:
    """
    Returns:
        p10_data:      {method: {persona: float|None}}
        quality_data:  {model: {method: [float]}}  (list of per-run averages)
    """
    p10_data: dict[str, dict[str, float | None]] = {m: {} for m in METHODS}
    quality_data: dict[str, dict[str, list[float]]] = {
        model: {method: [] for method in METHODS} for model in MODELS
    }

    persona_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())

    for persona_dir in persona_dirs:
        persona_id = persona_dir.name

        for method in METHODS:
            # Experiment A: P@10 from canonical model MD
            canonical_md = persona_dir / f"{method}_{CANONICAL_MODEL}.md"
            relevant, _ = parse_md_labels(canonical_md)
            p10_data[method][persona_id] = precision_at_k(relevant)

            # Experiment B: reasoning quality per model
            for model in MODELS:
                model_md = persona_dir / f"{method}_{model}.md"
                _, quality = parse_md_labels(model_md)
                if quality:
                    avg_q = round(sum(quality) / len(quality), 2)
                    quality_data[model][method].append(avg_q)

    return p10_data, quality_data


# ── Report Formatting ─────────────────────────────────────────────────────────

def format_val(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


def build_report(p10_data: dict, quality_data: dict) -> str:
    persona_ids = sorted({
        pid
        for method_dict in p10_data.values()
        for pid in method_dict
    })

    lines = ["# Evaluation Report\n"]

    # ── Experiment A ──
    lines.append("## Experiment A: Precision@10 by Retrieval Method\n")
    lines.append(
        "Relevant = correct domain + within ±1 seniority. "
        f"Labels from `_{CANONICAL_MODEL}.md` (jobs identical across models).\n"
    )

    header = "| Method | " + " | ".join(persona_ids) + " | **Average** | 95% CI |"
    sep    = "|--------|" + "---------|" * len(persona_ids) + "-------------|--------|"
    lines += [header, sep]

    method_scores: dict[str, list[float]] = {}
    for method in METHODS:
        vals = [p10_data[method].get(pid) for pid in persona_ids]
        labeled = [v for v in vals if v is not None]
        avg = round(sum(labeled) / len(labeled), 2) if labeled else None
        ci = bootstrap_ci(labeled) if labeled else None
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        row = f"| {method} | " + " | ".join(format_val(v) for v in vals)
        row += f" | **{format_val(avg)}** | {ci_str} |"
        lines.append(row)
        method_scores[method] = labeled

    lines.append("")

    # ── H3: Reranking delta (FAISS_PARSED vs FAISS_PARSED_NORERANK) ──
    lines.append("## H3: Reranking Delta (FAISS_PARSED vs FAISS_PARSED_NORERANK)\n")
    a_scores = method_scores.get("FAISS_PARSED", [])
    b_scores = method_scores.get("FAISS_PARSED_NORERANK", [])
    # Align to common personas
    paired_a, paired_b = [], []
    for pid in persona_ids:
        va = p10_data.get("FAISS_PARSED", {}).get(pid)
        vb = p10_data.get("FAISS_PARSED_NORERANK", {}).get(pid)
        if va is not None and vb is not None:
            paired_a.append(va)
            paired_b.append(vb)
    if paired_a:
        avg_reranked  = round(sum(paired_a) / len(paired_a), 3)
        avg_norerank  = round(sum(paired_b) / len(paired_b), 3)
        delta         = round(avg_reranked - avg_norerank, 3)
        p_val         = wilcoxon_p(paired_a, paired_b)
        p_str         = f"{p_val:.4f}" if p_val is not None else "n/a (need ≥4 paired)"
        lines += [
            f"- FAISS_PARSED (reranked) avg P@10 = **{avg_reranked}**",
            f"- FAISS_PARSED_NORERANK avg P@10 = **{avg_norerank}**",
            f"- Delta = **{delta:+.3f}** (positive = reranking helps)",
            f"- Wilcoxon signed-rank p = {p_str}",
            "",
        ]
    else:
        lines.append("_Not enough labeled data yet._\n")

    # ── Experiment B ──
    lines.append("## Experiment B: Reasoning Quality by Model (avg score 1–5)\n")
    lines.append(
        "Quality rated 1–5 per job explanation. "
        "1 = wrong/hallucinated, 3 = generic but correct, 5 = specific and insightful.\n"
    )

    header2 = "| Method | " + " | ".join(MODELS) + " |"
    sep2    = "|--------|" + "--------|" * len(MODELS)
    lines += [header2, sep2]

    for method in METHODS:
        row = f"| {method} |"
        for model in MODELS:
            scores = quality_data[model][method]
            avg = round(sum(scores) / len(scores), 2) if scores else None
            row += f" {format_val(avg)} |"
        lines.append(row)

    lines.append("")

    # Wilcoxon: Gemma vs DeepSeek and Gemma vs Claude (paired by method+persona)
    lines.append("### Wilcoxon Tests: Reasoning Quality\n")
    for other_model in ["deepseek", "claude"]:
        gemma_flat, other_flat = [], []
        for method in METHODS:
            ga = quality_data["gemma"][method]
            ob = quality_data[other_model][method]
            n = min(len(ga), len(ob))
            gemma_flat.extend(ga[:n])
            other_flat.extend(ob[:n])
        p = wilcoxon_p(gemma_flat, other_flat)
        p_str = f"{p:.4f}" if p is not None else "n/a (need ≥4 paired)"
        lines.append(f"- gemma vs {other_model}: p = {p_str}")

    lines += ["", "---", "_Generated by `score_results.py`. — = not yet labeled._"]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def score_results() -> None:
    print("Reading labeled MD files...")
    p10_data, quality_data = collect_results()

    report = build_report(p10_data, quality_data)
    print("\n" + report)

    out_path = RESULTS_DIR / "report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    score_results()
