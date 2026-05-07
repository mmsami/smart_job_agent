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

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"
METHODS = ["BM25_RAW", "BM25_PARSED", "FAISS_RAW", "FAISS_PARSED", "FAISS_PARSED_NORERANK", "FAISS_PARSED_MPNET"]
MODELS = ["gemma", "deepseek", "claude"]
CANONICAL_MODEL = "gemma"  # P@10 uses this model's MD (jobs identical across models)
DEFAULT_WORK_TYPE = "full-time"  # fixed preference used in run_evaluation.py


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


# ── Qualitative Observations ──────────────────────────────────────────────────

def load_json_results(persona_dir: Path, method: str) -> list[dict]:
    """Load top-10 job records from the canonical gemma JSON for a method."""
    json_path = persona_dir / f"{method}_{CANONICAL_MODEL}.json"
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("results", [])[:10]
    except Exception:
        return []


def diversity_metrics(jobs: list[dict]) -> dict:
    """Unique companies, experience levels, and title stems in top-10."""
    if not jobs:
        return {}
    companies = {str(j.get("company", "")).strip().lower() for j in jobs if j.get("company")}
    levels = {str(j.get("experience_level", "")).strip().lower() for j in jobs if j.get("experience_level")}
    title_stems = {" ".join(str(j.get("title", "")).split()[:3]).lower() for j in jobs if j.get("title")}
    return {
        "unique_companies": len(companies),
        "unique_levels": len(levels),
        "unique_title_stems": len(title_stems),
    }


def preference_satisfaction(jobs: list[dict], target_work_type: str = DEFAULT_WORK_TYPE) -> float | None:
    """Fraction of top-10 jobs where work_type matches stated preference (where metadata exists)."""
    with_data = [j for j in jobs if j.get("work_type")]
    if not with_data:
        return None
    matched = sum(1 for j in with_data if str(j["work_type"]).lower() == target_work_type.lower())
    return round(matched / len(with_data), 2)


def collect_qualitative(persona_dirs: list[Path]) -> dict:
    """
    Returns {method: {"unique_companies": [float], "unique_levels": [float],
                       "unique_title_stems": [float], "pref_satisfaction": [float]}}
    """
    data: dict[str, dict[str, list]] = {
        m: {"unique_companies": [], "unique_levels": [], "unique_title_stems": [], "pref_satisfaction": []}
        for m in METHODS
    }
    for persona_dir in persona_dirs:
        for method in METHODS:
            jobs = load_json_results(persona_dir, method)
            if not jobs:
                continue
            d = diversity_metrics(jobs)
            for k in ("unique_companies", "unique_levels", "unique_title_stems"):
                if k in d:
                    data[method][k].append(d[k])
            ps = preference_satisfaction(jobs)
            if ps is not None:
                data[method]["pref_satisfaction"].append(ps)
    return data


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


def build_report(p10_data: dict, quality_data: dict, qualitative: dict) -> str:
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

    # ── Hypothesis Tests ──────────────────────────────────────────────────────

    def _paired(method_a: str, method_b: str) -> tuple[list, list]:
        """Extract paired P@10 scores for two methods across common labeled personas."""
        pa, pb = [], []
        for pid in persona_ids:
            va = p10_data.get(method_a, {}).get(pid)
            vb = p10_data.get(method_b, {}).get(pid)
            if va is not None and vb is not None:
                pa.append(va)
                pb.append(vb)
        return pa, pb

    def _hypothesis_block(label: str, method_a: str, method_b: str, positive_direction: str) -> None:
        pa, pb = _paired(method_a, method_b)
        if not pa:
            lines.append(f"_Not enough labeled data yet for {label}._\n")
            return
        avg_a = round(sum(pa) / len(pa), 3)
        avg_b = round(sum(pb) / len(pb), 3)
        delta = round(avg_a - avg_b, 3)
        p_val = wilcoxon_p(pa, pb)
        p_str = f"{p_val:.4f}" if p_val is not None else "n/a (need ≥4 paired)"
        lines.extend([
            f"- {method_a} avg P@10 = **{avg_a}**",
            f"- {method_b} avg P@10 = **{avg_b}**",
            f"- Delta = **{delta:+.3f}** ({positive_direction})",
            f"- Wilcoxon signed-rank p = {p_str}",
            "",
        ])

    lines.append("## H1: Semantic vs Keyword Search (FAISS_PARSED vs BM25_PARSED)\n")
    _hypothesis_block("H1", "FAISS_PARSED", "BM25_PARSED", "positive = FAISS better")

    lines.append("## H2a: Parsed vs Raw Query — Keyword (BM25_PARSED vs BM25_RAW)\n")
    _hypothesis_block("H2a", "BM25_PARSED", "BM25_RAW", "positive = parsing helps")

    lines.append("## H2b: Parsed vs Raw Query — Semantic (FAISS_PARSED vs FAISS_RAW)\n")
    _hypothesis_block("H2b", "FAISS_PARSED", "FAISS_RAW", "positive = parsing helps")

    lines.append("## H3: Reranking Delta (FAISS_PARSED vs FAISS_PARSED_NORERANK)\n")
    _hypothesis_block("H3", "FAISS_PARSED", "FAISS_PARSED_NORERANK", "positive = reranking helps")

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

    lines.append("")

    # ── Qualitative Observations ──
    lines.append("## Qualitative Observations\n")
    lines.append(
        "Computed automatically from JSON results (no human labels required). "
        "Reported as directional evidence — not hypothesis tests.\n"
    )

    lines.append("### Observation A: Result Diversity (avg across personas)\n")
    lines.append("| Method | Unique Companies | Unique Levels | Unique Title Stems |")
    lines.append("|--------|-----------------|---------------|--------------------|")

    def _avg(lst): return f"{sum(lst)/len(lst):.1f}" if lst else "—"

    for method in METHODS:
        q = qualitative.get(method, {})
        lines.append(
            f"| {method} | {_avg(q.get('unique_companies', []))} "
            f"| {_avg(q.get('unique_levels', []))} "
            f"| {_avg(q.get('unique_title_stems', []))} |"
        )
    lines.append(
        "\n_10 = maximum diversity. Low unique companies = retriever clustering on one employer._\n"
    )

    lines.append("### Observation B: Preference Satisfaction (work_type = full-time)\n")
    lines.append("| Method | % Jobs Matching Preference | Coverage (jobs with work_type metadata) |")
    lines.append("|--------|---------------------------|----------------------------------------|")
    for method in METHODS:
        q = qualitative.get(method, {})
        ps_list = q.get("pref_satisfaction", [])
        avg_ps = f"{sum(ps_list)/len(ps_list)*100:.0f}%" if ps_list else "—"
        lines.append(f"| {method} | {avg_ps} | n={len(ps_list)} personas |")
    lines.append(
        "\n_Kaggle work_type metadata is incomplete — low coverage means the % is less reliable._\n"
    )

    lines += ["---", "_Generated by `score_results.py`. — = not yet labeled._"]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def score_results() -> None:
    print("Reading labeled MD files...")
    p10_data, quality_data = collect_results()

    persona_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    print("Computing qualitative observations from JSON results...")
    qualitative = collect_qualitative(persona_dirs)

    report = build_report(p10_data, quality_data, qualitative)
    print("\n" + report)

    out_path = RESULTS_DIR / "report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    score_results()
