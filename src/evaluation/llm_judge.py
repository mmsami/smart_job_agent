"""
llm_judge.py — LLM-as-judge evaluation (H4: Human vs LLM labels, Cohen's Kappa).

For each persona × method, reads human Relevant labels from *_gemini.md and asks
Claude Sonnet to make the same 0/1 relevance judgment for each job. Computes
Cohen's Kappa to quantify agreement between human annotators and the LLM judge.

Usage:
    python -m src.evaluation.llm_judge                          # all personas/methods
    python -m src.evaluation.llm_judge --method FAISS_PARSED    # one method
    python -m src.evaluation.llm_judge --persona cv1            # one persona

Output:
    evaluation/results/llm_judge_report.md
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"
ALL_METHODS = ["BM25_RAW", "BM25_PARSED", "FAISS_RAW", "FAISS_PARSED", "FAISS_PARSED_NORERANK", "FAISS_PARSED_MPNET"]
CANONICAL_MODEL = "gemini"
LLM_JUDGE_MODEL = "anthropic/claude-sonnet-4-6"

_JUDGE_PROMPT = """\
You are evaluating job relevance for a job-seeking candidate.

## Candidate Profile
{cv_summary}

## Job Posting
**Title:** {title}
**Company:** {company}
**Description:**
{description}

## Task
Is this job relevant for this candidate?

A job is RELEVANT (1) if:
- The role domain matches the candidate's background (same field, not a lateral domain jump)
- The seniority level is within ±1 level (entry/mid/senior)

A job is NOT RELEVANT (0) if it is a different domain, or more than one seniority level away.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"relevant": 0}} or {{"relevant": 1}}\
"""


# ── API ───────────────────────────────────────────────────────────────────────

def _get_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def judge_job(client, cv_summary: str, title: str, company: str, description: str) -> int | None:
    prompt = _JUDGE_PROMPT.format(
        cv_summary=cv_summary,
        title=title,
        company=company,
        description=description[:1500],
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            m = re.search(r'"relevant"\s*:\s*([01])', content)
            if m:
                return int(m.group(1))
        except Exception:
            if attempt == 2:
                return None
    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_human_labels(md_path: Path) -> list[int]:
    labels = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 5 or cells[1] in ("#", "") or "---" in cells[1]:
            continue
        if cells[2] in ("0", "1"):
            labels.append(int(cells[2]))
    return labels


def _load_jobs(json_path: Path, n: int) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("results", [])[:n]


def _load_cv_summary(persona_dir: Path) -> str:
    for json_file in sorted(persona_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            summary = data.get("reasoning", {}).get("cv_summary", "")
            if summary:
                return summary
        except Exception:
            continue
    return ""


# ── Statistics ────────────────────────────────────────────────────────────────

def cohen_kappa(human: list[int], llm: list[int]) -> float | None:
    n = len(human)
    if n == 0 or len(llm) != n:
        return None
    po = sum(h == l for h, l in zip(human, llm)) / n
    p1_h = sum(human) / n
    p1_l = sum(llm) / n
    pe = p1_h * p1_l + (1 - p1_h) * (1 - p1_l)
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_llm_judge(method_filter: str | None = None, persona_filter: str | None = None) -> None:
    client = _get_client()

    persona_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    if persona_filter:
        persona_dirs = [d for d in persona_dirs if d.name == persona_filter]

    methods = [m for m in ALL_METHODS if method_filter is None or m == method_filter]

    rows = []

    for persona_dir in persona_dirs:
        persona_id = persona_dir.name
        cv_summary = _load_cv_summary(persona_dir)

        for method in methods:
            md_path   = persona_dir / f"{method}_{CANONICAL_MODEL}.md"
            json_path = persona_dir / f"{method}_{CANONICAL_MODEL}.json"

            if not md_path.exists() or not json_path.exists():
                continue

            human_labels = _parse_human_labels(md_path)
            if not human_labels:
                print(f"  {persona_id} | {method}: no human labels yet — skipping")
                continue

            jobs = _load_jobs(json_path, len(human_labels))
            llm_labels = []
            for job in jobs:
                label = judge_job(
                    client,
                    cv_summary=cv_summary,
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    description=job.get("description", ""),
                )
                if label is None:
                    print(f"  WARNING: LLM judge returned None for '{job.get('title', '?')}' — treating as 0")
                llm_labels.append(label if label is not None else 0)

            kappa    = cohen_kappa(human_labels, llm_labels)
            human_p10 = round(sum(human_labels[:10]) / 10, 2) if len(human_labels) >= 10 else None
            llm_p10   = round(sum(llm_labels[:10])   / 10, 2) if len(llm_labels)   >= 10 else None

            rows.append({
                "persona":    persona_id,
                "method":     method,
                "human_p10":  human_p10,
                "llm_p10":    llm_p10,
                "kappa":      kappa,
                "n":          len(human_labels),
            })
            k_str = f"{kappa:.3f}" if kappa is not None else "n/a"
            print(f"  {persona_id} | {method}: κ={k_str}  human_p10={human_p10}  llm_p10={llm_p10}")

    # ── Build report ──────────────────────────────────────────────────────────
    lines = [
        "# LLM Judge Report (H4: Human vs LLM Labels)\n",
        f"Judge model: `{LLM_JUDGE_MODEL}` (via OpenRouter). "
        "Cohen's Kappa measures agreement with human labels.\n",
        "| Persona | Method | Human P@10 | LLM P@10 | κ (Kappa) | n |",
        "|---------|--------|------------|----------|-----------|---|",
    ]
    for r in rows:
        h = f"{r['human_p10']:.2f}" if r["human_p10"] is not None else "—"
        lv = f"{r['llm_p10']:.2f}" if r["llm_p10"] is not None else "—"
        k = f"{r['kappa']:.3f}"    if r["kappa"]    is not None else "—"
        lines.append(f"| {r['persona']} | {r['method']} | {h} | {lv} | {k} | {r['n']} |")

    lines += ["", "## Average Kappa by Method\n"]
    all_kappas = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method and r["kappa"] is not None]
        if method_rows:
            avg_k = sum(r["kappa"] for r in method_rows) / len(method_rows)
            all_kappas.extend(r["kappa"] for r in method_rows)
            lines.append(f"- **{method}**: κ = {avg_k:.3f}  (n={len(method_rows)} personas)")

    if all_kappas:
        overall = sum(all_kappas) / len(all_kappas)
        lines += [
            f"\n**Overall average κ = {overall:.3f}**\n",
            "> κ < 0.20 = poor agreement | 0.20–0.40 = fair | 0.40–0.60 = moderate | "
            "0.60–0.80 = substantial | > 0.80 = almost perfect",
        ]

    lines += ["", "---", "_Generated by `llm_judge.py`._"]

    out_path = RESULTS_DIR / "llm_judge_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation (H4)")
    parser.add_argument("--method",  default=None, help="Restrict to one retrieval method")
    parser.add_argument("--persona", default=None, help="Restrict to one persona")
    args = parser.parse_args()
    run_llm_judge(method_filter=args.method, persona_filter=args.persona)
