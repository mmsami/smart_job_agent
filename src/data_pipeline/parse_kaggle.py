"""
Parse Kaggle LinkedIn Job Postings dataset into clean documents ready for embedding.

Joins postings.csv with job_skills and job_industries tables, cleans data,
and outputs a single cleaned CSV with all fields needed for the pipeline.

Usage:
    python src/data_pipeline/parse_kaggle.py

Output:
    data/kaggle_cleaned/postings_cleaned.csv
    data/kaggle_cleaned/data_quality_report.txt
"""

import os
import re

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "kaggle_cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
MIN_DESCRIPTION_CHARS = 50
MIN_TITLE_CHARS = 3

KEEP_COLUMNS = [
    "job_id",
    "title",
    "company_name",
    "description",
    "location",
    "formatted_experience_level",
    "formatted_work_type",
    "min_salary",
    "max_salary",
    "application_url",
    "skill_labels",
    "industries",
    "source",
]


def strip_html(text):
    """Remove HTML tags and decode common entities."""
    if pd.isna(text):
        return text
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_join(data_dir):
    """Load postings and join with skills + industries tables."""
    print("Loading postings.csv...")
    postings = pd.read_csv(os.path.join(data_dir, "kaggle_raw", "postings.csv"))
    print(f"  Loaded {len(postings):,} rows, {len(postings.columns)} columns")

    # ── Skills join ────────────────────────────────────────────────────
    print("Joining skill labels...")
    job_skills = pd.read_csv(os.path.join(data_dir, "kaggle_raw", "jobs", "job_skills.csv"))
    skills_map = pd.read_csv(os.path.join(data_dir, "kaggle_raw", "mappings", "skills.csv"))

    skills_joined = job_skills.merge(skills_map, on="skill_abr")
    skills_per_job = (
        skills_joined.groupby("job_id")["skill_name"].apply(", ".join).reset_index()
    )
    skills_per_job.columns = ["job_id", "skill_labels"]
    print(f"  Skill labels available for {skills_per_job['job_id'].nunique():,} jobs")

    # ── Industries join ────────────────────────────────────────────────
    print("Joining industry names...")
    job_industries = pd.read_csv(os.path.join(data_dir, "kaggle_raw", "jobs", "job_industries.csv"))
    industries_map = pd.read_csv(os.path.join(data_dir, "kaggle_raw", "mappings", "industries.csv"))
    industries_map = industries_map.dropna(subset=["industry_name"])

    industry_joined = job_industries.merge(industries_map, on="industry_id")
    industry_joined = industry_joined.dropna(subset=["industry_name"])
    industry_per_job = (
        industry_joined.groupby("job_id")["industry_name"]
        .apply(", ".join)
        .reset_index()
    )
    industry_per_job.columns = ["job_id", "industries"]
    print(
        f"  Industry data available for {industry_per_job['job_id'].nunique():,} jobs"
    )

    # ── Merge onto postings ────────────────────────────────────────────
    postings = postings.merge(skills_per_job, on="job_id", how="left")
    postings = postings.merge(industry_per_job, on="job_id", how="left")

    return postings


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean the merged dataframe."""
    report = []
    n_start = len(df)
    report.append(f"Starting rows: {n_start:,}")

    # ── Strip HTML from descriptions ───────────────────────────────────
    print("Stripping HTML from descriptions...")
    df["description"] = df["description"].apply(strip_html)

    # ── Drop junk rows (both title AND description unusable) ───────────
    title_bad = df["title"].isna() | (df["title"].str.len() < MIN_TITLE_CHARS)
    desc_bad = df["description"].isna() | (
        df["description"].str.len() < MIN_DESCRIPTION_CHARS
    )
    both_bad = title_bad & desc_bad
    n_junk = both_bad.sum()
    df = df.loc[~both_bad].copy()
    report.append(f"Dropped (both title & desc junk): {n_junk:,}")
    print(f"  Dropped {n_junk:,} junk rows")

    # ── Validate experience_level ──────────────────────────────────────
    valid_exp = [
        "Entry level",
        "Associate",
        "Mid-Senior level",
        "Director",
        "Executive",
        "Internship",
    ]
    unexpected_exp = df["formatted_experience_level"].notna() & ~df[
        "formatted_experience_level"
    ].isin(valid_exp)
    n_unexpected_exp = unexpected_exp.sum()
    if int(n_unexpected_exp) > 0:
        bad_vals = df.loc[unexpected_exp, "formatted_experience_level"].unique()
        report.append(
            f"Unexpected experience_level values set to null: {n_unexpected_exp:,} ({bad_vals})"
        )
        df.loc[unexpected_exp, "formatted_experience_level"] = None
    else:
        report.append("Experience level values: all valid (no unexpected values)")

    # ── Add source column ──────────────────────────────────────────────
    df["source"] = "kaggle"

    # ── Select and order columns ───────────────────────────────────────
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df.loc[:, available].copy()

    # ── Final stats ────────────────────────────────────────────────────
    n_final = len(df)
    report.append(f"Final rows: {n_final:,}")
    report.append(f"Rows retained: {n_final / n_start * 100:.1f}%")
    report.append("")

    # Column-level nulls
    report.append("Column null counts:")
    for col in df.columns:
        null_n = df[col].isna().sum()
        null_pct = null_n / len(df) * 100
        report.append(f"  {col:35s}: {null_n:>7,} ({null_pct:5.1f}%)")

    # Description length stats
    desc_len = df["description"].dropna().str.len()
    report.append("")
    report.append(
        f"Description length — median: {desc_len.median():.0f}, mean: {desc_len.mean():.0f}, min: {desc_len.min():.0f}, max: {desc_len.max():.0f}"
    )

    # Experience level distribution
    report.append("")
    report.append("Experience level distribution:")
    exp_counts = df["formatted_experience_level"].value_counts(dropna=False)
    for val, cnt in exp_counts.items():
        label = val if bool(pd.notna(val)) else "(null)"
        report.append(f"  {label:25s}: {cnt:>7,} ({cnt / len(df) * 100:.1f}%)")

    # Skill labels coverage
    has_skills = df["skill_labels"].notna().sum()
    report.append("")
    report.append(
        f"Skill labels coverage: {has_skills:,} / {len(df):,} ({has_skills / len(df) * 100:.1f}%)"
    )

    # Industry coverage
    has_ind = df["industries"].notna().sum()
    report.append(
        f"Industry coverage: {has_ind:,} / {len(df):,} ({has_ind / len(df) * 100:.1f}%)"
    )

    return df, report


def main():
    print("=" * 60)
    print("Kaggle LinkedIn Job Postings — Data Pipeline")
    print("=" * 60)

    df = load_and_join(DATA_DIR)
    df, report = clean(df)

    # ── Save cleaned CSV ───────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "postings_cleaned.csv")
    print(f"\nSaving cleaned data to {out_csv}...")
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df):,} rows")

    # ── Save quality report ────────────────────────────────────────────
    out_report = os.path.join(OUTPUT_DIR, "data_quality_report.txt")
    report_text = "\n".join(report)
    with open(out_report, "w") as f:
        f.write("Kaggle Data Quality Report\n")
        f.write("=" * 40 + "\n")
        f.write(report_text)
    print(f"  Quality report saved to {out_report}")

    print("\n" + report_text)
    print("\nDone.")


if __name__ == "__main__":
    main()
