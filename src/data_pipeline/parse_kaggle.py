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

VALID_EXPERIENCE_LEVELS = [
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
    "Internship",
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the merged dataframe."""
    # ── Strip HTML from descriptions ───────────────────────────────────
    print("Stripping HTML from descriptions...")
    df["description"] = df["description"].apply(strip_html)

    # ── Drop junk rows ─────────────────────────────────────────────────
    title_bad = df["title"].isna() | (df["title"].str.len() < MIN_TITLE_CHARS)
    desc_bad = df["description"].isna() | (
        df["description"].str.len() < MIN_DESCRIPTION_CHARS
    )
    junk = title_bad | desc_bad
    n_junk = junk.sum()
    df = df.loc[~junk].copy()
    print(f"  Dropped {n_junk:,} junk rows")

    # ── Validate experience_level ──────────────────────────────────────
    unexpected_exp = df["formatted_experience_level"].notna() & ~df[
        "formatted_experience_level"
    ].isin(VALID_EXPERIENCE_LEVELS)
    n_unexpected_exp = unexpected_exp.sum()
    if int(n_unexpected_exp) > 0:
        bad_vals = df.loc[unexpected_exp, "formatted_experience_level"].unique()
        print(f"  Setting {n_unexpected_exp:,} unexpected experience_level values to null: {bad_vals}")
        df.loc[unexpected_exp, "formatted_experience_level"] = None

    # ── Add source column ──────────────────────────────────────────────
    df["source"] = "kaggle"

    # ── Select and order columns ───────────────────────────────────────
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df.loc[:, available].copy()

    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate postings by URL, then by (title, company) pair."""
    n_before = len(df)

    # Deduplicate by URL first — same URL = same posting
    has_url = df["application_url"].notna() & (df["application_url"] != "")
    df_with_url = df[has_url].drop_duplicates(subset=["application_url"], keep="first")
    df_no_url = df[~has_url]
    df = pd.concat([df_with_url, df_no_url], ignore_index=True)
    n_after_url = len(df)
    print(f"  URL dedup: removed {n_before - n_after_url:,} rows ({n_after_url:,} remaining)")

    # Deduplicate by normalized (title, company)
    df["_title_norm"] = df["title"].fillna("").str.strip().str.lower()
    df["_company_norm"] = df["company_name"].fillna("").str.strip().str.lower()
    df = df.drop_duplicates(subset=["_title_norm", "_company_norm"], keep="first")
    df = df.drop(columns=["_title_norm", "_company_norm"])
    n_after_tc = len(df)
    print(f"  Title+company dedup: removed {n_after_url - n_after_tc:,} rows ({n_after_tc:,} remaining)")

    return df


def build_quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> list[str]:
    """Generate quality report comparing before/after cleaning."""
    report = []
    n_start = len(df_before)
    n_final = len(df_after)

    report.append(f"Starting rows: {n_start:,}")
    report.append(f"Final rows: {n_final:,}")
    report.append(f"Rows retained: {n_final / n_start * 100:.1f}%")
    report.append("")

    # Column-level nulls
    report.append("Column null counts:")
    for col in df_after.columns:
        null_n = df_after[col].isna().sum()
        null_pct = null_n / len(df_after) * 100
        report.append(f"  {col:35s}: {null_n:>7,} ({null_pct:5.1f}%)")

    # Description length stats
    desc_len = df_after["description"].dropna().str.len()
    report.append("")
    report.append(
        f"Description length — median: {desc_len.median():.0f}, mean: {desc_len.mean():.0f}, min: {desc_len.min():.0f}, max: {desc_len.max():.0f}"
    )

    # Experience level distribution
    report.append("")
    report.append("Experience level distribution:")
    exp_counts = df_after["formatted_experience_level"].value_counts(dropna=False)
    for val, cnt in exp_counts.items():
        label = val if pd.notna(val) else "(null)"
        report.append(f"  {label:25s}: {cnt:>7,} ({cnt / len(df_after) * 100:.1f}%)")

    # Skill labels coverage
    has_skills = df_after["skill_labels"].notna().sum()
    report.append("")
    report.append(
        f"Skill labels coverage: {has_skills:,} / {len(df_after):,} ({has_skills / len(df_after) * 100:.1f}%)"
    )

    # Industry coverage
    has_ind = df_after["industries"].notna().sum()
    report.append(
        f"Industry coverage: {has_ind:,} / {len(df_after):,} ({has_ind / len(df_after) * 100:.1f}%)"
    )

    return report


def main():
    print("=" * 60)
    print("Kaggle LinkedIn Job Postings — Data Pipeline")
    print("=" * 60)

    # Load and clean
    print("\nLoading and joining...")
    df_raw = load_and_join(DATA_DIR)
    n_raw = len(df_raw)

    print("\nCleaning...")
    df_clean = clean(df_raw)

    print("\nDeduplicating...")
    df_dedup = deduplicate(df_clean)

    # Generate report
    print("\nGenerating quality report...")
    report = build_quality_report(df_raw, df_dedup)

    # Save outputs
    print("\nSaving outputs...")
    out_csv = os.path.join(OUTPUT_DIR, "postings_cleaned.csv")
    df_dedup.to_csv(out_csv, index=False)
    print(f"  Saved {len(df_dedup):,} rows to {out_csv}")

    out_report = os.path.join(OUTPUT_DIR, "data_quality_report.txt")
    report_text = "\n".join(report)
    with open(out_report, "w") as f:
        f.write("Kaggle Data Quality Report\n")
        f.write("=" * 40 + "\n")
        f.write(report_text)
    print(f"  Saved report to {out_report}")

    print("\n" + report_text)
    print("\nDone.")


if __name__ == "__main__":
    main()
