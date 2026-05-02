"""
main.py — The Smart Job Market Agent Orchestrator.

UX Flow:
  1. Parallel Onboarding: Profile CV (BG thread) while gathering preferences (FG).
     Preferences are collected as freeform natural language — no rigid form.
  2. Verification: Show CVProfile and allow natural language edits.
  3. Pipeline: Job Retrieval -> Reranking -> Match Reasoning.
  4. Refinement Loop: After results, user can refine preferences and re-run.
  5. Tiered Output:
     - Top 3 matches: High-detail Rich Panels.
     - Matches 4-10: Compact Rich Table.
     - Roadmap: Overall missing skills.
  6. Persistence: Save the full session result to 'iterations/results_[cv_name].md'.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging as _logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.workflow.cv_profiler import profile_cv
from src.workflow.cv_reader import extract_text_from_pdf
from src.workflow.job_search import retrieve_jobs
from src.workflow.models import CVProfile, JobRecord, JobSearchPreferences
from src.workflow.pref_parser import (
    apply_profile_edit,
    parse_preferences,
    refine_preferences,
)
from src.workflow.reasoning import ReasoningReport, analyze_job_matches
from src.workflow.reranker import rerank_jobs

load_dotenv()

# suppress all SDK + internal logs in demo context — noise in Rich prompts


for _noisy in (
    "httpx",
    "httpcore",
    "google.auth",
    "google.genai",
    "google.generativeai",
    "src.workflow",
):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

console = Console()
PROJECT_ROOT = Path(__file__).parent.parent
ITERATIONS_DIR = PROJECT_ROOT / "iterations"
ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)


def print_header():
    console.print(
        Panel.fit(
            "[bold cyan] Smart Job Market Agent[/bold cyan]\n"
            "[italic white]Collaborative AI-powered Career Matcher[/italic white]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


# ── Preference Collection ─────────────────────────────────────────────────────


def _prefs_summary(prefs: JobSearchPreferences) -> str:
    return (
        f"Location: [bold]{prefs.target_location}[/bold]   "
        f"Work type: [bold]{prefs.work_type}[/bold]   "
        f"Remote: [bold]{prefs.remote_preference}[/bold]   "
        f"Relocate: [bold]{'Yes' if prefs.willing_to_relocate else 'No'}[/bold]\n"
        f"Roles: [bold]{', '.join(prefs.target_roles) or 'any'}[/bold]   "
        f"Industries: [bold]{', '.join(prefs.industry_preference) or 'any'}[/bold]"
    )


def collect_preferences() -> JobSearchPreferences:
    """Collect job search preferences via freeform natural language."""
    console.print("\n[bold magenta] Job Search Preferences[/bold magenta]")
    console.print("[dim]Describe what you're looking for in plain language.[/dim]\n")

    prefs = None

    while True:
        if prefs is None:
            text = Prompt.ask(
                "[bold white]What kind of role are you looking for?[/bold white]\n"
                "[dim]  mention any of: role · location · remote/hybrid/onsite · industry · open to relocation · contract/full-time[/dim]\n"
                "[dim]  e.g. 'senior Python engineer in Berlin, hybrid, open to relocation'[/dim]\n"
                "[dim]  e.g. 'contract data scientist role, remote, fintech or healthtech'[/dim]\n "
            )
            if not text.strip():
                continue
            with console.status("[dim]Parsing preferences...[/dim]"):
                try:
                    prefs = parse_preferences(text)
                except Exception as e:
                    console.print(f"[red]Could not parse: {e}. Try again.[/red]")
                    continue
        else:
            # Retry — merge correction into existing prefs
            text = Prompt.ask(
                "[bold white]What would you like to add or change?[/bold white]\n"
                "[dim]  e.g. 'add marketing manager as role' or 'change location to New York'[/dim]\n "
            )
            if not text.strip():
                continue
            with console.status("[dim]Updating preferences...[/dim]"):
                try:
                    prefs = refine_preferences(prefs, text)
                except Exception as e:
                    console.print(f"[red]Could not update: {e}. Try again.[/red]")
                    continue

        console.print(
            Panel(_prefs_summary(prefs), title="Understood as", border_style="cyan")
        )

        if Confirm.ask("Does this look right?", default=True):
            return prefs


# ── Profile Verification ──────────────────────────────────────────────────────


def verify_profile(profile: CVProfile) -> CVProfile:
    """Show the parsed CVProfile and allow natural language edits."""
    while True:
        console.print(
            "\n[bold yellow]Agent's Understanding of Your Profile[/bold yellow]"
        )

        summary = (
            f"Experience: [bold]{profile.experience_level}[/bold] ({profile.years_experience} yrs)\n"
            f"Education: [bold]{profile.education_level}[/bold] in {profile.field_of_study}\n"
            f"Key Skills: {', '.join(profile.skills[:10])}{'...' if len(profile.skills) > 10 else ''}\n"
            f"Industries: {', '.join(profile.industries)}\n"
            f"Current Location: {profile.current_location or 'Not specified'}"
        )
        console.print(
            Panel(summary, title="Professional Persona", border_style="yellow")
        )

        if Confirm.ask("Does this look accurate?", default=True):
            return profile

        edit_text = Prompt.ask(
            "\n[bold white]What would you like to change?[/bold white]\n"
            "[dim]  e.g. 'change experience to senior' or 'add Kubernetes and Terraform to skills'[/dim]\n "
        )
        if not edit_text.strip():
            return profile

        with console.status("[dim]Applying changes...[/dim]"):
            try:
                profile = apply_profile_edit(profile, edit_text)
                console.print("[green]Changes applied.[/green]")
            except Exception as e:
                console.print(f"[red]Could not apply edit: {e}[/red]")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _job_url(job: JobRecord) -> str | None:
    if job.url:
        return job.url
    if job.source == "kaggle" and job.job_id:
        return f"https://www.linkedin.com/jobs/view/{job.job_id}"
    return None


def _run_pipeline(
    profile: CVProfile, prefs: JobSearchPreferences
) -> tuple[list[JobRecord], ReasoningReport]:
    """Retrieve → Rerank → Reason. Returns (top10 jobs, report)."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        t1 = progress.add_task("Searching for best job matches...", total=1)
        jobs_top20 = retrieve_jobs(profile, prefs)
        progress.update(t1, advance=1)

        t2 = progress.add_task("Refining results for quality...", total=1)
        jobs_top10 = rerank_jobs(profile, prefs, jobs_top20)
        progress.update(t2, advance=1)

        t3 = progress.add_task("Generating detailed match analysis...", total=1)
        report = analyze_job_matches(profile, jobs_top10)
        progress.update(t3, advance=1)

    return jobs_top10, report


def _display_results(jobs_top10: list[JobRecord], report: ReasoningReport) -> None:
    """Render tiered results: top 3 panels + compact table + roadmap."""
    console.print("\n[bold green]Match Analysis Complete![/bold green]\n")

    console.print("[bold cyan]Top 3 Recommended Matches[/bold cyan]")
    for i in range(min(3, len(jobs_top10))):
        job = jobs_top10[i]
        exp = next((e for e in report.job_explanations if e.job_id == job.job_id), None)
        border_style = "green" if i == 0 else "bright_blue"
        title = f"#{i + 1} {job.title} @ {job.company}"
        if exp:
            missing_str = (
                ", ".join(exp.missing_skills) if exp.missing_skills else "None"
            )
            content = f"{exp.match_reason}\n\n[bold red]Missing Skills:[/bold red] {missing_str}"
        else:
            content = "[italic]No detailed reasoning available.[/italic]"
        url = _job_url(job)
        if url:
            content += f"\n\n[dim][link={url}]Apply →  {url}[/link][/dim]"
        console.print(Panel(content, title=title, border_style=border_style))

    if len(jobs_top10) > 3:
        console.print("\n[bold cyan]Other Strong Contenders[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Title", width=28)
        table.add_column("Company", width=18)
        table.add_column("Key Match Reason", width=46)
        table.add_column("Link", width=12)
        for i in range(3, len(jobs_top10)):
            job = jobs_top10[i]
            exp = next(
                (e for e in report.job_explanations if e.job_id == job.job_id), None
            )
            reason_short = (
                (exp.match_reason[:75] + "...")
                if exp and len(exp.match_reason) > 75
                else (exp.match_reason if exp else "")
            )
            url = _job_url(job)
            if url:
                # show shortened URL — always visible even without terminal hyperlink support
                short = url.replace("https://", "").replace("www.", "")
                link_cell = (
                    f"[link={url}]{short[:30]}[/link]"
                    if len(short) <= 30
                    else f"[link={url}]{short[:27]}…[/link]"
                )
            else:
                link_cell = "—"
            table.add_row(str(i + 1), job.title, job.company, reason_short, link_cell)
        console.print(table)

    console.print("\n[bold yellow]Career Growth Roadmap[/bold yellow]")
    skills_text = (
        ", ".join(report.overall_missing_skills)
        or "Your profile is a great match for these roles!"
    )
    console.print(
        Panel(
            f"To increase your match rate, consider focusing on:\n\n[bold white]{skills_text}[/bold white]",
            border_style="yellow",
        )
    )


# ── Audit Trail ───────────────────────────────────────────────────────────────


def save_full_audit_trail(
    cv_path: Path,
    profile: CVProfile,
    prefs: JobSearchPreferences,
    report: ReasoningReport,
    jobs: list[JobRecord],
):
    cv_name = cv_path.stem
    filename = f"results_{cv_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    filepath = ITERATIONS_DIR / filename

    content = f"# Agent Session Audit Trail: {cv_name}\n\n"
    content += f"**Date:** {datetime.now().isoformat()}\n"
    content += f"**CV File:** `{cv_path.name}`\n\n"
    content += "## 1. User Persona (CVProfile)\n"
    content += f"```json\n{profile.model_dump_json(indent=2)}\n```\n\n"
    content += "## 2. Search Preferences\n"
    content += f"```json\n{prefs.model_dump_json(indent=2)}\n```\n\n"
    content += "## 3. Final Reasoning Report\n"
    content += f"### CV Summary\n{report.cv_summary}\n\n"
    content += f"### Recommendation\n{report.recommendation}\n\n"
    content += "### Job Matches (Full Details)\n"
    score_map = {j.job_id: j.score for j in jobs}
    sorted_explanations = sorted(
        report.job_explanations,
        key=lambda e: score_map.get(e.job_id, 0),
        reverse=True,
    )
    for i, job_exp in enumerate(sorted_explanations, 1):
        job_rec = next((j for j in jobs if j.job_id == job_exp.job_id), None)
        score = job_rec.score if job_rec else "N/A"
        content += f"#### {i}. {job_exp.title} @ {job_exp.company} (Score: {score})\n"
        if job_rec:
            url = _job_url(job_rec)
            if url:
                content += f"**Apply:** {url}\n\n"
            desc = job_rec.description
            desc_preview = (desc[:600] + "…") if len(desc) > 600 else desc
            content += f"**Description:**\n{desc_preview}\n\n"
        content += f"**Match Reason:** {job_exp.match_reason}\n\n"
        content += (
            f"**Missing Skills:** {', '.join(job_exp.missing_skills) or 'None'}\n\n"
        )
        content += "---\n\n"
    content += "## 4. Career Roadmap\n"
    content += f"**Top recommended skills to acquire:** {', '.join(report.overall_missing_skills)}\n"

    filepath.write_text(content, encoding="utf-8")
    return filepath


# ── Main ──────────────────────────────────────────────────────────────────────


def _pick_cv() -> Path:
    """Let user pick a CV from data/resumes/ or enter a custom path. Loops until valid."""
    resumes_dir = PROJECT_ROOT / "data" / "resumes"
    available = sorted(resumes_dir.glob("*.pdf")) if resumes_dir.exists() else []

    if available:
        console.print("\n[bold white]Available CVs in data/resumes/:[/bold white]")
        for i, p in enumerate(available, 1):
            console.print(f"  [cyan]{i}[/cyan]  {p.name}")
        console.print("  [cyan]0[/cyan]  Enter a custom path\n")

    while True:
        if available:
            choice = Prompt.ask(
                "[bold white]Select a CV[/bold white]", default="1"
            ).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(available):
                return available[int(choice) - 1]
            elif choice == "0":
                available = []  # drop to custom path from here on
                continue
            else:
                console.print(
                    "[red]Invalid — enter a number from the list above.[/red]"
                )
                continue

        cv_path_str = Prompt.ask(
            "[bold white]Enter the full path to your CV (PDF)[/bold white]"
        ).strip()
        cv_path = Path(cv_path_str)
        if cv_path.exists() and cv_path.suffix.lower() == ".pdf":
            return cv_path
        elif not cv_path.exists():
            console.print(f"[red]File not found: {cv_path_str}[/red]")
        else:
            console.print("[red]File must be a PDF.[/red]")


def run_agent():
    print_header()

    cv_path = _pick_cv()

    # Step 1: Parallel Onboarding — profile CV in background while collecting preferences
    with ThreadPoolExecutor(max_workers=2) as executor:

        def async_profile():
            raw_text = extract_text_from_pdf(cv_path)
            return profile_cv(raw_text)

        profile_future = executor.submit(async_profile)
        prefs = collect_preferences()

        # Wait for profiling to finish if still running
        with console.status("[bold cyan]Finishing CV analysis...[/bold cyan]"):
            try:
                profile = profile_future.result(timeout=120)
            except Exception as e:
                console.print(f"[red]Profiling failed: {e}[/red]")
                return

    # Step 2: Verification
    profile = verify_profile(profile)

    # Step 3: Pipeline + Refinement Loop
    iteration = 0
    jobs_top10, report = None, None

    while True:
        iteration += 1
        if iteration > 1:
            console.rule(f"[dim]Refinement #{iteration - 1}[/dim]")

        jobs_top10, report = _run_pipeline(profile, prefs)
        _display_results(jobs_top10, report)

        if not Confirm.ask("\nRefine your search?", default=False):
            break

        refine_text = Prompt.ask(
            "\n[bold white]What would you like to change?[/bold white]\n"
            "[dim]  e.g. 'focus more on remote roles' or 'try fintech instead'[/dim]\n "
        )
        if not refine_text.strip():
            break

        with console.status("[dim]Updating preferences...[/dim]"):
            try:
                prefs = refine_preferences(prefs, refine_text)
            except Exception as e:
                console.print(
                    f"[red]Could not parse: {e}. Keeping current preferences.[/red]"
                )
                continue

        summary = (
            f"Location: [bold]{prefs.target_location}[/bold]   "
            f"Remote: [bold]{prefs.remote_preference}[/bold]   "
            f"Roles: [bold]{', '.join(prefs.target_roles) or 'any'}[/bold]"
        )
        console.print(Panel(summary, title="Updated preferences", border_style="cyan"))

    # Step 4: Persistence
    audit_file = save_full_audit_trail(cv_path, profile, prefs, report, jobs_top10)
    console.print(
        f"\n[dim]Full report saved to: [bold white]{audit_file.name}[/bold white][/dim]"
    )


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Exiting...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]A critical error occurred: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
