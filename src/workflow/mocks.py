"""
Mock data for contract validation across pipeline steps.

Provides stable, predictable test data to enable parallel development:
  - CVProfile objects (3 personas: Senior Finance, Mid Tech, Junior HR)
  - JobSearchPreferences objects (matching personas)
  - JobRecord objects (10 realistic Kaggle samples with edge cases)

All mocks are strictly typed to models.py schemas.
"""

from .models import CVProfile, JobSearchPreferences, JobRecord


# ────────────────────────────────────────────────────────────────────────────
# CV PROFILES (Factual data extracted from CV)
# ────────────────────────────────────────────────────────────────────────────

mock_cv_senior_finance = CVProfile(
    skills=[
        "SAP", "SQL", "R", "accounting", "tax", "FP&A", "closing",
        "accounts receivable", "accounts payable", "general ledger", "journal entries"
    ],
    experience_level="senior",
    years_experience=12,
    current_location="New York, NY",
    education_level="master",
    field_of_study="Accounting",
    certifications=["CPA", "MBA"],
    languages=["English", "Korean"],
    job_titles_held=["Accounting Manager", "Finance Director", "Senior Accountant"],
    industries=["Finance", "Consulting", "Fortune 500"],
    domain_keywords=["GAAP", "IFRS", "SOX", "reconciliation", "audit"],
    tools=["NetSuite", "Oracle", "QuickBooks", "Excel"]
)

mock_cv_mid_tech = CVProfile(
    skills=[
        "Python", "React", "JavaScript", "TypeScript", "AWS", "Docker", "PostgreSQL",
        "REST APIs", "Git", "testing", "debugging"
    ],
    experience_level="mid",
    years_experience=5,
    current_location="San Francisco, CA",
    education_level="bachelor",
    field_of_study="Computer Science",
    certifications=[],
    languages=["English"],
    job_titles_held=["Software Engineer", "Full-Stack Developer"],
    industries=["Tech", "SaaS", "Startups"],
    domain_keywords=["microservices", "CI/CD", "agile"],
    tools=["VS Code", "GitHub", "Docker", "Kubernetes", "Jenkins"]
)

mock_cv_junior_hr = CVProfile(
    skills=[
        "recruiting", "HR", "payroll", "employee relations", "onboarding",
        "benefits administration", "compliance", "communication"
    ],
    experience_level="entry",
    years_experience=2,
    current_location="Chicago, IL",
    education_level="bachelor",
    field_of_study="Human Resources",
    certifications=["SHRM-CP"],
    languages=["English", "Spanish"],
    job_titles_held=["HR Coordinator", "Recruiting Coordinator"],
    industries=["HR", "Tech", "Consulting"],
    domain_keywords=["EEOC", "ATS", "talent acquisition"],
    tools=["Workday", "LinkedIn Recruiter", "Excel", "Google Workspace"]
)


# ────────────────────────────────────────────────────────────────────────────
# JOB SEARCH PREFERENCES (What user wants, not from CV)
# ────────────────────────────────────────────────────────────────────────────

mock_preferences_senior_finance = JobSearchPreferences(
    target_location="New York, NY",
    work_type="full-time",
    employment_type="full-time",
    willing_to_relocate=False,
    target_roles=["Finance Director", "Controller", "Senior Accountant"],
    industry_preference=["Finance", "Banking", "Consulting"],
    remote_preference="hybrid"
)

mock_preferences_mid_tech = JobSearchPreferences(
    target_location="San Francisco, CA",
    work_type="full-time",
    employment_type="full-time",
    willing_to_relocate=True,
    target_roles=["Senior Software Engineer", "Full-Stack Engineer"],
    industry_preference=["Tech", "SaaS"],
    remote_preference="remote"
)

mock_preferences_junior_hr = JobSearchPreferences(
    target_location="Chicago, IL",
    work_type="full-time",
    employment_type="full-time",
    willing_to_relocate=False,
    target_roles=["HR Specialist", "Recruiting Coordinator", "HR Business Partner"],
    industry_preference=["HR", "Tech", "Healthcare"],
    remote_preference="hybrid"
)


# ────────────────────────────────────────────────────────────────────────────
# JOB RECORDS (Realistic Kaggle samples with edge cases)
# ────────────────────────────────────────────────────────────────────────────

mock_job_records = [
    # 1. Senior Finance role (high seniority, matches senior_finance profile)
    JobRecord(
        job_id="k001",
        title="Senior Accountant - FP&A",
        company="Goldman Sachs",
        description=(
            "We are seeking a Senior Accountant with 8+ years of experience. "
            "Requirements: SAP experience, GAAP/IFRS knowledge, strong SQL skills. "
            "You will lead closing processes, manage reconciliations, and mentor junior staff."
        ),
        location="New York, NY",
        experience_level="senior",
        work_type="full-time",
        min_salary=140000,
        max_salary=180000,
        url="https://example.com/jobs/k001",
        skill_labels="accounting, SAP, SQL, GAAP, FP&A",
        source="kaggle",
        score=92.5
    ),

    # 2. Mid-level Tech role (matches mid_tech profile)
    JobRecord(
        job_id="k002",
        title="Senior Backend Engineer - Python",
        company="Stripe",
        description=(
            "Join our backend team building payment infrastructure. "
            "5+ years Python/Go experience required. Strong REST API design, PostgreSQL, AWS. "
            "Microservices architecture, CI/CD pipelines. Remote-friendly."
        ),
        location="San Francisco, CA",
        experience_level="mid",
        work_type="full-time",
        min_salary=160000,
        max_salary=220000,
        url="https://example.com/jobs/k002",
        skill_labels="Python, PostgreSQL, AWS, REST API, Docker",
        source="kaggle",
        score=88.3
    ),

    # 3. Junior HR role (matches junior_hr profile)
    JobRecord(
        job_id="k003",
        title="HR Coordinator",
        company="TechCorp Inc",
        description=(
            "Entry-level HR role supporting recruiting and onboarding. "
            "Responsibilities: posting jobs, screening resumes, coordinating interviews, "
            "new hire onboarding, benefits admin. Workday experience a plus."
        ),
        location="Chicago, IL",
        experience_level="entry",
        work_type="full-time",
        min_salary=45000,
        max_salary=55000,
        url="https://example.com/jobs/k003",
        skill_labels="recruiting, onboarding, HR, benefits",
        source="kaggle",
        score=85.7
    ),

    # 4. Finance role with null salary (edge case)
    JobRecord(
        job_id="k004",
        title="Finance Manager",
        company="Accenture",
        description=(
            "Manage financial planning and analysis for client projects. "
            "8-10 years finance experience. Excel expert, SQL preferred. "
            "Travel required (20-30%)."
        ),
        location="New York, NY",
        experience_level="senior",
        work_type="full-time",
        min_salary=None,
        max_salary=None,
        url="https://example.com/jobs/k004",
        skill_labels="FP&A, finance, Excel, SQL",
        source="kaggle",
        score=78.2
    ),

    # 5. Remote tech role (matches mid_tech remote preference)
    JobRecord(
        job_id="k005",
        title="Full-Stack Engineer (Remote)",
        company="GitLab",
        description=(
            "Build features for our collaboration platform. 4-6 years experience. "
            "React, Node.js, PostgreSQL. Fully remote, async-first culture. "
            "We value clear communication and autonomy."
        ),
        location="Remote",
        experience_level="mid",
        work_type="full-time",
        min_salary=150000,
        max_salary=200000,
        url="https://example.com/jobs/k005",
        skill_labels="React, Node.js, PostgreSQL, JavaScript",
        source="kaggle",
        score=84.1
    ),

    # 6. Part-time / hybrid role (edge case: part-time)
    JobRecord(
        job_id="k006",
        title="HR Specialist - Part Time",
        company="Local Health Clinic",
        description=(
            "Part-time HR support (20 hrs/week). Recruiting, onboarding, compliance. "
            "Ideal for someone balancing other commitments. Hybrid (2 days in office)."
        ),
        location="Chicago, IL",
        experience_level="entry",
        work_type="part-time",
        min_salary=32000,
        max_salary=40000,
        url="https://example.com/jobs/k006",
        skill_labels="HR, recruiting, compliance",
        source="kaggle",
        score=72.5
    ),

    # 7. Entry-level tech role (edge case: mismatch for mid_tech, but good for testing filters)
    JobRecord(
        job_id="k007",
        title="Junior Frontend Developer",
        company="Startup XYZ",
        description=(
            "Entry-level role for recent graduates. Learn React, JavaScript, CSS. "
            "Mentorship from senior engineers. Competitive startup culture."
        ),
        location="San Francisco, CA",
        experience_level="entry",
        work_type="full-time",
        min_salary=80000,
        max_salary=100000,
        url="https://example.com/jobs/k007",
        skill_labels="React, JavaScript, HTML, CSS",
        source="kaggle",
        score=65.3
    ),

    # 8. Director-level finance role (too senior for mid-level profiles)
    JobRecord(
        job_id="k008",
        title="Director of Finance",
        company="JPMorgan Chase",
        description=(
            "Lead finance operations across multiple business units. "
            "15+ years experience required. CPA and MBA preferred. "
            "Executive leadership role reporting to CFO."
        ),
        location="New York, NY",
        experience_level="senior",
        work_type="full-time",
        min_salary=220000,
        max_salary=300000,
        url="https://example.com/jobs/k008",
        skill_labels="finance, leadership, accounting, strategy",
        source="kaggle",
        score=82.6
    ),

    # 9. Tech role in different location (geographic mismatch for mid_tech)
    JobRecord(
        job_id="k009",
        title="Backend Engineer - AWS",
        company="Amazon",
        description=(
            "AWS-focused backend role. 4-7 years experience. Python, Java, or Go. "
            "Microservices, distributed systems. Relocation support available."
        ),
        location="Seattle, WA",
        experience_level="mid",
        work_type="full-time",
        min_salary=170000,
        max_salary=240000,
        url="https://example.com/jobs/k009",
        skill_labels="AWS, Python, microservices, backend",
        source="kaggle",
        score=87.9
    ),

    # 10. Hybrid accounting role (matches senior_finance, domain + seniority match)
    JobRecord(
        job_id="k010",
        title="Senior Accounting Manager - Hybrid",
        company="Deloitte",
        description=(
            "Join our audit/accounting practice. 10+ years accounting experience. "
            "GAAP, IFRS, SOX knowledge. Hybrid: 3 days office, 2 remote. "
            "Lead teams, manage client relationships, mentor junior staff."
        ),
        location="New York, NY",
        experience_level="senior",
        work_type="full-time",
        min_salary=130000,
        max_salary=170000,
        url="https://example.com/jobs/k010",
        skill_labels="accounting, GAAP, audit, management",
        source="kaggle",
        score=91.2
    ),
]
