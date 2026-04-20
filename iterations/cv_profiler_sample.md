# CV Profiler — Sample Outputs

Canonical examples showing what `cv_profiler.py` (LOGIC_VERSION=v10) extracts from real CVs.
Raw per-run logs are gitignored — this file shows the reference outputs for team review.

---

## Perosona_Finance.pdf — Senior Finance / Accounting Persona

15+ years, multinational experience (Renewable Energy, Manufacturing, Consumer Goods).
Korean market specialist with Big 4 tools (SAP, HFM, Cognos).

```json
{
  "skills": [
    "Fp&a",
    "Sales & Cost Analysis",
    "Tax Accounting",
    "Closing",
    "Accounts Receivable",
    "Accounts Payable",
    "General Accounting",
    "Financial Reporting",
    "Budgeting",
    "Forecasting",
    "VAT Reconciliation",
    "Corporate Tax Return",
    "Internal Process Development",
    "Intercompany Reconciliation",
    "Subsidiary Liquidation",
    "Revenue Accounting",
    "Cash Forecasting",
    "Capex Management",
    "Data Pre-processing",
    "Data Analysis",
    "Data Visualization"
  ],
  "experience_level": "senior",
  "years_experience": 15,
  "current_location": "Seoul, Korea",
  "education_level": "bachelor",
  "field_of_study": "Law",
  "certifications": [
    "Computerized Accounting Qualification- Korean Association Of Certified Public Tax Account"
  ],
  "languages": [
    "Korean",
    "English",
    "Chinese"
  ],
  "job_titles_held": [
    "Accounting Manager",
    "Sr. Accountant",
    "Financial Analyst",
    "Accountant",
    "Tax Accountant",
    "Internship",
    "Year-end Tax Adjustment Review",
    "Regional Sales Manager"
  ],
  "industries": [
    "Renewable Energy",
    "Risk Management",
    "Manufacturing",
    "Consumer Goods",
    "Education",
    "Accounting And Tax Consulting"
  ],
  "domain_keywords": [
    "G&a",
    "VAT",
    "Cit",
    "Fss",
    "SOX",
    "Dml",
    "Ddl",
    "Kmm",
    "Svc",
    "Wht",
    "Tp Analysis",
    "Dso",
    "General Ledger",
    "Financial Statements"
  ],
  "tools": [
    "SAP",
    "Cognos",
    "HFM",
    "ERP",
    "SQL",
    "R"
  ]
}
```

**Notes:**
- `domain_keywords` correctly captures Korean-market-specific acronyms (FSS, CIT, WHT) that skills list omits
- `tools` cleanly separated from skills — SAP/HFM/Cognos correctly classified as tools not skills
- `field_of_study: "Law"` is correct — the persona has a law degree but pivoted to finance
- `industries` spans 6 sectors correctly reflecting multi-industry career

---

## Resume Sami.pdf — Senior Technical Product Manager / Business Analyst

12 years, cross-functional: BA, TPM, Data Engineering. Based in Mannheim, Germany.

```json
{
  "skills": [
    "Business Analysis",
    "Requirements Elicitation",
    "Requirements Documentation",
    "Business Process Analysis",
    "User Stories",
    "Acceptance Criteria",
    "Requirements Traceability",
    "Uat",
    "Product Discovery",
    "Feature Prioritization",
    "Roadmap Contribution",
    "Requirements Change Management",
    "Impact Analysis",
    "Delivery Coordination",
    "Story Mapping",
    "Cross-functional Coordination",
    "API Development",
    "API Testing",
    "System Architecture",
    "Workflow Automation"
  ],
  "experience_level": "senior",
  "years_experience": 12,
  "current_location": "Mannheim, Germany",
  "education_level": "master",
  "field_of_study": "Data Science",
  "certifications": [
    "Pmi-acp\u00ae",
    "Cspo\u00ae",
    "Csm\u00ae",
    "Tkp\u00ae",
    "Digital Product Management Specialization"
  ],
  "languages": [
    "Bengali",
    "English",
    "German"
  ],
  "job_titles_held": [
    "Working Student, S Factory Technical Support",
    "Working Student, IT Business Relationship",
    "Technical Product Manager",
    "Software Development Manager",
    "Data Engineer / Software Engineer",
    "System Engineer"
  ],
  "industries": [
    "IT Services",
    "Fintech",
    "Telecommunications",
    "Software Development"
  ],
  "domain_keywords": [
    "Frd",
    "Prd",
    "Babok",
    "Business Model Canvas",
    "SaaS",
    "Mvp",
    "Mfs",
    "ETL",
    "Iso 20000",
    "Sla",
    "Blockchain",
    "Web3",
    "Nft",
    "Llm Tooling",
    "Iot"
  ],
  "tools": [
    "Jira",
    "Confluence",
    "Visio",
    "Lucidchart",
    "Draw.io",
    "Miro",
    "Figma",
    "Git",
    "SQL",
    "Python",
    "Javascript",
    "Postman",
    "Mqtt",
    "Websocket",
    "Docker",
    "Hardhat",
    "Oracle Pl/sql",
    "Java",
    "Bash"
  ]
}
```

**Notes:**
- 19 tools correctly extracted — broad technical stack including blockchain (Hardhat) and IoT (MQTT)
- `domain_keywords` captures methodology acronyms (BABOK, FRD, PRD) that complement skills
- `current_location: "Mannheim, Germany"` — relevant for EU job matching against Arbeitnow data
- Certifications: PMI-ACP, CSPO, CSM correctly identified with unicode symbols preserved

---

## Key Observations (across all test CVs)

| Field | Reliability | Notes |
|---|---|---|
| `skills` | High | Consistent across runs; minor variation in granularity |
| `experience_level` | Medium | Template CVs with future dates confuse LLM (cv1: entry vs actual ~3yr) |
| `years_experience` | Medium | Same issue — LLM cannot verify temporal consistency of dates |
| `current_location` | High | Null when not present (cv3) — correct behaviour, not an error |
| `tools` | High | Clean separation from skills once v10 prompt landed |
| `domain_keywords` | High | Best field — captures acronyms and jargon skills list misses |
| `industries` | Medium | Sometimes omitted (cv3), sometimes over-inferred |
| `certifications` | High | Correctly empty when absent |
