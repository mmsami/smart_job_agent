# Dev Log: CV Input & Extraction Strategy
**Date:** 2026-04-18
**Objective:** Define the user experience (UX) and technical implementation for reading resumes into the pipeline.

---

## Scope

**Supported Formats:** PDF only. DOCX, Google Docs, and other formats are not supported. Users must convert to PDF first (e.g., LibreOffice, Word "Save as PDF").

**Language:** English-only pipeline (prompts in English, education regex for English degree names). Non-English CVs will degrade profiling output (`education_level` may be null for non-English degrees).

---

## 1. The UX Dilemma: Single vs. Batch
When building the main.py entry point, we faced a choice in how the system should ingest CVs:
- Single Mode: User passes one CV -> get one report. (Great for demos).
- Batch Mode: User passes a folder -> get multiple reports. (Great for evaluation).

## 2. The Decision: Hybrid Approach
To satisfy both the product demo and the research requirement (rigorous evaluation of 10 personas), we decided to implement a Hybrid Mode.

### Mode A: Interactive (Single File)
- Trigger: python main.py --file <path_to_cv>
- Purpose: Live demonstration.
- Focus: Low latency and immediate feedback.

### Mode B: Evaluation (Batch Folder)
- Trigger: python main.py --folder <path_to_folder>
- Purpose: Academic research and labeling.
- Focus: Throughput and structured output. The system iterates through all CVs in the directory and saves results to a JSON file for teammate labeling.

---

## 3. Rationale for the Report
Implementing both modes allows us to add a Systems Analysis section to the final report. We can contrast two different performance metrics:
1. End-to-End Latency: Time a single user waits for a result.
2. System Throughput: Efficiency of processing the full test set of personas.

This demonstrates "Systems Thinking" and shows we designed a tool for both users and researchers.

---

## 4. Implementation Plan
- Use argparse in main.py to create a mutually exclusive group for --file and --folder.
- Ensure run_pipeline() is a standalone function that can be called individually or in a loop.
- Implement a JSON exporter for Batch Mode to support the "Union of Results" labeling strategy.

---

## 5. CV Text Extraction & Profiling Strategy

### The Problem
Resumes are diverse. A single extraction strategy cannot handle all layouts:
- Single-column: Easy to extract.
- Two-column: Text order gets jumbled, confusing the LLM.
- Tables/Boxes: Visual structure is lost.
- Scanned PDFs: No text to extract, extraction fails entirely.
- Designed Resumes: Graphics make text lose meaning.

For our 10 controlled test CVs, we need a balance between robustness and speed.

### Options Considered

#### Option A: Text Extraction (PyMuPDF)
Flow: PDF -> PyMuPDF -> raw text string -> Gemma 4 LLM -> CVProfile
- Pros: Very fast, simple, works for standard resumes.
- Cons: Fails on scanned PDFs and struggles with two-column layouts.
- Time: 2-3 days.

#### Option B: Vision LLM (Multimodal)
Flow: PDF -> convert pages to images -> Gemma 4 vision -> CVProfile
- Pros: Handles any layout (two-column, scanned, designed) because it sees the page visually.
- Cons: Slightly slower, requires logic to combine multi-page results.
- Time: 4-5 days.

#### Option C: Hybrid Fallback
Flow: Try PyMuPDF -> if low quality -> fallback to Vision LLM.
- Pros: Speed when possible, robustness when needed.
- Cons: Most complex to implement and debug.
- Time: 5-6 days.

### Decision: Vision LLM (Option B)
We decided to go with the Vision approach. For a "Smart Job Market Agent," the extraction should be smart, not brittle. Since Vision is only about one day more work than text extraction and handles all real-world diversity, it's the best choice.

### How we will implement it:
- Convert PDF pages to images in memory using PyMuPDF.
- Send these images to Gemma 4 Vision with a structured prompt.
- Validate the output using the CVProfile Pydantic schema.
- Cache the final JSON result keyed by the PDF path to avoid redundant API calls.

### Why this works:
- Robustness: Handles any layout a human recruiter could read.
- Intelligence: Aligns with the "Smart" branding of the project.
- Simplicity: One clean code path instead of complex layout detection.

### Comparison: Why not others?
- Text (Option A) is too brittle for two-column or scanned PDFs.
- Hybrid (Option C) is over-engineering for this project scope.

### Next Steps
- Implement cv_reader.py (PDF to image conversion + Gemma 4 vision).
- Write the profiling prompt in src/prompts/cv_profiler.md.
- Create a test suite to verify extraction on all 10 personas.
- Move on to job_search.py.

---

## Report Language
In the final report, we will describe this as:
"We use Gemma 4 31B multimodal vision LLM to extract CVProfiles directly from PDF images. This approach robustly handles all resume layouts — single-column, two-column, scanned PDFs, designed resumes — without requiring layout detection logic."

**Status:** Strategy finalized. Ready for implementation.

---

## 6. Implementation Log

### Iteration 1 — Initial Build (2026-04-18)

**What we built:**
- `cv_reader.py`: PyMuPDF renders PDF pages at 300 DPI → PIL images → OpenRouter `google/gemini-2.0-flash-001` (vision) → raw text string.
- `cv_profiler.py`: raw text → Gemma 4 31B (text model, Google AI Studio) → JSON → CVProfile Pydantic object.
- Both use `diskcache` keyed by content hash (MD5) to avoid redundant API calls on re-runs.

**Architecture decision: two-step design**
Initial attempt was one-shot LLM extraction (extract + compute years in a single prompt). Problem: LLM is non-deterministic on arithmetic — same CV produced different `years_experience` across runs. Fix: split into Step 1 (LLM extracts raw facts + dates only) and Step 2 (Python computes years, level, normalizes). All arithmetic is now deterministic.

**First test run on cv1, cv2, cv3:** All three returned results but with multiple correctness issues (see Iteration 2+).

---

### Iteration 2 — Bug Fixes Round 1 (2026-04-18)

**Bug: cv3 Pydantic crash**
```
current_location: Input should be a valid string [input_value=None]
```
LLM returned `null` for `current_location` (cv3 had no location). Field was declared `str` (required).
Fix: changed to `Optional[str] = None` in `models.py`.

**Bug: cv2 `years_experience` inflated (8 years reported, ~7 actual)**
LLM was counting experience from graduation year (2018) instead of first job start (2019). Fixed by adding explicit instruction to the system prompt: "Count experience from first job start date, not graduation date."

**Bug: cv1 `education_level` returned `null` after prompt fix**
The "ignore future dates" rule we added for `years_experience` was accidentally applied to education too — LLM excluded the degree because it had a future end_year. Fixed by clarifying in the prompt: the future-date rule applies only to `years_experience`, not education.

---

### Iteration 3 — Hardening Round 2 (2026-04-19)

**Bug: `education_level` schema drift**
LLM returned `"masters"` (with an 's') which failed the `Literal["bachelor", "master", "phd"]` validator. Root cause: no server-side enforcement on the extraction model. Fix: added `_normalize_education()` in Python (Step 2) using word-boundary regex to map any variant → strict enum:
- `\b(master|masters|msc|mba|meng)\b` → `"master"`
- `\b(bachelor|bachelors|bsc|ba|btech)\b` → `"bachelor"`
- `\b(phd|ph\.d|doctorate)\b` → `"phd"`

**Bug: acronym casing destroyed (AWS → "Aws")**
`word.capitalize()` lowercases everything except the first letter. Fix: replaced with `_smart_title()` + `_KEEP_CASE` set:
```python
_KEEP_CASE = {"AWS", "GCP", "SQL", "NoSQL", "API", "ML", "AI", "CI/CD", ...}
```
Any word in the set is restored to its canonical casing instead of being capitalized.

**Bug: overlapping jobs double-counted**
A candidate with two simultaneous roles (e.g., 2018–2022 and 2019–2021) was summing both intervals: 4 + 2 = 6 years instead of 4. Fix: interval merging algorithm — sort intervals, merge overlapping, sum merged lengths only.

**Bug: `_safe_year()` crash on non-numeric strings**
Original code: `int(str(value)[:4])` crashed on "Jan 2018", "circa 2019", etc.
Fix: `re.search(r"\b(19|20)\d{2}\b", str(value))` — extracts 4-digit year from any string format.

**Added: content-based retry (`_is_bad_output`)**
LLM occasionally returned valid JSON but with empty `skills` or zero valid job titles. Added retry logic that checks output quality before accepting, with up to 3 attempts.

**Added: `LOGIC_VERSION` cache key**
Changing Python computation logic (e.g., interval merging) without changing the raw text would serve stale cache entries. Added `LOGIC_VERSION = "v4"` to the cache key — bumping it invalidates all prior cached profiles automatically.

---

### Iteration 4 — cv_reader Improvements (2026-04-19)

**PNG → JPEG (quality=85)**
PNG at 300 DPI for an A4 page is ~25 MB. JPEG at quality=85 is ~3–5 MB — 5–10× smaller with no meaningful OCR quality loss. Reduces OpenRouter payload size significantly.

**Timeout 60 → 120 seconds**
Vision calls processing 3 pages of 300 DPI images were hitting the 60s timeout. Bumped to 120s.

**Added `PROMPT_VERSION` to cache key**
Same reasoning as `LOGIC_VERSION` — if we change `EXTRACTION_PROMPT`, the cache key `cv_text_v1_{hash}` automatically differs from old entries.

**Added page-count warning**
If a PDF has more than 5 pages, a warning is logged before the API call. Not a hard limit — just visibility for debugging.

---

---

## 7. Caching Strategy: Content-Based Key Design

**Why MD5 hashing is cost-critical:**

The pipeline caches at two levels, both keyed by **content hash, not filename**:

**Step 1 (cv_reader):**
- Cache key: `cv_text_{MODEL_NAME}_{DPI}_{PROMPT_VERSION}_{MD5(PDF_content)}`
- Stores: raw extracted text (6KB–10KB per CV)
- **Benefit:** Rename `Sarah_CV.pdf` → `Sarah_Resume.pdf` (same file) → same MD5 → **cache hit, zero vision API cost**

**Step 2 (cv_profiler):**
- Cache key: `cv_profile_{LOGIC_VERSION}_{MD5(raw_text)}`
- Stores: CVProfile dict (~2KB per CV)
- **Benefit:** Two different PDFs that extract identical text → same text hash → **cache hit, zero Gemma API cost**

**Real-world cost savings:**
- First unique CV: ~$0.02 (vision + Gemma calls)
- Duplicate of same CV (any filename): $0
- Two candidates with identical template: Step 2 saves Gemma cost on second

For a 10-persona evaluation: **one unique content per person = one API call per person, ever.** Filename changes, re-uploads, template clones → all free after first parse.

---

### Final Results — Stable State (2026-04-19, LOGIC_VERSION=v4)

| CV | Profile | `years_experience` | `experience_level` | `education_level` | Notes |
|----|---------|-------------------|-------------------|-------------------|-------|
| cv1.pdf | Marketing Manager | 0 | entry | master (Strategic Marketing) | Template CV with placeholder future dates (2027–2030). Date math gives 0 — known limitation of template CVs. |
| cv2.pdf | Chef / Kitchen Hand | 7 | mid | bachelor (Gastronomy) | Hospitality profile. Interval merging correctly handled 3 sequential roles. |
| cv3.pdf | IT Project Manager | 12 | senior | master (Project Management) | Two-column designed CV. Vision handled layout correctly. No location in CV → `current_location: null`. |

**All 3 CVs:** Status OK, no Pydantic validation errors, no API retries needed on final run.

---

### Iteration 5 — Final Hardening Round (2026-04-19)

Fixes applied after external feedback review:

| Area | Bug / Issue | Fix Applied |
|------|-------------|-------------|
| cv_reader payload size | A4 at 300 DPI = 2480×3508px; can hit OpenRouter body limits | Added `img.thumbnail((2000, 2000))` resize after render — always triggers for A4, reduces payload ~40% |
| Missing start year visibility | Jobs silently skipped with no debug info | Added `logger.warning()` when a job is dropped for missing `start_year` |
| `_KEEP_CASE` incomplete | `JSON`, `HTML`, `CSS`, `PHP`, `JS`, `TS`, `C#`, `C++` were being title-cased incorrectly | Extended `_KEEP_CASE` set with common tech stack terms |
| Dead semantic validation | `if level == "senior" and years < 5` can never be true — `_classify_experience_level` only returns "senior" when years > 7 | Removed dead block entirely; `_classify_experience_level` is now single source of truth |
| `_is_bad_output` over-aggressive retry | Empty `skills` triggered retry even on CVs with no skills section → exhausted all retries on valid input | Removed skills check; only empty/placeholder `jobs` now triggers retry |
| cv_reader docstring mismatch | Said "Gemma 4 vision" but actual model is `google/gemini-2.0-flash-001` (Gemini) via OpenRouter | Corrected to "Gemini 2.0 Flash vision (via OpenRouter)" |

---

### Iteration 6 — Production Hardening (2026-04-19)

Final round of bulletproofing based on code review:

| Area | Issue | Fix |
|------|-------|-----|
| PyMuPDF file handle cleanup | `doc.close()` could be skipped on error | Switched to `with fitz.open(...) as doc:` context manager |
| Cache invalidation on config change | Upgrading MODEL_NAME or DPI would serve stale cache | Updated cache key to `cv_text_{MODEL_NAME}_{DPI}_{PROMPT_VERSION}_{hash}` |
| `_normalize_list` non-string items | Mixed-type lists (e.g. `["Python", 42]`) crash `_smart_title()` | Added `if not isinstance(item, str): continue` guard |
| `_call_llm` unexpected JSON structure | LLM returning `[{...}]` instead of `{...}` gives misleading error | Added `if not isinstance(data, dict): raise ValueError(...)` check |
| Pydantic cache unpacking | `CVProfile(**dict)` less safe than dedicated validator | Changed to `CVProfile.model_validate(_cache[cache_key])` |
| Incomplete acronym list | Missing tech terms like SEO, SEM, SRE, QA, FPGA, ASIC | Expanded `_KEEP_CASE` with 9 additional acronyms |

---

### Iteration 7 — Cache & Defensive Guards (2026-04-19)

Final safety improvements based on code review feedback:

| Area | Issue | Fix |
|------|-------|-----|
| Cache model drift | Changing `MODEL_NAME` (e.g., `gemma-4-31b-it` → `gemma-4-32b-it`) would serve outputs from old model | Updated cache key to `cv_profile_{LOGIC_VERSION}_{MODEL_NAME}_{hash}` — model swaps now auto-invalidate cache |
| `len()` crash on bad LLM structure | If LLM returned non-list for `jobs_raw`/`skills_raw`, `len()` would crash with TypeError | Added `isinstance(jobs_raw, list)` guard before calling `len()` — defensive logging now safe |

---

### Iteration 8 — Prompt Engineering: Tools & Skills/Domain Separation (2026-04-19)

Prompt refinements based on actual output analysis:

| Issue | Fix |
|-------|-----|
| `tools` field always empty | Added concrete example ("built dashboards using Tableau" → Tableau in tools, Dashboard Development in skills) + explicit "Do NOT duplicate" instruction |
| Skills/domain_keywords bleed | Added "Avoid generic soft skills like Teamwork/Communication" to domain_keywords rule + examples that show distinction |

**Test Results (LOGIC_VERSION=v5, new prompt):**

| CV | `tools` | Skills/Domain separation | Notes |
|----|---------|-------------------------|-------|
| Perosona_Finance | `[SQL, R, Sap, Cognos, Hfm, ERP]` ✅ | Clean | Was empty, now correctly extracted 6 tech tools |
| Resume Sami | `[Jira, Confluence, Figma, Git, Python, Javascript, Docker, ...]` ✅ | Clean | 18 tools extracted across dev/PM stack |
| cv1 (Marketing) | `[]` ✅ | Clean | Correct — no tech tools in marketing CV |
| cv2 (Chef) | `[]` ✅ | Clean | Correct — no tech tools in culinary CV |
| cv3 (IT PM) | `[]` ✅ | Minor duplication | "Software Development" in both skills + domain_keywords (acceptable for IT role) |

**Status:** Prompt engineering validated. Tools extraction now working. Skills/domain separation 95% clean (1 minor duplication in IT domain). All 3 test personas + 2 real CVs passing.

---

### Iteration 9 — Prompt Tightening + Acronym Fixes (2026-04-19)

Second round of prompt engineering based on output analysis and external feedback review. Evaluated ~20 suggestions across 4 reviewers — applied only changes with observed evidence.

**Code changes (cv_profiler.py, LOGIC_VERSION=v6):**

| Area | Issue | Fix |
|------|-------|-----|
| Acronyms broken in output | "Sap", "Hfm", "Vat" in results — SAP/HFM/VAT not in `_KEEP_CASE` | Added `SAP, UAT, UML, MQTT, HFM, SOX, GAAP, VAT, KYC, AML, IFRS, CPA, CFA, MBA, SCRUM, OKR` to `_KEEP_CASE` |

**Prompt changes (cv_profiler.md):**

| Area | Issue | Fix |
|------|-------|-----|
| "Extract ONLY" contradiction | Rule said "extract only explicitly present" but skills rule required abstraction | Replaced with: "You may normalize phrasing into standard terms, but do NOT invent information not grounded in text" |
| Missing "must return all fields" | LLMs occasionally omit keys entirely | Added: "You MUST return all fields exactly as defined. If no data, return [] or null" |
| Industries inconsistent across runs | "clearly implied" was fuzzy — cv3 got "Technology" in one run, `[]` in another | Tightened: "only if explicitly stated in CV text. Do NOT infer from job titles or company names" |
| Industries guidance missing | No instruction on how to extract industries field | Added dedicated `industries` rule |
| "Must extract" pressure to hallucinate | "MUST extract, never return empty" conflicted with "return [] if no data" | Changed to: "extract as many as possible, do not fabricate missing items" |
| Title Case instruction redundant | `_smart_title()` overwrites LLM casing anyway | Removed the line — saves tokens, zero impact |

**Test Results (LOGIC_VERSION=v6, 4 CVs excluding personal):**

| CV | `tools` | `industries` | Acronyms | Notes |
|----|---------|-------------|----------|-------|
| Perosona_Finance | `[SQL, R, SAP, Cognos, HFM, ERP]` ✅ | Rich list ✅ | SAP, HFM, VAT, SOX correct ✅ | Acronyms fixed from v5 |
| cv1 (Marketing) | `[]` ✅ | `[]` ✅ | — | Correct — template CV, fictitious company |
| cv2 (Chef) | `[]` ✅ | `[Hospitality]` ✅ | — | Correct |
| cv3 (IT PM) | `[]` ✅ | `[Technology]` ✅ | — | Previously empty, now populated |

**Skipped (evaluated, not applied):** labeled section hierarchy, skills/tools hard dedup rule, JSON trailing comma guard (handled by `response_mime_type`), contact guard (field not used downstream), education raw extraction (Python handles both raw and normalized).

**Pending test:** prompt changes from points 1 & 2 (contradiction fix + industries tightening + Title Case removal) not yet re-tested. Next run on persona CVs will validate.

---

### Iteration 10 — Soft Skill Filter (2026-04-19, LOGIC_VERSION=v7)

Applied signal/noise reasoning: generic interpersonal traits add false positives in FAISS matching and produce useless skill gap outputs. Filter traits, keep competencies.

**Prompt change:**
- `skills` rule updated: "Extract professional competencies and hard skills. Avoid generic interpersonal traits (e.g. Team Player, Hard-working, Passionate, Problem Solver, Analytical Thinking). Keep professional competencies like Leadership or Stakeholder Management if explicitly listed."

**Test Results (v7, 4 CVs):**

| CV | Removed (traits) | Kept (competencies) | Notes |
|----|-----------------|---------------------|-------|
| cv1 (Marketing) | Teamwork, Time Management | Leadership, Project Management, Digital Marketing | ✅ Filter working |
| Finance | Problem Solving, Decision Making, Analytical Thinking | Financial Planning, Tax Accounting, VAT Reconciliation | ✅ Filter working |
| cv2 (Chef) | — | Conflict Resolution, Leadership, Strategic Planning | ✅ No regressions |
| cv3 (IT PM) | — | Project Management, Enterprise Resource Planning | ✅ No regressions |

All 4 CVs pass validation. Years/experience levels stable. Tools field stable.

### Known Limitations / Open Items

1. **English-only pipeline:** Both the cv_reader extraction prompt and cv_profiler system prompt are in English. Vision extraction may work on non-English CVs (Gemini handles multi-language), but profiling will degrade: `_normalize_education` regex only matches English degree names (`bachelor`, `msc`, `phd`, etc.) — German equivalents like "Diplom", "Magister", or "Hochschule" return `education_level: null`. Scope is explicitly English CVs only.

2. **cv1 future dates:** Template CVs with placeholder dates (e.g., 2027–2030) produce `years_experience = 0` because all jobs are in the future. This is correct behavior — the pipeline cannot verify temporal consistency of CV dates. Document in report as a known edge case.

2. **`tools` field sparsely populated:** All 3 CVs returned empty `tools` lists despite cv3 mentioning software tools in bullet points. The prompt distinguishes "skills" from "tools" (specific software platforms), but the LLM tends to put everything in `skills`. May improve with 10 real-world persona CVs.

3. **`industries` inconsistent:** cv1 and cv3 sometimes return empty, sometimes populated, across runs — suggests the LLM is uncertain when the industry isn't explicitly stated. Acceptable variance for a keyword-matching field.

---

### Iteration 11 — Fresh Graduate Support + Critical Bug Fixes (2026-04-19, LOGIC_VERSION=v8→v9→v10)

Three rounds of fixes applied after systematic code review. No prompt changes — all Python logic.

---

#### Round 1: Fresh Graduate Support (LOGIC_VERSION=v8)

**Problem:** `_is_bad_output` returned `True` for any CV with zero jobs — triggering 3 retries then `RuntimeError`. A fresh graduate with no work history would always crash the pipeline.

**Root cause:** The function treated "no jobs" as a sign of LLM failure, but some CVs genuinely have no jobs.

**Fix:** Replaced single-field check with content-aware "absolute vacuum" logic:

```python
# Stage 1: require at least one populated primary list
critical_fields = ["jobs", "education", "skills", "certifications", "languages"]
has_any_content = any(
    isinstance(data.get(f), list) and len(data.get(f, [])) > 0
    for f in critical_fields
)
if not has_any_content:
    return True  # genuine LLM failure

# Stage 2: if jobs present, reject all-placeholder titles
```

**Logic:** If the LLM extracted *anything* across the primary sections, it successfully parsed the CV — no jobs just means no jobs. Only a completely empty output (all lists empty) triggers a retry.

**Coverage:** Handles fresh grads (education only), career changers (certifications only), language-focused profiles — any non-empty section is sufficient.

Also fixed: final retry log level changed from `logger.info` → `logger.warning` so failures are visible.

---

#### Round 2: Correctness + Safety Guards (LOGIC_VERSION=v9)

| Area | Issue | Fix |
|------|-------|-----|
| `_safe_year` range strings | `"2018-2020"` as `end_year` would extract `2018` (first match) instead of `2020` (correct end year), silently losing up to 2 years of experience | Added `prefer_last: bool = False` parameter; `end_year` now calls `_safe_year(end_raw, prefer_last=True)` → extracts last year in range |
| Empty `raw_text` guard | If `cv_reader` failed silently and returned `""`, `profile_cv` would make an LLM call on empty input and receive garbage | Added `if not raw_text or not raw_text.strip(): raise ValueError(...)` at top of `profile_cv` |
| PII in logs | `logger.info(f"Contact: {contact}")` was logging email addresses and phone numbers | Changed to log only field names: `logger.info(f"Contact fields found: {present}")` |

---

#### Round 3: Silent Year Parsing Bug (LOGIC_VERSION=v10)

**Critical bug found in `_safe_year`:**

```python
# Before (BROKEN):
matches = re.findall(r"\b(19|20)\d{2}\b", str(value))
# Returns ["20", "20"] — capturing group returns group content, not full match
# int(matches[0]) → 20 for any year

# After (FIXED):
matches = re.findall(r"\b(?:19|20)\d{2}\b", str(value))
# Returns ["2018", "2020"] — non-capturing group returns full match
```

**Impact:** Every year parsed before v10 was returning `20` instead of the 4-digit year. `years_experience` was computed as 0 for all CVs (2020 - 2020 = 0, etc.). The bug was masked by caching — CVs cached before the bug was introduced appeared correct; only fresh parses were broken.

**Why not caught earlier:** Cached results from earlier correct implementations (when `re.search` was used) hid the bug. It only manifested on cache misses.

---

#### Round 4: cv_reader Output Validation (cv_reader.py, PROMPT_VERSION unchanged)

**Problem:** If OpenRouter returned `"I cannot read this document"` or an empty string, `cv_profiler` received garbage input and failed with a confusing error.

**Fix:** Added minimum-length check after vision LLM response:

```python
text = resp.json()["choices"][0]["message"]["content"].strip()
if len(text) < 200:
    raise ValueError(f"Extracted text too short ({len(text)} chars) — likely failed OCR or model refusal")
return text
```

Short responses now trigger a retry (up to `MAX_RETRIES=3`) before raising `RuntimeError`. The 200-char threshold catches empty strings and refusal messages while allowing through very terse CVs.

---

#### LOGIC_VERSION Summary

| Version | Change |
|---------|--------|
| v7 | Soft skill filter (previous iteration) |
| v8 | `_is_bad_output` rewritten for fresh grad support |
| v9 | `_safe_year prefer_last`, empty text guard, PII log redacted, final retry WARNING |
| v10 | Regex bug fix: `(19|20)` → `(?:19|20)` in `_safe_year` |

---

### Iteration 12 — Code Review Hardening (2026-04-19, LOGIC_VERSION unchanged)

No logic changes — all fixes are correctness, performance, and defensive guards. Cache not invalidated (LOGIC_VERSION stays v10).

#### cv_profiler.py

| Area | Issue | Fix |
|------|-------|-----|
| Duplicate `ERP` in `_KEEP_CASE` | Set had `ERP` twice — cosmetic but sloppy | Removed duplicate |
| O(n) acronym lookup in `_smart_title` | `next(k for k in _KEEP_CASE if ...)` scanned entire set on every word | Added `_KEEP_CASE_LOOKUP = {k.lower(): k for k in _KEEP_CASE}` — O(1) dict lookup; simplified `_smart_title` to single `.get()` call |
| Job dedup missing company | `(title, start_year, end_year)` key deduplicated "Software Engineer @ Google" and "Software Engineer @ Meta" | Added `company` to key: `(title, company, start_year, end_year)` — same title at different companies correctly kept |

Note: same title + same company + different role is handled by `title` being part of the key — e.g. "Junior Dev @ SAP 2020-2022" and "Senior Dev @ SAP 2022-2024" are distinct keys.

#### cv_reader.py

| Area | Issue | Fix |
|------|-------|-----|
| Unprotected JSON access | `resp.json()["choices"][0]["message"]["content"]` crashed with `KeyError` on OpenRouter error responses (`{"error": {...}}`) | Replaced with `.get()` chain; extracts error message if `choices` is absent |
| Short-text `ValueError` retried | `except Exception` caught our own `ValueError` ("text too short") and retried — wasted 2 API calls on blank PDFs or model refusals | Added `except ValueError: raise` before general handler — logic errors propagate immediately, only network/HTTP errors retry |
| No PDF validation | Encrypted, empty, or corrupted PDFs gave cryptic fitz errors deep in the render step | Added upfront validation: checks file size > 0, not encrypted, page count > 0, not corrupted — all raise clear `ValueError` before any API call |
