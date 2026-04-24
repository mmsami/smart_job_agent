# Dev Log: FAISS Job Retrieval (job_search.py)
**Date:** 2026-04-24
**Objective:** Implement semantic job retrieval — embed CVProfile + JobSearchPreferences, search FAISS index, return top-20 ranked jobs.

---

## Approach

Single combined embedding for CV + preferences → cosine similarity search against 471k job chunks → return top-20.

**Why combined?** The query vector needs to capture both *who the candidate is* (skills, experience, domain) and *what they want* (location, work type, target roles). Combining them into one vector keeps the retrieval step simple and avoids score fusion at this stage.

**Serialization strategy:** Each CVProfile field is prefixed with a label (`"Skills: Python, AWS"`) and joined into a plain sentence. Same for preferences. This mirrors how job chunks are stored (prefixed with title/company/skills) — matching representation on both sides improves cosine alignment.

---

## Iteration 1 — Teammate's Initial Code (2026-04-24)

Teammate implemented the file from the delegation guide. Structure and design were sound — the right functions, right approach. But the code would not run.

### What was okay
- `serialize_cv_profile()` and `serialize_preferences()` — correct field coverage and format
- `embed_profile_and_preferences()` — correct combined text approach
- `search_jobs()` logic — correct cosine similarity loop structure
- Using `all-MiniLM-L6-v2` — matches index embedding model

### What was broken

**Bug 1: Wrong FAISS loader**
```python
# Teammate wrote (BROKEN):
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/vector_store", embeddings, allow_dangerous_deserialization=True)
```
The index was built as raw `faiss_minilm.index` + `docstore_minilm.json` — not LangChain FAISS format. `FAISS.load_local()` expects its own serialization format and crashes on this index.

**Bug 2: Undefined variables at search call**
```python
# Teammate wrote (BROKEN):
results = search_jobs(
    query_embedding=query_embedding,
    index=index,       # undefined
    job_texts=job_texts,     # undefined
    job_metadata=job_metadata,  # undefined
    top_k=20
)
```
The vectorstore was loaded but never unpacked. `index`, `job_texts`, and `job_metadata` were never extracted — they simply didn't exist.

**Bug 3: Duplicate model loading**
Both `HuggingFaceEmbeddings` (LangChain) and `SentenceTransformer` were instantiated. Two separate model loads, neither used correctly.

**Bug 4: Wrong file paths**
`FAISS.load_local("data/vector_store", ...)` — relative path with no filename. Actual files are `faiss_minilm.index` and `docstore_minilm.json`.

---

## Iteration 2 — Core Fix (2026-04-24)

Replaced the broken loader with correct raw FAISS loading. Extracted `index`, `job_texts`, `job_metadata` from the docstore.

```python
# Fixed:
import faiss, json

index = faiss.read_index(str(INDEX_PATH))
with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
    docstore = json.load(f)

job_texts = [d["page_content"] for d in docstore]
job_metadata = [d["metadata"] for d in docstore]
```

Removed LangChain imports entirely. Removed duplicate `HuggingFaceEmbeddings` — kept `SentenceTransformer` only. Fixed paths to use absolute `Path(__file__)` references pointing to `faiss_minilm.index` and `docstore_minilm.json`.

**Result:** Script ran for the first time. Top-20 results returned correctly.

---

## Iteration 3 — Structural Fixes from Feedback Round 1 (2026-04-24)

Evaluated ~10 feedback points. Applied 5 valid ones. Rejected 5 as over-scoped or incorrect.

**Applied:**

| Fix | Why |
|-----|-----|
| `write_results()` → writes markdown table to `RESULTS_FILE` | Docstring said results logged to file — nothing was written |
| `if __name__ == "__main__"` guard | Module-level execution meant index loaded on every import — test file confirmed this |
| Removed unused globals `cv`, `prefs`, `jobs` | Dead code, misleading |
| Removed unused `normalize()` function | FAISS's `normalize_L2` used instead; dead code |
| Added `logger.info(f"Search complete: {len(results)} results returned")` | Docstring promised logging; only index load was logged |

**Rejected (recurring invalid feedback):**
- *"FAISS normalization mismatch"* — Invalid. `build_vector_store_minilm.py:263` calls `faiss.normalize_L2(vectors)` with `IndexFlatIP`. Cosine is guaranteed by construction. Refuted 3 times across feedback rounds.
- *"Handle -1 indices"* — Noise. 471k vectors, top_k=20. Never fires in practice.
- *"File I/O try/except"* — Over-scoped for research script.
- *"Post-retrieval location/remote filter"* — Phase 2c reranker's job, not retrieval's.

---

## Iteration 4 — Correctness Guards from Feedback Round 2 (2026-04-24)

Three small correctness fixes:

**Fix 1: float32 at encode time**
```python
# Before:
return model.encode(combined_text)  # returns Tensor, .astype() fails

# After:
return model.encode(combined_text, convert_to_numpy=True).astype("float32")
```
Linter flagged `Tensor` has no `.astype()`. `convert_to_numpy=True` returns ndarray directly.

**Fix 2: Correct None check**
```python
# Before:
if job_metadata:  # falsy on empty list [] — wrong

# After:
if job_metadata is not None:  # explicit None check
```

**Fix 3: Assert docstore/index length match**
```python
assert len(job_texts) == index.ntotal, (
    f"Docstore/index mismatch: {len(job_texts):,} entries vs {index.ntotal:,} vectors"
)
```
Catches mismatched files at load time with a clear message rather than a silent `IndexError` later.

---

## Iteration 5 — Linting Cleanup (2026-04-24)

All E402 "module level import not at top" errors resolved by restructuring import order.

| Issue | Fix |
|-------|-----|
| `import sys, os` on one line (E401) | Split to two lines |
| `from datetime import datetime` — unused | Removed |
| `from src.workflow.mocks import ...` after module-level code (E402) | Moved to top with try/except fallback |
| `import faiss`, `import numpy`, `from sentence_transformers import ...` after module-level code (E402) | Moved to top block |
| Extra alignment spaces `job_texts    =` (E221) | Removed |
| `List`, `Dict` from `typing` | Replaced with built-in `list`, `dict` (Python 3.9+) |
| `# type: ignore[call-arg]` on `index.search()` | FAISS stubs incorrectly expect output array params; two-return form is correct Python API |

---

## Final Test Results (16/16 passing)

```
tests/workflow/test_job_search.py — 16 passed in 21s
```

| Test Group | Coverage |
|------------|----------|
| `TestIndexLoad` (4) | index.ntotal > 0, docstore length == ntotal, page_content non-empty, metadata has title/company/source |
| `TestEmbedding` (4) | returns ndarray, shape (384,), dtype float32, different personas produce different vectors |
| `TestSearchJobs` (8) | 20 results, has score + job_description, scores in [-1,1], scores descending, mid-tech ≥5 tech titles, finance vs tech sets diverge, job_metadata=None path works |

**Sanity check:** Mid-tech persona (5 yrs Python, SF, remote) returns software engineer / backend / data engineer roles in top results. Finance persona returns distinct result set with <15/20 overlap with tech set.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Raw `faiss.read_index()` not LangChain | Index built with raw faiss — LangChain format incompatible |
| `convert_to_numpy=True` on encode | SentenceTransformer returns Tensor; `.astype()` requires ndarray |
| `assert` on docstore/index length | Silent `IndexError` is worse than a clear assertion message at load time |
| `if job_metadata is not None:` | Empty list `[]` is falsy but valid — explicit None check needed |
| `# type: ignore[call-arg]` on `index.search()` | Stubs wrong; two-return form is correct documented API |
| Combined CV + preferences embedding | Single query vector; no score fusion needed at retrieval stage |

---

## Next Step

Phase 2c: `reranker.py` — Gemma 4 31B scores the top-20 in a single batch call, returns top-10 sorted by relevance.
