# DEV LOG: Vectorizing & FAISS Build
**Date:** 2026-04-12
**Objective:** Transform 124,806 raw job documents into a searchable vector space using a Bi-Encoder architecture.

---

## 1. The Indexing Pipeline
The goal was to create a high-performance index that is small enough to share via Google Drive but precise enough to surface relevant jobs.

### Technical Specifications
- **Embedding Model:** `all-MiniLM-L6-v2` (sentence-transformers).
- **Dimensions:** 384.
- **Algorithm:** FAISS `IndexFlatIP` (Inner Product) with L2-normalization. This effectively implements **Cosine Similarity**.
- **Build Time:** ~69 minutes on M1 Max for 192,514 vectors.

### Chunking Strategy: Paragraph-based with Fallback
Job postings vary wildly in length. We implemented a hybrid approach:
1. **Primary:** Split on double newlines (`\n\n`). This preserves the natural structure of "Responsibilities," "Requirements," and "Benefits."
2. **Fallback:** If a paragraph exceeds the model's token limit, we use a sliding word window of ~196 words with a **~38-word overlap**.
**Why overlap?** It prevents the model from losing context if a critical requirement is split exactly at the boundary of two chunks.

### Signal Boosting (Context Injection)
A major problem with chunking is that a chunk like *"Must be a team player and have 5 years experience"* is meaningless if it's separated from the job title.
**The Fix:** We prepend the job's core identity to **every single chunk**:
`{title} at {company}. {skill_labels}. {chunk_text}`
**Example:**
*"Senior Python Developer at SolarEdge. Information Technology. Must be a team player..."*
This ensures that the embedding for every chunk is "anchored" to the role and company, drastically improving retrieval precision.

---

## 2. Validation: Layer 1 Sanity Tests
We ran `tests/data_pipeline/test_retrieval.py` to verify the index before building the LLM pipeline. This isolated the embedding model's performance from the CV parser's performance.

### Results Summary

| Query | Top Result | Score | Verdict | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| "Python soft eng remote" | Python Developer | 0.7022 | ✅ Pass | Correct domain and work type. |
| "Senior accountant" | Senior Accountant | 0.7598 | ✅ Pass | High confidence match. |
| "B2B sales biz dev" | Account Executive | 0.7428 | ✅ Pass | Correct semantic mapping. |
| "Python + AWS + K8s" | Cloud Engineer | 0.7700 | ✅ Pass | Correctly handled multi-attribute query. |
| "CFO / Finance Dir" | Finance Manager | 0.7084 | ⚠️ Partial | Right domain, but too junior (Bi-encoder limitation). |
| "React + Accessibility" | Front-end Dev | 0.6485 | ⚠️ Partial | Found React, but accessibility signal was weak. |
| "Basket weaving" | Random Job | 0.4646 | ✅ Pass | Correctly identified as irrelevant (Low score). |
| "Quantum Cold Fusion" | ColdFusion Dev | 0.6011 | ❌ Fail | Confused physics "Cold Fusion" with the coding language. |

---

## 3. Key Insights & "The Reranker Justification"
The sanity tests proved that the Bi-encoder (`all-MiniLM-L6-v2`) is an excellent **coarse filter** but a poor **fine-grained ranker**.

### The "Semantic Blur" Problem
The "CFO vs Finance Manager" and "Cold Fusion" errors are classic Bi-encoder failures. The model maps "CFO" and "Finance Manager" to a similar region in vector space because they share the "Finance" and "Management" concepts. However, in the real world, these are very different roles.

**Conclusion:** We cannot rely on FAISS alone for the Top 10. 
**The Solution:**
1. **FAISS (Coarse):** Retrieve the Top 20 candidates (High Recall).
2. **LLM Reranker (Fine):** Use Gemma 4 31B to read the full text of those 20 jobs and re-rank them based on exact requirements (High Precision).

## 4. Final Assets (v1 — built with 512-token settings, superseded)
- **`faiss.index` (282MB):** The compressed vector space.
- **`docstore.json` (562MB):** The mapping of vector IDs to full job metadata.
- **Total Size:** ~844MB.
- **Status:** Shared via Google Drive. Superseded by v2 rebuild below.

---

## 5. Iteration 2: Token Limit Fix + Model-Named Outputs (2026-04-19)

### Bugs Found via Code Review

**Bug 1 — Token limit mismatch:** `MAX_TOKENS = 512` was used as both the token threshold and the word-slice size. `all-MiniLM-L6-v2` has a hard limit of **256 tokens**. Any chunk with more than ~197 words was silently truncated by the tokenizer — the tail of the chunk was never embedded. The sliding window was taking 512 words ≈ 665 tokens, more than 2× the model's capacity.

**Bug 2 — Prefix overflow:** The signal-boosting prefix (`"Title at Company. Skills. "`, ~15–25 words) was prepended to every chunk after chunking. A paragraph that fit within the 256-token budget on its own would overflow after the prefix was added. Both the paragraph-fit check and the sliding window needed to reserve space for the prefix.

### Fix Applied (`build_vector_store_minilm.py`)

```
MODEL_MAX_TOKENS = 256         # hard limit for all-MiniLM-L6-v2
MAX_WORDS = int(256 / 1.3)    # ≈ 196 words → safe chunk body
OVERLAP_WORDS = int(50 / 1.3) # ≈ 38 words overlap
```

- `split_into_chunks` accepts `max_words` parameter (not hardcoded global)
- Paragraph-fit check uses word count (`len(para.split()) <= max_words`) — consistent with sliding window
- `build_chunks` computes `prefix_words = len(prefix.split())` per doc and passes `effective_max = MAX_WORDS - prefix_words` (floor 50) — both branches now respect the prefix budget

### Naming Convention Change
Script and output files now include model short-name to prevent collisions if the embedding model is swapped:
- Script: `build_vector_store_minilm.py`
- Index: `data/vector_store/faiss_minilm.index`
- Docstore: `data/vector_store/docstore_minilm.json`

### Planned Rebuild
v2 index to be built locally. Expected chunk count slightly higher (more chunks per long description since window is smaller). Embedding quality expected to improve — model now sees complete chunk content instead of silently truncated text.

---

## 6. v2 Rebuild Results (2026-04-20)

### Build Stats

| Metric | v1 (buggy) | v2 (fixed) |
|--------|-----------|-----------|
| Token window | 512 words (~665 tokens) | 196 words (~255 tokens) |
| Total vectors | 192,514 | **471,671** |
| Build time | ~69 min | ~109 min (6547.9s) |
| Index files | `faiss.index` / `docstore.json` | `faiss_minilm.index` / `docstore_minilm.json` |

471,671 vectors vs 192,514 — **2.45× more chunks**. As expected: smaller windows mean long descriptions get properly split rather than silently truncated.

### Layer 1 Sanity Test — v1 vs v2

Same 8 queries run against `tests/data_pipeline/test_retrieval.py` (paths updated to minilm files).

| Query | v1 Top Result | v1 Score | v2 Top Result | v2 Score | Delta | Verdict |
|-------|--------------|----------|--------------|----------|-------|---------|
| Python soft eng remote | Python Developer | 0.7022 | Python Developer | 0.7029 | +0.0007 | ✅ Same |
| Senior accountant | Senior Accountant | 0.7598 | Senior Accountant (CFS) | 0.7611 | +0.0013 | ✅ Same |
| B2B sales biz dev | Account Executive | 0.7428 | VP Business Dev - B2B Sales | 0.7653 | +0.0225 | ✅ Better title |
| Python + AWS + K8s | Cloud Engineer | 0.7700 | Python Dev with AWS & DevOps | 0.7700 | 0.0000 | ✅ Same |
| CFO / Finance Dir | Finance Manager ⚠️ | 0.7084 | Financial Director | 0.7115 | +0.0031 | ✅ **Fixed** |
| React + Accessibility | Front-end Dev ⚠️ | 0.6485 | AEM React Dev | 0.6485 | 0.0000 | ➡️ Same score |
| Basket weaving | Random job | 0.4646 | Tubing Fabricator | 0.4646 | 0.0000 | ✅ Correctly low |
| Quantum Cold Fusion | ColdFusion Dev ❌ | 0.6011 | Post-Doc Quantum Computing | 0.5908 | −0.0103 | ✅ **Fixed** |

### Key Wins

**CFO / Finance Director (was ⚠️ Partial → now ✅):**
v1 returned "Finance Manager" — wrong seniority. v2 top 5 returns: Financial Director → Finance Manager (Report to CFO) → Chief Financial Officer → Director & Head of Finance → Director FP&A. The seniority signals ("CFO", "Director", "15+ years") that distinguished executive roles from manager roles were previously sitting in the truncated tail of job description chunks. Now they're embedded.

**Quantum Cold Fusion (was ❌ Fail → now ✅):**
v1 returned a ColdFusion (programming language) developer job — the model confused "Cold Fusion" the physics concept with the Adobe ColdFusion language. v2 returns "Post Doctoral Fellow, Quantum Computing" at 0.5908. Two things explain the fix: (1) disambiguating context (quantum physics framing) was in truncated portions of the description; (2) score dropped appropriately — the query is still out-of-domain so confidence correctly fell.

**B2B Sales:**
v1 returned "Account Executive" (generic). v2 returns "Vice President, Business Development - B2B Sales" — exact title match. Higher confidence (0.7653 vs 0.7428).

### What Didn't Change

- **React + Accessibility**: Same score (0.6485), same rank 1. Rank 2 is now an Accessibility Analyst role rather than a generic front-end dev — marginal improvement in diversity. This remains a known Bi-encoder weakness: "accessibility" is a weak signal relative to "React". The reranker will fix this at retrieval time.
- **Basket weaving / Quantum Cold Fusion scores**: Both correctly in the 0.46–0.59 range — below the domain-relevant results (~0.70+). The score gap validates the index is discriminating signal from noise.

### Conclusion

The token fix had a real impact on seniority and domain disambiguation. The index is ready for the retriever layer (`job_search.py`). The two previously failing queries now pass. Remaining weaknesses (accessibility signal, seniority precision at the CFO level) are expected Bi-encoder limitations and are addressed by the LLM reranker in Phase 2b.
