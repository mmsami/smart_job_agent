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
2. **Fallback:** If a paragraph exceeds 512 tokens, we use a sliding window of 512 tokens with a **50-token overlap**.
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

## 4. Final Assets
- **`faiss.index` (282MB):** The compressed vector space.
- **`docstore.json` (562MB):** The mapping of vector IDs to full job metadata.
- **Total Size:** ~844MB.
- **Status:** Verified, Healthy, and Shared via Google Drive.
