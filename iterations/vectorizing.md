# DEV LOG: Vectorizing & FAISS Build

## [Build] Indexing the Data
Finally got the index built. 
- Loaded: 123,849 Kaggle + 957 Arbeitnow = 124,806 jobs.
- Chunking: Used the paragraph split on `\n\n`. For the huge paragraphs, used a 512-token window with 50 overlap. 
- Result: 192,514 chunks total.
- Model: `all-MiniLM-L6-v2` (384-dim). 
- Time: Took ~69 mins on the M1 Max. Not fast, but it's a one-time thing.
- Normalized all vectors to L2 so `IndexFlatIP` acts as cosine similarity.

Files saved:
- `faiss.index` (282MB)
- `docstore.json` (562MB)
Total ~844MB. Easy to share via Drive.

**Key Detail:** Added "Signal Boosting". Every chunk starts with `{title} at {company}. {skills}.`. Without this, a chunk that just says "must be a team player" has no meaning. Now the embedding knows exactly which job it belongs to.

---

## [Testing] Layer 1 Sanity & Stress Tests
Ran `test_retrieval.py` to make sure I didn't break the index.

### Results Summary

| Query | Top Job ID | Score | Verdict |
| :--- | :--- | :--- | :--- |
| Python soft eng remote | 3901673392 | 0.7022 | Correct domain |
| Senior accountant | 3885854685 | 0.7598 | Correct domain |
| B2B sales biz dev | 3901800114 | 0.7428 | Correct domain |
| Python + AWS + K8s | 3901942886 | 0.7700 | Strong match |
| CFO / Finance Dir | 3901348468 | 0.7084 | Right ballpark, but junior |
| React + Accessibility | 3890894276 | 0.6485 | Mixed results |
| Basket weaving | 3898175654 | 0.4646 | Irrelevant (Correct) |
| Quantum Cold Fusion | 3886842163 | 0.6011 | Confused ColdFusion dev |

### Observations
- **General Sanity:** Works. Python, Accounting, and Sales all return the right roles.
- **Specificity:** It's a bit shaky. For the CFO query, it returned a "Finance Manager". This is a classic Bi-encoder problem: it sees "Finance" and "Manager" and thinks it's close enough. This proves we really need the LLM Reranker to fix the exact ranking.
- **Negative Tests:** The scores dropped a lot (down to 0.45), which is good. It means the system knows when it doesn't have a match.
- **Cold Fusion Glitch:** The system matched "Cold Fusion" (physics) to "ColdFusion" (the coding language). Another great reason for the Reranker.

## Final Verdict
Index quality is **HIGH**. 
The retrieval is doing exactly what a Bi-encoder should do: surface a set of candidates that are topically relevant. The "mistakes" (ColdFusion dev, Finance Manager vs CFO) are exactly why we are adding the Reranking step.
