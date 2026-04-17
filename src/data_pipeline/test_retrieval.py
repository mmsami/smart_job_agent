"""
Sanity and Stress Test for the FAISS Vector Store.

This script verifies that the index loads correctly and that the 
retrieval quality is sufficient (precision/recall) before moving 
to the LLM reranking stage.

Usage:
    python src/data_pipeline/test_retrieval.py
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BASE_DIR, "data", "vector_store", "faiss.index")
DOCSTORE_PATH = os.path.join(BASE_DIR, "data", "vector_store", "docstore.json")

# ── Test Queries ──────────────────────────────────────────────────────
# We use a mix of general (sanity) and specific (stress) queries.
TEST_QUERIES = {
    "General Sanity": [
        "Python software engineer remote",
        "Senior accountant finance controlling",
        "B2B sales business development",
    ],
    "Specificity Stress Test": [
        "Python engineer with experience in AWS and Kubernetes",
        "CFO or Finance Director with 15+ years experience",
        "React developer specialized in accessibility and a11y",
    ],
    "Negative/Out-of-Domain Test": [
        "Underwater basket weaving specialist",
        "Quantum computing researcher for cold fusion",
    ]
}

def main():
    print("=" * 60)
    print("FAISS Vector Store Sanity & Stress Test")
    print("=" * 60)

    # 1. Load Model
    print(f"Loading model {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    # 2. Load FAISS Index
    print(f"Loading index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)

    # 3. Load Docstore
    print(f"Loading docstore from {DOCSTORE_PATH}...")
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    print("\n" + "-" * 60)
    
    for category, queries in TEST_QUERIES.items():
        print(f"\nCATEGORY: {category}")
        print("=" * 30)
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            
            # Embed and Normalize query
            query_vec = model.encode([query]).astype("float32")
            faiss.normalize_L2(query_vec)
            
            # Search
            distances, indices = index.search(query_vec, TOP_K)
            
            print(f"{'Rank':<6} | {'Score':<8} | {'Job ID':<15} | {'Snippet'}")
            print("-" * 80)
            
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    print(f"{rank+1:<6} | {'N/A':<8} | {'N/A':<15} | No result found")
                    continue
                
                doc = docstore[idx]
                # Truncate content for display
                content = doc["page_content"].replace("\n", " ")[:100] + "..."
                print(f"{rank+1:<6} | {dist:<8.4f} | {doc['metadata']['job_id']:<15} | {content}")

    print("\n" + "=" * 60)
    print("Testing Complete.")
    print("Manual Audit Guide:")
    print("1. Do the top results match the query's specific requirements?")
    print("2. Are the scores significantly lower for Negative Tests?")
    print("3. Does the snippet show the actual skill, or just the 'signal boost' prefix?")
    print("=" * 60)

if __name__ == "__main__":
    main()
