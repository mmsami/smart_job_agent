"""
Tests for job_search.py — FAISS retrieval with mock CV profiles.

Verifies:
  1. Index + docstore load correctly (lengths match)
  2. CV + preferences embed to correct shape and float32 dtype
  3. search_jobs returns 20 results with expected fields
  4. Scores are in valid cosine similarity range
  5. search_jobs works with job_metadata=None
  6. Mid-tech persona returns software/tech roles (sanity check)
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.workflow.mocks import (
    mock_cv_mid_tech,
    mock_cv_senior_finance,
    mock_preferences_mid_tech,
    mock_preferences_senior_finance,
)
from src.workflow.job_search import (
    embed_profile_and_preferences,
    search_jobs,
    index,
    job_texts,
    job_metadata,
)


class TestIndexLoad:
    def test_index_has_vectors(self):
        assert index.ntotal > 0, "FAISS index is empty"

    def test_docstore_length_matches_index(self):
        assert len(job_texts) == index.ntotal
        assert len(job_metadata) == index.ntotal

    def test_docstore_entries_have_page_content(self):
        assert all(isinstance(t, str) and len(t) > 0 for t in job_texts[:10])

    def test_metadata_has_expected_fields(self):
        required = {"title", "company", "source"}
        for meta in job_metadata[:10]:
            assert required.issubset(meta.keys()), f"Missing fields in metadata: {meta.keys()}"


class TestEmbedding:
    def test_embedding_returns_array(self):
        vec = embed_profile_and_preferences(mock_cv_mid_tech, mock_preferences_mid_tech)
        assert isinstance(vec, np.ndarray)

    def test_embedding_correct_dim(self):
        vec = embed_profile_and_preferences(mock_cv_mid_tech, mock_preferences_mid_tech)
        assert vec.shape == (384,), f"Expected (384,), got {vec.shape}"

    def test_embedding_is_float32(self):
        vec = embed_profile_and_preferences(mock_cv_mid_tech, mock_preferences_mid_tech)
        assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"

    def test_different_profiles_differ(self):
        vec1 = embed_profile_and_preferences(mock_cv_mid_tech, mock_preferences_mid_tech)
        vec2 = embed_profile_and_preferences(mock_cv_senior_finance, mock_preferences_senior_finance)
        assert not np.allclose(vec1, vec2), "Different personas should produce different embeddings"


class TestSearchJobs:
    def setup_method(self):
        self.query = embed_profile_and_preferences(mock_cv_mid_tech, mock_preferences_mid_tech)
        self.results = search_jobs(
            query_embedding=self.query,
            index=index,
            job_texts=job_texts,
            job_metadata=job_metadata,
            top_k=20,
        )

    def test_returns_20_results(self):
        assert len(self.results) == 20

    def test_results_have_score(self):
        for r in self.results:
            assert "score" in r

    def test_results_have_job_description(self):
        for r in self.results:
            assert "job_description" in r
            assert len(r["job_description"]) > 0

    def test_scores_in_valid_range(self):
        for r in self.results:
            assert -1.0 <= r["score"] <= 1.0, f"Score out of range: {r['score']}"

    def test_scores_descending(self):
        scores = [r["score"] for r in self.results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_mid_tech_returns_tech_roles(self):
        titles = [r.get("title", "").lower() for r in self.results]
        tech_keywords = {"engineer", "developer", "software", "data", "python", "backend", "frontend", "ml", "analyst"}
        hits = sum(1 for t in titles if any(kw in t for kw in tech_keywords))
        assert hits >= 5, f"Expected ≥5 tech roles for mid-tech persona, got {hits}. Titles: {titles}"

    def test_search_without_metadata(self):
        results = search_jobs(
            query_embedding=self.query,
            index=index,
            job_texts=job_texts,
            job_metadata=None,
            top_k=20,
        )
        assert len(results) == 20
        for r in results:
            assert "job_description" in r
            assert "score" in r

    def test_finance_persona_differs_from_tech(self):
        finance_query = embed_profile_and_preferences(mock_cv_senior_finance, mock_preferences_senior_finance)
        finance_results = search_jobs(
            query_embedding=finance_query,
            index=index,
            job_texts=job_texts,
            job_metadata=job_metadata,
            top_k=20,
        )
        tech_ids = {r.get("job_id") for r in self.results}
        finance_ids = {r.get("job_id") for r in finance_results}
        overlap = len(tech_ids & finance_ids)
        assert overlap < 15, f"Too much overlap between tech and finance results: {overlap}/20"
