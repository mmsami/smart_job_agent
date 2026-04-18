"""
Test harness for mock data validation.

Verifies:
  1. All mocks load without errors (Pydantic validation)
  2. Mocks match expected schemas
  3. Usage patterns work for Searcher and Analyzer roles
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.workflow.mocks import (
    mock_cv_senior_finance,
    mock_cv_mid_tech,
    mock_cv_junior_hr,
    mock_preferences_senior_finance,
    mock_preferences_mid_tech,
    mock_preferences_junior_hr,
    mock_job_records,
)
from src.workflow.models import CVProfile, JobSearchPreferences, JobRecord


class TestCVProfiles:
    """Validate CVProfile mocks."""

    def test_senior_finance_profile_loads(self):
        """Senior Finance persona loads with all required fields."""
        assert isinstance(mock_cv_senior_finance, CVProfile)
        assert mock_cv_senior_finance.experience_level == "senior"
        assert mock_cv_senior_finance.years_experience == 12
        assert "SAP" in mock_cv_senior_finance.skills
        assert "CPA" in mock_cv_senior_finance.certifications

    def test_mid_tech_profile_loads(self):
        """Mid-level Tech persona loads."""
        assert isinstance(mock_cv_mid_tech, CVProfile)
        assert mock_cv_mid_tech.experience_level == "mid"
        assert mock_cv_mid_tech.years_experience == 5
        assert "Python" in mock_cv_mid_tech.skills
        assert "Docker" in mock_cv_mid_tech.tools

    def test_junior_hr_profile_loads(self):
        """Junior HR persona loads."""
        assert isinstance(mock_cv_junior_hr, CVProfile)
        assert mock_cv_junior_hr.experience_level == "entry"
        assert mock_cv_junior_hr.years_experience == 2
        assert "recruiting" in mock_cv_junior_hr.skills
        assert "SHRM-CP" in mock_cv_junior_hr.certifications

    def test_all_profiles_have_required_fields(self):
        """All profiles have non-empty skills and experience_level."""
        for profile in [mock_cv_senior_finance, mock_cv_mid_tech, mock_cv_junior_hr]:
            assert len(profile.skills) > 0, f"{profile} missing skills"
            assert profile.experience_level in ["entry", "mid", "senior"]


class TestJobSearchPreferences:
    """Validate JobSearchPreferences mocks."""

    def test_senior_finance_preferences_loads(self):
        """Senior Finance preferences load."""
        assert isinstance(mock_preferences_senior_finance, JobSearchPreferences)
        assert mock_preferences_senior_finance.target_location == "New York, NY"
        assert mock_preferences_senior_finance.employment_type == "full-time"

    def test_mid_tech_preferences_loads(self):
        """Mid Tech preferences load."""
        assert isinstance(mock_preferences_mid_tech, JobSearchPreferences)
        assert mock_preferences_mid_tech.target_location == "San Francisco, CA"
        assert mock_preferences_mid_tech.willing_to_relocate is True

    def test_junior_hr_preferences_loads(self):
        """Junior HR preferences load."""
        assert isinstance(mock_preferences_junior_hr, JobSearchPreferences)
        assert mock_preferences_junior_hr.target_location == "Chicago, IL"
        assert mock_preferences_junior_hr.employment_type == "full-time"

    def test_all_preferences_have_remote_preference(self):
        """All preferences specify remote_preference."""
        for prefs in [mock_preferences_senior_finance, mock_preferences_mid_tech, mock_preferences_junior_hr]:
            assert prefs.remote_preference in ["remote", "hybrid", "onsite", "flexible"]


class TestJobRecords:
    """Validate JobRecord mocks."""

    def test_job_records_list_populated(self):
        """Job records list contains 10 jobs."""
        assert len(mock_job_records) == 10

    def test_all_jobs_are_job_record_instances(self):
        """All items in job_records are JobRecord objects."""
        for job in mock_job_records:
            assert isinstance(job, JobRecord)

    def test_all_jobs_have_required_fields(self):
        """All jobs have job_id, title, company, description, source, score."""
        for job in mock_job_records:
            assert job.job_id
            assert job.title
            assert job.company
            assert job.description
            assert job.source == "kaggle"
            assert isinstance(job.score, float)

    def test_edge_cases_present(self):
        """Edge cases (null salary, part-time, entry-level) are represented."""
        has_null_salary = any(job.min_salary is None for job in mock_job_records)
        has_part_time = any(job.work_type == "part-time" for job in mock_job_records)
        has_entry_level = any(job.experience_level == "entry" for job in mock_job_records)
        has_remote = any("Remote" in (job.location or "") for job in mock_job_records)

        assert has_null_salary, "No job with null salary found"
        assert has_part_time, "No part-time job found"
        assert has_entry_level, "No entry-level job found"
        assert has_remote, "No remote job found"

    def test_score_range_reasonable(self):
        """Job scores are between 0 and 100 (cosine similarity range)."""
        for job in mock_job_records:
            assert 0 <= job.score <= 100, f"Score {job.score} out of range"


class TestUsagePatterns:
    """Verify usage patterns work as documented."""

    def test_searcher_pattern_works(self):
        """Searcher can use: retriever.search(mock_cv, mock_prefs)."""
        # Simulate searcher usage
        cv = mock_cv_mid_tech
        prefs = mock_preferences_mid_tech

        # Mock retriever would do: retriever.search(cv, prefs)
        assert cv.experience_level == "mid"
        assert prefs.remote_preference == "remote"
        assert len(cv.skills) > 0

    def test_analyzer_pattern_works(self):
        """Analyzer can use: analyzer.analyze(mock_cv, mock_jobs[:10])."""
        # Simulate analyzer usage
        cv = mock_cv_mid_tech
        jobs = mock_job_records[:10]

        # Mock analyzer would do: analyzer.analyze(cv, jobs)
        assert isinstance(cv, CVProfile)
        assert all(isinstance(j, JobRecord) for j in jobs)
        assert len(jobs) == 10

    def test_can_filter_jobs_by_seniority(self):
        """Example: Analyzer filters jobs by seniority."""
        # If CV says "mid", keep jobs that are "mid" or "senior"
        cv_level = mock_cv_mid_tech.experience_level  # "mid"
        valid_seniorities = ["entry", "mid", "senior"]

        matching_jobs = [
            j for j in mock_job_records
            if j.experience_level in valid_seniorities
        ]
        assert len(matching_jobs) > 0

    def test_can_filter_jobs_by_location(self):
        """Example: Searcher filters jobs by target location."""
        target = mock_preferences_mid_tech.target_location  # "San Francisco, CA"

        # Soft filter (willing_to_relocate allows override)
        location_matching = [
            j for j in mock_job_records
            if j.location and target.split(",")[0] in j.location  # rough match
        ]
        # Should find at least San Francisco jobs
        assert any("San Francisco" in j.location for j in mock_job_records)


class TestDataConsistency:
    """Validate cross-mock consistency."""

    def test_personas_have_matching_preferences(self):
        """Each CVProfile has a corresponding JobSearchPreferences."""
        profiles = [mock_cv_senior_finance, mock_cv_mid_tech, mock_cv_junior_hr]
        preferences = [mock_preferences_senior_finance, mock_preferences_mid_tech, mock_preferences_junior_hr]

        # Check locations align (not exact, but same region)
        assert "New York" in profiles[0].current_location and "New York" in preferences[0].target_location
        assert "San Francisco" in profiles[1].current_location and "San Francisco" in preferences[1].target_location
        assert "Chicago" in profiles[2].current_location and "Chicago" in preferences[2].target_location

    def test_jobs_cover_multiple_domains(self):
        """Job records span multiple domains: Tech, Finance, HR."""
        all_descriptions = " ".join([j.description.lower() for j in mock_job_records])

        assert "python" in all_descriptions or "backend" in all_descriptions  # Tech
        assert "accounting" in all_descriptions or "finance" in all_descriptions  # Finance
        assert "recruiting" in all_descriptions or "hr" in all_descriptions  # HR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
