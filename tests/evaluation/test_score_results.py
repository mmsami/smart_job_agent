import textwrap
from pathlib import Path

import pytest

from src.evaluation.score_results import (
    bootstrap_ci,
    parse_md_labels,
    precision_at_k,
    wilcoxon_p,
)


# ── parse_md_labels ───────────────────────────────────────────────────────────

def _write_md(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sheet.md"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_parse_md_labels_basic(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 | 1 | 5 | Engineer | Acme |
        | 2 | 0 | 3 | Manager | Corp |
        | 3 | 1 | 2 | Analyst | Biz |
    """)
    relevant, quality = parse_md_labels(md)
    assert relevant == [1, 0, 1]
    assert quality == [5, 3, 2]


def test_parse_md_labels_partial_fill(tmp_path):
    # only relevant filled, quality blank
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 | 1 |   | Engineer | Acme |
        | 2 | 0 |   | Manager  | Corp |
    """)
    relevant, quality = parse_md_labels(md)
    assert relevant == [1, 0]
    assert quality == []


def test_parse_md_labels_quality_only(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 |   | 4 | Engineer | Acme |
        | 2 |   | 2 | Manager  | Corp |
    """)
    relevant, quality = parse_md_labels(md)
    assert relevant == []
    assert quality == [4, 2]


def test_parse_md_labels_skips_header_and_separator(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 | 1 | 5 | Engineer | Acme |
    """)
    relevant, quality = parse_md_labels(md)
    assert len(relevant) == 1
    assert len(quality) == 1


def test_parse_md_labels_rejects_out_of_range_quality(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 | 1 | 6 | Engineer | Acme |
        | 2 | 0 | 0 | Manager  | Corp |
    """)
    relevant, quality = parse_md_labels(md)
    assert relevant == [1, 0]
    assert quality == []


def test_parse_md_labels_nonexistent_file(tmp_path):
    relevant, quality = parse_md_labels(tmp_path / "missing.md")
    assert relevant == []
    assert quality == []


def test_parse_md_labels_empty_file(tmp_path):
    md = tmp_path / "empty.md"
    md.write_text("", encoding="utf-8")
    relevant, quality = parse_md_labels(md)
    assert relevant == []
    assert quality == []


# ── precision_at_k ────────────────────────────────────────────────────────────

def test_precision_at_k_perfect():
    assert precision_at_k([1] * 10) == 1.0


def test_precision_at_k_zero():
    assert precision_at_k([0] * 10) == 0.0


def test_precision_at_k_mixed():
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert precision_at_k(labels) == 0.5


def test_precision_at_k_uses_first_k():
    # extra labels beyond k should be ignored
    labels = [1] * 10 + [0] * 5
    assert precision_at_k(labels, k=10) == 1.0


def test_precision_at_k_too_few_labels():
    assert precision_at_k([1, 0, 1], k=10) is None


def test_precision_at_k_exactly_k():
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    result = precision_at_k(labels, k=10)
    assert result == 0.5


# ── bootstrap_ci ──────────────────────────────────────────────────────────────

def test_bootstrap_ci_returns_tuple():
    values = [0.5, 0.6, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6]
    ci = bootstrap_ci(values)
    assert ci is not None
    lo, hi = ci
    assert lo <= hi


def test_bootstrap_ci_bounds_contain_mean():
    values = [0.5, 0.6, 0.7, 0.4, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6]
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_uniform_values():
    values = [0.5] * 10
    lo, hi = bootstrap_ci(values)
    assert lo == pytest.approx(0.5, abs=0.01)
    assert hi == pytest.approx(0.5, abs=0.01)


def test_bootstrap_ci_single_value_returns_none():
    assert bootstrap_ci([0.5]) is None


def test_bootstrap_ci_empty_returns_none():
    assert bootstrap_ci([]) is None


def test_bootstrap_ci_deterministic():
    values = [0.3, 0.5, 0.7, 0.4, 0.6, 0.5, 0.4, 0.7, 0.6, 0.3]
    ci1 = bootstrap_ci(values)
    ci2 = bootstrap_ci(values)
    assert ci1 == ci2  # seeded rng


# ── wilcoxon_p ────────────────────────────────────────────────────────────────

def test_wilcoxon_p_returns_float_for_valid_input():
    a = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.6, 0.7, 0.5, 0.8]
    b = [0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.3, 0.6]
    p = wilcoxon_p(a, b)
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_wilcoxon_p_too_few_samples():
    assert wilcoxon_p([0.5, 0.6, 0.7], [0.3, 0.4, 0.5]) is None


def test_wilcoxon_p_mismatched_lengths():
    assert wilcoxon_p([0.5, 0.6, 0.7, 0.8], [0.3, 0.4, 0.5]) is None


def test_wilcoxon_p_identical_lists():
    # all differences are zero — scipy returns p=1.0 (newer) or raises ValueError (older → None)
    a = [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6]
    p = wilcoxon_p(a, a)
    assert p is None or p == pytest.approx(1.0)


def test_wilcoxon_p_clearly_different():
    # one list consistently higher — expect small p-value
    a = [0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8]
    b = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
    p = wilcoxon_p(a, b)
    assert p is not None
    assert p < 0.05
