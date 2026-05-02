import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.llm_judge import (
    _parse_human_labels,
    cohen_kappa,
    judge_job,
)


# ── _parse_human_labels ───────────────────────────────────────────────────────

def _write_md(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sheet.md"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_parse_human_labels_basic(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 | 1 | 5 | Engineer | Acme |
        | 2 | 0 | 3 | Manager | Corp |
        | 3 | 1 |   | Analyst | Biz |
    """)
    labels = _parse_human_labels(md)
    assert labels == [1, 0, 1]


def test_parse_human_labels_skips_unlabeled(tmp_path):
    md = _write_md(tmp_path, """
        | # | Relevant (0/1) | Quality (1-5) | Title | Company |
        |---|----------------|---------------|-------|---------|
        | 1 |   | 5 | Engineer | Acme |
        | 2 | 1 | 3 | Manager  | Corp |
    """)
    labels = _parse_human_labels(md)
    assert labels == [1]


def test_parse_human_labels_empty(tmp_path):
    md = tmp_path / "empty.md"
    md.write_text("", encoding="utf-8")
    assert _parse_human_labels(md) == []


# ── cohen_kappa ───────────────────────────────────────────────────────────────

def test_cohen_kappa_perfect_agreement():
    human = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    assert cohen_kappa(human, human) == 1.0


def test_cohen_kappa_complete_disagreement():
    # human all 1s, llm all 0s: pe=0, po=0 → kappa=0.0 (not negative — both are pure)
    human = [1, 1, 1, 1, 1]
    llm   = [0, 0, 0, 0, 0]
    k = cohen_kappa(human, llm)
    assert k is not None
    assert k <= 0.0


def test_cohen_kappa_mixed():
    human = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    llm   = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    k = cohen_kappa(human, llm)
    assert k is not None
    assert -1.0 <= k <= 1.0


def test_cohen_kappa_empty():
    assert cohen_kappa([], []) is None


def test_cohen_kappa_mismatched_lengths():
    assert cohen_kappa([1, 0, 1], [1, 0]) is None


def test_cohen_kappa_all_same_class_human():
    # pe approaches 1 when one class dominates both sides
    human = [1, 1, 1, 1, 1]
    llm   = [1, 1, 1, 1, 1]
    k = cohen_kappa(human, llm)
    assert k == 1.0


def test_cohen_kappa_range():
    human = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    llm   = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    k = cohen_kappa(human, llm)
    assert k is not None
    assert -1.0 <= k <= 1.0


# ── judge_job ─────────────────────────────────────────────────────────────────

def _mock_client(content: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    client.chat.completions.create.return_value.choices = [choice]
    return client


def test_judge_job_relevant():
    client = _mock_client('{"relevant": 1}')
    result = judge_job(client, "Senior Python dev", "Engineer", "Acme", "Python role")
    assert result == 1


def test_judge_job_not_relevant():
    client = _mock_client('{"relevant": 0}')
    result = judge_job(client, "Senior Python dev", "Janitor", "Corp", "Cleaning role")
    assert result == 0


def test_judge_job_extracts_from_noisy_response():
    # model wraps json in prose
    client = _mock_client('Sure! Here is my answer: {"relevant": 1} done.')
    result = judge_job(client, "cv", "title", "co", "desc")
    assert result == 1


def test_judge_job_malformed_returns_none():
    client = _mock_client("I cannot determine relevance.")
    result = judge_job(client, "cv", "title", "co", "desc")
    assert result is None


def test_judge_job_api_exception_returns_none():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("timeout")
    result = judge_job(client, "cv", "title", "co", "desc")
    assert result is None


def test_judge_job_retries_then_returns_none():
    client = MagicMock()
    # first two calls fail, third returns malformed
    choice = MagicMock()
    choice.message.content = "not json"
    client.chat.completions.create.side_effect = [
        Exception("fail 1"),
        Exception("fail 2"),
        Exception("fail 3"),
    ]
    result = judge_job(client, "cv", "title", "co", "desc")
    assert result is None
    assert client.chat.completions.create.call_count == 3
