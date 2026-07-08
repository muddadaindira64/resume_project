"""Evaluation helpers for comparing generated answers to ground-truth responses."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation data cannot be loaded or processed."""


def load_ground_truth(excel_path: str | Path) -> pd.DataFrame:
    """Load the evaluation workbook containing sample questions and expected answers."""
    path = Path(excel_path)
    if not path.exists():
        raise EvaluationError(f"Ground truth file not found: {path}")

    dataframe = pd.read_excel(path)
    expected_columns = {"Sample_query", "GroundTruth_Answer", "Name"}
    missing = expected_columns.difference(set(dataframe.columns))
    if missing:
        raise EvaluationError(f"Ground truth workbook is missing columns: {sorted(missing)}")

    return dataframe


def _extract_skill_set(text: str) -> set[str]:
    """Extract normalized skill tokens from answer text."""
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[|,;/]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if not normalized:
        return set()

    tokens = normalized.split()
    return {token for token in tokens if token}


def compute_similarity_score(generated_answer: str, expected_answer: str) -> float:
    """Compute a skill overlap similarity score as a percentage."""
    generated_skills = _extract_skill_set(generated_answer)
    expected_skills = _extract_skill_set(expected_answer)

    if not expected_skills:
        if not generated_skills:
            return 100.0
        return 0.0

    matched_skills = generated_skills & expected_skills
    score = len(matched_skills) / len(expected_skills) * 100.0
    return round(float(score), 2)


def evaluate_answer(generated_answer: str, expected_answer: str) -> Dict[str, float]:
    """Return the similarity and relevance metrics for a generated answer."""
    similarity = compute_similarity_score(generated_answer, expected_answer)
    relevance = 1.0 if similarity >= 20.0 else 0.0
    retrieval_accuracy = 1.0 if similarity >= 20.0 else 0.0
    return {
        "similarity_score": similarity,
        "answer_relevance": relevance,
        "retrieval_accuracy": retrieval_accuracy,
    }


def save_results(results: List[Dict[str, Any]], output_path: str | Path) -> Path:
    """Persist evaluation rows to an Excel workbook inside the results folder."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(results)
    if output.exists():
        existing = pd.read_excel(output)
        dataframe = pd.concat([existing, dataframe], ignore_index=True)

    dataframe.to_excel(output, index=False)
    return output
