"""Evaluation helpers for comparing generated answers to ground-truth responses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def compute_similarity_score(generated_answer: str, expected_answer: str) -> float:
    """Compute a cosine similarity score between generated and expected answers."""
    if not generated_answer.strip() or not expected_answer.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    documents = [generated_answer, expected_answer]
    vectors = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(float(similarity), 4)


def evaluate_answer(generated_answer: str, expected_answer: str) -> Dict[str, float]:
    """Return the similarity and relevance metrics for a generated answer."""
    similarity = compute_similarity_score(generated_answer, expected_answer)
    relevance = 1.0 if similarity >= 0.2 else 0.0
    retrieval_accuracy = 1.0 if similarity >= 0.2 else 0.0
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
    dataframe.to_excel(output, index=False)
    return output
