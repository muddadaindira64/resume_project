"""Text cleaning utilities for resume documents."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


class PreprocessError(Exception):
    """Raised when resume text cannot be cleaned."""


def clean_resume_text(text: str) -> str:
    """Remove noisy OCR and formatting artifacts while keeping useful resume content."""
    if not isinstance(text, str):
        raise PreprocessError("Resume text must be a string")

    if not text.strip():
        return ""

    cleaned = text.replace("\x0c", " ")
    cleaned = re.sub(r"\[(?:image|img|figure|diagram|screenshot)\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpage\s*\d+\s*(?:of\s*\d+)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:confidential|copyright|all rights reserved)\b", "", cleaned, flags=re.IGNORECASE)

    lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        if re.fullmatch(r"\[(?:image|img|figure|diagram|screenshot)\]", line, flags=re.IGNORECASE):
            continue

        if re.fullmatch(r"page\s*\d+\s*(?:of\s*\d+)?", line, flags=re.IGNORECASE):
            continue

        if line.lower() in {"resume", "cv", "curriculum vitae"}:
            continue

        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", line):
            continue

        lines.append(line)

    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()
