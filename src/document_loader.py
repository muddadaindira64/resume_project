"""Resume document loading utilities for the Resume RAG pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

try:
    from src.preprocess import clean_resume_text
except ModuleNotFoundError:  # pragma: no cover - supports script execution
    from preprocess import clean_resume_text

logger = logging.getLogger(__name__)


class ResumeLoaderError(Exception):
    """Raised when resume documents cannot be loaded."""


def _derive_person_name(file_path: Path, text: str) -> str:
    """Infer a person name from the filename or the text content."""
    stem = file_path.stem.replace("_", " ").replace("-", " ")
    stem_tokens = [token for token in stem.split() if token.lower() not in {"resume", "cv", "profile"}]

    if stem_tokens:
        return " ".join(stem_tokens).strip().title()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:10]:
        if len(line.split()) <= 4 and not any(char.isdigit() for char in line):
            candidate = line.replace("|", " ").strip()
            if candidate and candidate.lower() not in {"resume", "curriculum vitae"}:
                return candidate.title()

    return file_path.stem.replace("_", " ").title()


def load_single_resume(file_path: Union[str, Path]) -> Document:
    """Load a single resume PDF and return it as a LangChain Document."""
    path = Path(file_path)

    if not path.exists():
        raise ResumeLoaderError(f"Resume file does not exist: {path}")

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ResumeLoaderError(f"Only PDF resumes are supported: {path}")

    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        extracted_text = "\n\n".join(page.page_content for page in pages if getattr(page, "page_content", ""))
        cleaned_text = clean_resume_text(extracted_text)

        metadata = {
            "person_name": _derive_person_name(path, cleaned_text),
            "source": path.name,
            "document_type": "resume",
        }

        logger.info("Loaded resume '%s' with %s characters", path.name, len(cleaned_text))
        return Document(page_content=cleaned_text, metadata=metadata)
    except Exception as exc:
        logger.error("Failed to load resume '%s': %s", path.name, exc)
        raise ResumeLoaderError(f"Unable to process resume '{path.name}'") from exc


def load_resume_documents(resumes_dir: Union[str, Path]) -> List[Document]:
    """Load every PDF resume from a folder into a list of LangChain Documents."""
    directory = Path(resumes_dir)

    if not directory.exists():
        raise ResumeLoaderError(f"Resume directory does not exist: {directory}")

    if not directory.is_dir():
        raise ResumeLoaderError(f"Resume path is not a directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise ResumeLoaderError(f"No PDF resumes found in {directory}")

    documents: List[Document] = []
    for pdf_file in pdf_files:
        try:
            documents.append(load_single_resume(pdf_file))
        except ResumeLoaderError as exc:
            logger.warning("Skipping '%s': %s", pdf_file.name, exc)

    if not documents:
        raise ResumeLoaderError(f"No resume documents could be loaded from {directory}")

    return documents
