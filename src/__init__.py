"""Resume RAG backend package."""

__version__ = "1.0.0"
__author__ = "Backend Development Team"

from src.chunking import chunk_text
from src.embedding import EmbeddingError, EmbeddingService
from src.ingest import load_resumes_from_directory

__all__ = [
    "EmbeddingError",
    "EmbeddingService",
    "chunk_text",
    "load_resumes_from_directory",
]
