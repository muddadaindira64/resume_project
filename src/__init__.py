"""
Resume RAG Backend Package

This package contains modules for:
- Document ingestion (ingest.py)
- Text chunking (chunking.py)
- Embedding generation and vector storage (embedding.py)
"""

__version__ = "1.0.0"
__author__ = "Backend Development Team"

from src.ingest import load_resumes_from_directory
from src.chunking import chunk_text
from src.embedding import EmbeddingService

__all__ = [
    "load_resumes_from_directory",
    "chunk_text",
    "EmbeddingService",
]
