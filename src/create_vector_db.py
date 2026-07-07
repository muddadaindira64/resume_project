"""Create a persistent ChromaDB vector database from resume PDFs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

try:
    from src.embedding import EmbeddingService
    from src.ingest import load_resumes_from_directory
except ModuleNotFoundError:  # pragma: no cover - supports script execution
    from embedding import EmbeddingService
    from ingest import load_resumes_from_directory

logger = logging.getLogger(__name__)


def build_resume_vector_db(
    resumes_dir: str = "resumes",
    vector_db_dir: str = "vector_db",
) -> Dict[str, Any]:
    """Load resume PDFs, create semantic embeddings, and store them in ChromaDB."""
    resumes_path = Path(resumes_dir)
    vector_db_path = Path(vector_db_dir)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    documents = load_resumes_from_directory(str(resumes_path))
    embeddings_service = EmbeddingService(db_path=str(vector_db_path))
    documents_processed = embeddings_service.embed_and_store(documents)

    logger.info("Created ChromaDB vector store for %s resume(s)", documents_processed)

    return {
        "documents_processed": documents_processed,
        "vector_db_dir": str(vector_db_path),
        "collection_name": embeddings_service.collection_name,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_resume_vector_db()
    print("Resume vector database created successfully")
    print(f"Documents processed: {result['documents_processed']}")
    print(f"Database directory: {result['vector_db_dir']}")
    print(f"Collection: {result['collection_name']}")
