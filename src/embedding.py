"""Semantic embedding pipeline for resume RAG using SentenceTransformers and ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embedding generation or storage fails."""


class EmbeddingService:
    """Generate semantic embeddings for full resume documents and store them in ChromaDB."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        db_path: str = "vector_db",
        collection_name: str = "resume_collection",
    ) -> None:
        """Initialize the embedding model and persistent ChromaDB collection."""
        try:
            self.model_name = model_name
            self.db_path = str(Path(db_path))
            self.collection_name = collection_name

            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            logger.info("Initializing EmbeddingService with model '%s'", model_name)

            self.model = SentenceTransformer(model_name)
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB collection '%s' ready at '%s' with %s document(s)",
                self.collection_name,
                self.db_path,
                self.collection.count(),
            )
        except Exception as exc:
            logger.error("Failed to initialize EmbeddingService: %s", exc)
            raise EmbeddingError(f"Failed to initialize EmbeddingService: {exc}") from exc

    def _prepare_metadata(self, document: Document) -> Dict[str, Any]:
        """Create a metadata payload with the required resume fields."""
        metadata = dict(document.metadata or {})
        metadata["person_name"] = metadata.get("person_name") or metadata.get("name") or "Unknown"
        metadata["source_filename"] = (
            metadata.get("source_filename")
            or metadata.get("source")
            or metadata.get("filename")
            or "unknown.pdf"
        )
        return metadata

    def embed_text(self, text: str) -> List[float]:
        """Generate a semantic embedding for a single resume text."""
        if not text or not isinstance(text, str):
            logger.error("Invalid text input for embedding")
            raise EmbeddingError("Text must be a non-empty string")

        try:
            logger.debug("Generating embedding for text of length %s", len(text))
            return self.model.encode(text, convert_to_tensor=False).tolist()
        except Exception as exc:
            logger.error("Error generating embedding: %s", exc)
            raise EmbeddingError(f"Failed to generate embedding: {exc}") from exc

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate semantic embeddings for a batch of resume texts."""
        if not texts:
            logger.error("Empty text list provided")
            raise EmbeddingError("Text list cannot be empty")

        try:
            logger.debug("Generating embeddings for batch of %s texts", len(texts))
            embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
            logger.debug("Successfully generated %s embeddings", len(embeddings))
            return embeddings
        except Exception as exc:
            logger.error("Error generating batch embeddings: %s", exc)
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    def embed_and_store(self, documents: List[Document], batch_size: int = 32) -> int:
        """Embed full resume documents and store them in ChromaDB as one document each."""
        if not documents:
            logger.warning("No documents provided for embedding and storage")
            raise EmbeddingError("Documents list cannot be empty")

        logger.info("Starting embedding and storage of %s resume document(s)", len(documents))

        try:
            texts = [document.page_content for document in documents]
            embeddings = self.embed_batch(texts)

            ids = [f"resume_{index}" for index in range(len(documents))]
            metadatas = [self._prepare_metadata(document) for document in documents]

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )

            logger.info("Successfully embedded and stored %s resume document(s)", len(documents))
            return len(documents)
        except Exception as exc:
            logger.error("Error during embedding and storage: %s", exc)
            raise EmbeddingError(f"Failed to embed and store documents: {exc}") from exc

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the ChromaDB collection for semantically similar resumes."""
        if not query or not isinstance(query, str):
            logger.error("Invalid query for search")
            raise EmbeddingError("Query must be a non-empty string")

        try:
            logger.debug("Searching for query '%s' (k=%s)", query, k)
            query_embedding = self.embed_text(query)
            results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

            formatted_results: List[Dict[str, Any]] = []
            if results.get("ids") and results["ids"][0]:
                for index, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "id": chunk_id,
                            "content": results["documents"][0][index] if results.get("documents") else "",
                            "metadata": results["metadatas"][0][index] if results.get("metadatas") else {},
                            "distance": results["distances"][0][index] if results.get("distances") else 0,
                        }
                    )

            logger.debug("Found %s result(s)", len(formatted_results))
            return formatted_results
        except Exception as exc:
            logger.error("Error during search: %s", exc)
            raise EmbeddingError(f"Failed to search: {exc}") from exc

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                "model_name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "total_documents": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path,
            }
        except Exception as exc:
            logger.error("Error getting stats: %s", exc)
            return {}

    def delete_collection(self) -> None:
        """Delete and recreate the current ChromaDB collection."""
        try:
            logger.warning("Deleting collection '%s'", self.collection_name)
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' deleted and recreated", self.collection_name)
        except Exception as exc:
            logger.error("Error deleting collection: %s", exc)
            raise EmbeddingError(f"Failed to delete collection: {exc}") from exc
