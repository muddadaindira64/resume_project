"""
Embedding Generation and Vector Database Module

This module handles generating semantic embeddings for text chunks and
storing them in a ChromaDB vector database for efficient retrieval.

The module uses Sentence Transformers for embedding generation, which provides
high-quality semantic embeddings suitable for similarity search.

Key Components:
    - EmbeddingService: Main class for embedding and storage operations

Dependencies:
    - sentence-transformers: For semantic embedding generation
    - chromadb: For vector database operations
    - typing: Type hints for better code clarity
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict

import chromadb
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embedding generation or storage fails."""
    pass


class EmbeddingService:
    """
    Service class for generating embeddings and managing vector database.
    
    This class encapsulates all operations related to embedding generation
    and vector database management using ChromaDB.
    
    Attributes:
        model_name: Name of the sentence-transformers model to use
        model: Loaded SentenceTransformer model
        client: ChromaDB client for vector database operations
        db_path: Path where ChromaDB is stored
        collection: ChromaDB collection for resume embeddings
        
    Example:
        >>> service = EmbeddingService(db_path="vector_db/chroma_db")
        >>> service.embed_and_store(chunks)
        >>> results = service.search("Python developer with 5 years experience", k=5)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        db_path: str = "vector_db/chroma_db",
        collection_name: str = "resume_collection"
    ):
        """
        Initialize the EmbeddingService.
        
        Args:
            model_name: Name of sentence-transformers model (default: all-MiniLM-L6-v2)
            db_path: Path to store ChromaDB (default: vector_db/chroma_db)
            collection_name: Name of the ChromaDB collection (default: resume_collection)
            
        Raises:
            EmbeddingError: If model loading or database initialization fails
            
        Example:
            >>> # Using default model
            >>> service = EmbeddingService()
            
            >>> # Using custom model
            >>> service = EmbeddingService(
            ...     model_name="all-mpnet-base-v2",
            ...     db_path="data/vectors"
            ... )
        """
        try:
            self.model_name = model_name
            self.db_path = db_path
            self.collection_name = collection_name
            
            logger.info(f"Initializing EmbeddingService with model: {model_name}")
            
            # Load embedding model
            logger.debug(f"Loading sentence-transformers model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: "
                       f"{self.model.get_sentence_embedding_dimension()}")
            
            # Initialize ChromaDB
            logger.debug(f"Initializing ChromaDB at: {db_path}")
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{collection_name}' ready. "
                       f"Current size: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {str(e)}")
            raise EmbeddingError(
                f"Failed to initialize EmbeddingService: {str(e)}"
            ) from e
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
            
        Example:
            >>> embedding = service.embed_text("Python developer")
            >>> print(len(embedding))  # Embedding dimension
            384
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid text input for embedding")
            raise EmbeddingError("Text must be a non-empty string")
        
        try:
            logger.debug(f"Generating embedding for text of length {len(text)}")
            embedding = self.model.encode(text, convert_to_tensor=False).tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (more efficient).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
            
        Example:
            >>> texts = ["Python developer", "Java engineer", "DevOps specialist"]
            >>> embeddings = service.embed_batch(texts)
        """
        if not texts:
            logger.error("Empty text list provided")
            raise EmbeddingError("Text list cannot be empty")
        
        try:
            logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e
    
    def embed_and_store(
        self,
        chunks: List[Document],
        batch_size: int = 32
    ) -> int:
        """
        Generate embeddings for chunks and store them in ChromaDB.
        
        This function processes chunks in batches for efficiency,
        generates embeddings, and stores them with metadata.
        
        Args:
            chunks: List of Document chunks to embed and store
            batch_size: Number of chunks to process in each batch (default: 32)
            
        Returns:
            Number of chunks successfully stored
            
        Raises:
            EmbeddingError: If embedding or storage fails
            
        Example:
            >>> from src.chunking import chunk_text
            >>> chunks = chunk_text(documents)
            >>> num_stored = service.embed_and_store(chunks)
            >>> print(f"Stored {num_stored} chunks")
        """
        if not chunks:
            logger.warning("No chunks provided for embedding and storage")
            raise EmbeddingError("Chunks list cannot be empty")
        
        logger.info(f"Starting embedding and storage of {len(chunks)} chunks "
                   f"with batch_size={batch_size}")
        
        try:
            # Extract texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            
            # Generate embeddings in batches
            total_processed = 0
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_size_actual = batch_end - batch_start
                
                logger.debug(f"Processing batch {batch_start // batch_size + 1}: "
                           f"chunks {batch_start} to {batch_end}")
                
                # Extract batch
                batch_texts = texts[batch_start:batch_end]
                batch_chunks = chunks[batch_start:batch_end]
                
                # Generate embeddings
                batch_embeddings = self.embed_batch(batch_texts)
                
                # Prepare data for ChromaDB
                ids = [f"chunk_{chunk.metadata.get('chunk_id', i)}" 
                       for i, chunk in enumerate(batch_chunks)]
                metadatas = [chunk.metadata for chunk in batch_chunks]
                
                # Store in ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    metadatas=metadatas,
                    documents=batch_texts
                )
                
                total_processed += batch_size_actual
                logger.debug(f"Batch stored. Total processed: {total_processed}")
            
            logger.info(f"Successfully embedded and stored {total_processed} chunks")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error during embedding and storage: {str(e)}")
            raise EmbeddingError(
                f"Failed to embed and store chunks: {str(e)}"
            ) from e
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Query text to search for
            k: Number of results to return (default: 5)
            
        Returns:
            List of search results with content and metadata
            
        Raises:
            EmbeddingError: If search fails
            
        Example:
            >>> results = service.search("Python developer with AI experience", k=3)
            >>> for result in results:
            ...     print(f"Match: {result['metadata']['filename']}")
        """
        if not query or not isinstance(query, str):
            logger.error("Invalid query for search")
            raise EmbeddingError("Query must be a non-empty string")
        
        try:
            logger.debug(f"Searching for: '{query}' (k={k})")
            
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:  # Check if results exist
                for i, chunk_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "id": chunk_id,
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    })
            
            logger.debug(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise EmbeddingError(f"Failed to search: {str(e)}") from e
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics
            
        Example:
            >>> stats = service.get_stats()
            >>> print(f"Total chunks: {stats['total_chunks']}")
        """
        try:
            count = self.collection.count()
            
            return {
                "model_name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "total_chunks": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path,
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> None:
        """
        Delete the current collection from the database.
        
        Warning: This action is irreversible.
        
        Example:
            >>> # Clear old data and start fresh
            >>> service.delete_collection()
        """
        try:
            logger.warning(f"Deleting collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection {self.collection_name} deleted and recreated")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise EmbeddingError(f"Failed to delete collection: {str(e)}") from e
