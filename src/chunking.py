"""
Text Chunking Module

This module handles splitting large resume documents into smaller,
overlapping chunks for efficient processing and embedding generation.

The module uses langchain's RecursiveCharacterTextSplitter to maintain
semantic coherence while creating appropriately sized chunks.

Key Functions:
    - chunk_text: Split documents into chunks
    - chunk_with_overlap: Advanced chunking with custom parameters

Dependencies:
    - langchain: RecursiveCharacterTextSplitter for text splitting
    - typing: Type hints for better code clarity
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logger for this module
logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingError(Exception):
    """Raised when text chunking fails."""
    pass


def chunk_text(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Split documents into overlapping chunks using recursive character splitting.
    
    This function maintains semantic coherence by splitting on paragraph,
    sentence, and word boundaries in that order. Overlapping chunks help
    preserve context between chunk boundaries.
    
    Args:
        documents: List of langchain Document objects to chunk
        chunk_size: Target size of each chunk in characters (default: 1000)
        chunk_overlap: Number of overlapping characters between chunks (default: 200)
        
    Returns:
        List of langchain Document objects representing chunks
        
    Raises:
        ChunkingError: If chunking parameters are invalid or processing fails
        
    Example:
        >>> from src.ingest import load_resumes_from_directory
        >>> documents = load_resumes_from_directory("resumes")
        >>> chunks = chunk_text(documents, chunk_size=1000, chunk_overlap=200)
        >>> print(f"Created {len(chunks)} chunks")
        Created 42 chunks
        
        >>> # Using custom parameters
        >>> large_chunks = chunk_text(documents, chunk_size=2000, chunk_overlap=500)
    """
    # Validate inputs
    if not documents:
        logger.warning("No documents provided for chunking")
        raise ChunkingError("Documents list cannot be empty")
    
    if chunk_size <= 0:
        logger.error(f"Invalid chunk_size: {chunk_size}. Must be positive")
        raise ChunkingError("chunk_size must be a positive integer")
    
    if chunk_overlap < 0:
        logger.error(f"Invalid chunk_overlap: {chunk_overlap}. Cannot be negative")
        raise ChunkingError("chunk_overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        logger.error(f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
        raise ChunkingError("chunk_overlap must be less than chunk_size")
    
    logger.info(f"Starting chunking with size={chunk_size}, overlap={chunk_overlap}")
    logger.debug(f"Total documents to chunk: {len(documents)}")
    
    try:
        # Initialize the text splitter
        # RecursiveCharacterTextSplitter splits on:
        # 1. "\n\n" (paragraph)
        # 2. "\n" (newline)
        # 3. " " (space)
        # 4. "" (character)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split all documents
        chunks: List[Document] = []
        
        for doc_idx, document in enumerate(documents, 1):
            logger.debug(f"Chunking document {doc_idx}/{len(documents)}: "
                        f"{document.metadata.get('filename', 'unknown')}")
            
            # Split the document
            doc_chunks = text_splitter.split_documents([document])
            
            # Update metadata for each chunk
            for chunk_idx, chunk in enumerate(doc_chunks, 1):
                chunk.metadata["chunk_id"] = len(chunks)
                chunk.metadata["chunk_index"] = chunk_idx
                chunk.metadata["total_chunks_in_doc"] = len(doc_chunks)
                chunks.append(chunk)
            
            logger.debug(f"Document {doc_idx} created {len(doc_chunks)} chunks")
        
        logger.info(f"Chunking complete. Created {len(chunks)} total chunks "
                   f"from {len(documents)} documents")
        
        # Log statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        logger.debug(f"Chunk statistics - Min: {min(chunk_lengths)}, "
                    f"Max: {max(chunk_lengths)}, "
                    f"Avg: {sum(chunk_lengths) // len(chunk_lengths)}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error during text chunking: {str(e)}")
        raise ChunkingError(f"Failed to chunk documents: {str(e)}") from e


def chunk_with_overlap(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separator: Optional[str] = None
) -> List[Document]:
    """
    Advanced chunking with custom separator for specialized use cases.
    
    This function provides more control over chunking behavior by allowing
    custom separators. Useful for resume-specific formatting.
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Target size of each chunk (default: 1000)
        chunk_overlap: Number of overlapping characters (default: 200)
        separator: Custom separator (if None, uses default separators)
        
    Returns:
        List of chunked Document objects
        
    Raises:
        ChunkingError: If parameters are invalid
        
    Example:
        >>> # Split by sections (e.g., "---")
        >>> chunks = chunk_with_overlap(documents, separator="---")
    """
    if separator:
        logger.debug(f"Using custom separator: {repr(separator)}")
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[separator, "\n\n", "\n", " ", ""]
            )
            
            chunks: List[Document] = []
            for document in documents:
                doc_chunks = text_splitter.split_documents([document])
                chunks.extend(doc_chunks)
            
            logger.info(f"Created {len(chunks)} chunks with custom separator")
            return chunks
            
        except Exception as e:
            logger.error(f"Error with custom separator: {str(e)}")
            raise ChunkingError(f"Failed to chunk with custom separator: {str(e)}") from e
    
    # Fall back to default chunking
    return chunk_text(documents, chunk_size, chunk_overlap)


def get_chunking_stats(chunks: List[Document]) -> dict:
    """
    Calculate statistics about chunked documents.
    
    Args:
        chunks: List of chunked Document objects
        
    Returns:
        Dictionary containing chunking statistics
        
    Example:
        >>> stats = get_chunking_stats(chunks)
        >>> print(f"Average chunk size: {stats['avg_chunk_size']}")
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "total_characters": 0,
        }
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    total_chars = sum(chunk_sizes)
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": total_chars // len(chunks),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "total_characters": total_chars,
    }
