"""
Resume Document Ingestion Module

This module handles loading and extracting text from PDF resume files.
It reads all PDF files from a specified directory and extracts their text content.

Key Functions:
    - load_resumes_from_directory: Load all PDFs from a directory
    - load_single_resume: Load a single PDF file

Dependencies:
    - langchain: PyPDFLoader for PDF processing
    - logging: For tracking ingestion activities
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Configure logger for this module
logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass


def load_single_resume(file_path: str) -> Document:
    """
    Load and extract text from a single PDF resume file.
    
    This function reads a PDF file and extracts all text content,
    returning a langchain Document object.
    
    Args:
        file_path: Path to the PDF file to process
        
    Returns:
        langchain Document object containing extracted text
        
    Raises:
        FileNotFoundError: If the file does not exist
        PDFProcessingError: If PDF extraction fails
        
    Example:
        >>> doc = load_single_resume("resumes/john_doe.pdf")
        >>> print(doc.page_content[:100])
    """
    file_path = str(file_path)
    
    # Validate file exists
    if not os.path.exists(file_path):
        logger.error(f"Resume file not found: {file_path}")
        raise FileNotFoundError(f"Resume file does not exist: {file_path}")
    
    # Validate file is PDF
    if not file_path.lower().endswith('.pdf'):
        logger.warning(f"File is not a PDF: {file_path}")
        raise PDFProcessingError(f"File must be a PDF: {file_path}")
    
    try:
        logger.debug(f"Loading resume: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Combine all pages into a single document
        combined_content = "\n".join([page.page_content for page in pages])
        
        # Create a single document with metadata
        document = Document(
            page_content=combined_content,
            metadata={
                "source": file_path,
                "filename": os.path.basename(file_path),
                "total_pages": len(pages)
            }
        )
        
        logger.info(f"Successfully loaded resume: {os.path.basename(file_path)} "
                   f"({len(pages)} pages, {len(combined_content)} characters)")
        
        return document
        
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        raise PDFProcessingError(
            f"Error processing PDF file {file_path}: {str(e)}"
        ) from e


def load_resumes_from_directory(
    directory_path: str,
    recursive: bool = False
) -> List[Document]:
    """
    Load and extract text from all PDF resume files in a directory.
    
    This function scans a directory for PDF files and loads them sequentially.
    Files are processed in sorted order for consistency. Failed files are
    logged but do not stop processing of remaining files.
    
    Args:
        directory_path: Path to the directory containing PDF resumes
        recursive: If True, search subdirectories recursively (default: False)
        
    Returns:
        List of langchain Document objects containing resume text and metadata
        
    Raises:
        FileNotFoundError: If directory does not exist
        PDFProcessingError: If no valid PDF files are found
        
    Example:
        >>> resumes = load_resumes_from_directory("resumes")
        >>> print(f"Loaded {len(resumes)} resumes")
        Loaded 5 resumes
        
        >>> # With recursive search
        >>> all_resumes = load_resumes_from_directory("data", recursive=True)
    """
    directory_path = Path(directory_path)
    
    # Validate directory exists
    if not directory_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    if not directory_path.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        raise PDFProcessingError(f"Path must be a directory: {directory_path}")
    
    logger.info(f"Starting resume ingestion from: {directory_path}")
    
    # Find all PDF files
    if recursive:
        pdf_files = sorted(directory_path.glob("**/*.pdf"))
        logger.debug(f"Searching recursively in {directory_path}")
    else:
        pdf_files = sorted(directory_path.glob("*.pdf"))
        logger.debug(f"Searching {directory_path} (non-recursive)")
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        raise PDFProcessingError(f"No PDF files found in {directory_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files in directory")
    
    # Load resumes
    documents: List[Document] = []
    failed_files: List[str] = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        try:
            logger.debug(f"Processing file {idx}/{len(pdf_files)}: {pdf_file.name}")
            document = load_single_resume(str(pdf_file))
            documents.append(document)
            
        except (FileNotFoundError, PDFProcessingError) as e:
            logger.warning(f"Skipping file {pdf_file.name}: {str(e)}")
            failed_files.append(pdf_file.name)
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_file.name}: {str(e)}")
            failed_files.append(pdf_file.name)
    
    # Log results
    logger.info(f"Successfully loaded {len(documents)}/{len(pdf_files)} resumes")
    
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {', '.join(failed_files)}")
    
    if not documents:
        raise PDFProcessingError(
            f"Failed to load any resume files from {directory_path}"
        )
    
    logger.info(f"Resume ingestion complete. Total documents: {len(documents)}")
    
    return documents


def get_ingestion_stats(documents: List[Document]) -> dict:
    """
    Calculate and return statistics about ingested documents.
    
    Args:
        documents: List of langchain Document objects
        
    Returns:
        Dictionary containing statistics about the documents
        
    Example:
        >>> stats = get_ingestion_stats(documents)
        >>> print(f"Total characters: {stats['total_characters']}")
    """
    if not documents:
        return {
            "total_documents": 0,
            "total_characters": 0,
            "average_document_length": 0,
        }
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "average_document_length": total_chars // len(documents),
        "min_document_length": min(len(doc.page_content) for doc in documents),
        "max_document_length": max(len(doc.page_content) for doc in documents),
    }
