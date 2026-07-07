# Resume RAG - Coding Standards & Guidelines

This document outlines the coding standards, best practices, and conventions for the Resume RAG Project backend development.

---

## 🐍 Python Version

- **Minimum Python Version:** 3.9
- **Recommended Python Version:** 3.10+
- **Virtual Environment:** Required for all development

---

## 📂 Folder Responsibilities

| Folder | Responsibility |
|--------|-----------------|
| `resumes/` | Input storage for PDF resume files (Git ignored, not tracked) |
| `src/` | Backend source code - all production Python modules |
| `vector_db/` | Persistent ChromaDB vector database storage (Git ignored) |
| `docs/` | Technical documentation, architecture diagrams, API specs |
| `evaluation/` | Evaluation metrics and testing scripts (Team Member 3) |

---

## 💻 Coding Guidelines

### 1. **Code Structure**
- Follow **PEP 8** style guide strictly
- Use **Clean Code Architecture** principles
- Keep functions **small and focused** (Single Responsibility Principle)
- Maximum function length: 30 lines (excluding docstrings)
- Maximum line length: 100 characters

### 2. **Type Hints**
- **Mandatory** for all function parameters and return types
- Use `typing` module for complex types
- Example:
```python
from typing import List, Dict, Optional

def process_resumes(file_paths: List[str]) -> Dict[str, str]:
    """Process resume files and return extracted text."""
    pass
```

### 3. **Docstrings**
- Use **Google-style docstrings** for all functions and classes
- Include description, parameters, returns, and exceptions
- Example:
```python
def load_pdf(file_path: str) -> str:
    """
    Load and extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from PDF
        
    Raises:
        FileNotFoundError: If file does not exist
        PDFException: If PDF parsing fails
    """
    pass
```

### 4. **Imports**
- Organize imports in three groups:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Sort alphabetically within each group
- Example:
```python
import logging
import os
from pathlib import Path
from typing import List

import chromadb
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text
```

---

## 📝 Naming Conventions

### Variables & Functions
- Use **snake_case** for variables and functions
- Use **CONSTANT_CASE** for constants
- Avoid single-letter variables (except loop counters)
- Be descriptive and explicit

```python
# Good
resume_documents = []
chunk_size = 1000
MAX_RETRIES = 3
def extract_text_from_pdf(file_path: str) -> str:
    pass

# Bad
docs = []
size = 1000
max_r = 3
def extract(fp):
    pass
```

### Classes
- Use **PascalCase** for class names
- Use **PascalCase** for exception classes

```python
class EmbeddingService:
    pass

class PDFProcessingError(Exception):
    pass
```

### Files & Modules
- Use **snake_case** for file names
- Group related functionality in modules
- Use `__init__.py` for package initialization

---

## ⚠️ Error Handling Guidelines

### 1. **Exception Handling**
- Never use bare `except` clauses
- Catch specific exceptions
- Log the exception with context
- Raise custom exceptions when appropriate

```python
import logging

logger = logging.getLogger(__name__)

try:
    document = PyPDFLoader(file_path).load()
except FileNotFoundError as e:
    logger.error(f"Resume file not found: {file_path}")
    raise PDFProcessingError(f"Unable to find resume: {file_path}") from e
except Exception as e:
    logger.error(f"Unexpected error processing {file_path}: {str(e)}")
    raise
```

### 2. **Custom Exceptions**
Create custom exceptions for domain-specific errors:

```python
class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass

class ChunkingError(Exception):
    """Raised when text chunking fails."""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass
```

### 3. **Validation**
- Validate inputs at function entry points
- Raise `ValueError` for invalid parameters
- Use type hints and pydantic for data validation

```python
def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Chunk text into overlapping segments."""
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    # Implementation
```

---

## 📊 Logging Guidelines

### 1. **Logger Configuration**
- Initialize logger per module using `__name__`
- Use appropriate log levels

```python
import logging

logger = logging.getLogger(__name__)
```

### 2. **Log Levels**

| Level | Usage |
|-------|-------|
| **DEBUG** | Detailed diagnostic information for debugging |
| **INFO** | General informational messages about progress |
| **WARNING** | Warning messages for potentially harmful situations |
| **ERROR** | Error messages for serious problems |
| **CRITICAL** | Critical errors that may cause failure |

### 3. **Logging Examples**

```python
import logging
from typing import List

logger = logging.getLogger(__name__)

def load_resumes_from_directory(directory_path: str) -> List[Document]:
    """Load all resumes from directory."""
    logger.info(f"Starting resume loading from {directory_path}")
    
    try:
        files = os.listdir(directory_path)
        logger.debug(f"Found {len(files)} files in directory")
        
        resume_count = len([f for f in files if f.endswith('.pdf')])
        logger.info(f"Found {resume_count} PDF resumes")
        
        if resume_count == 0:
            logger.warning(f"No PDF files found in {directory_path}")
        
        # Process resumes...
        logger.info(f"Successfully loaded {resume_count} resumes")
        return documents
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {directory_path}")
        raise
```

### 4. **Log Messages**
- Use clear, descriptive messages
- Include relevant context (file names, counts, etc.)
- Use structured logging where applicable

---

## 📌 Git Commit Message Format

Follow **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### Examples

```bash
git commit -m "feat(ingest): add PDF loader for resume processing"
git commit -m "fix(embedding): handle empty text before embedding generation"
git commit -m "refactor(chunking): optimize RecursiveCharacterTextSplitter configuration"
git commit -m "docs(backend): add API documentation for embedding service"
git commit -m "test(ingest): add unit tests for PDF extraction"
```

### Commit Message Guidelines
- Use imperative mood ("add" not "added")
- Don't capitalize first letter
- No period at the end of subject
- Limit subject to 50 characters
- Keep body to 72 characters per line
- Reference issues: "Closes #123"

---

## 🏗️ Clean Architecture Principles

### 1. **Modularity**
- Each file has a single, well-defined responsibility
- Functions are reusable and independent
- Minimize coupling between modules

### 2. **Reusability**
- Create utility functions for common operations
- Use configuration for varying parameters
- Avoid hardcoding values

### 3. **Testability**
- Functions should be pure when possible
- Minimize external dependencies in tests
- Use dependency injection

### 4. **Readability**
- Self-documenting code with clear names
- Comments for "why" not "what"
- Consistent formatting and structure

---

## 📦 Dependencies Management

### Adding Dependencies
1. Add to `requirements.txt` with pinned versions
2. Document the purpose in README.md
3. Update requirements with: `pip freeze > requirements.txt`

### Dependency List (Backend)
- `langchain` - Document processing and LLM framework
- `pypdf` - PDF text extraction
- `sentence-transformers` - Semantic embeddings
- `chromadb` - Vector database
- `python-dotenv` - Environment variables
- `pydantic` - Data validation

---

## 🔐 Security Practices

### 1. **Environment Variables**
- Never commit `.env` files
- Use `.env.example` as template
- Keep sensitive data in environment variables

### 2. **Path Handling**
- Use `pathlib.Path` for cross-platform compatibility
- Validate paths before processing
- Handle relative and absolute paths correctly

### 3. **Input Validation**
- Validate file types and sizes
- Sanitize file names
- Check directory existence

---

## ✅ Code Review Checklist

Before committing, verify:
- [ ] Type hints on all functions
- [ ] Docstrings present and formatted correctly
- [ ] No hardcoded values or paths
- [ ] Proper error handling and logging
- [ ] PEP 8 compliance
- [ ] No commented-out code
- [ ] No print statements (use logging)
- [ ] Functions under 30 lines
- [ ] No bare except clauses
- [ ] Variables follow snake_case convention
- [ ] Appropriate commit message format

---

## 🚀 Development Workflow

1. Create a new branch: `git checkout -b feat/feature-name`
2. Make changes following these guidelines
3. Test locally with sample resumes
4. Commit with proper message format
5. Push and create pull request
6. Address review comments
7. Merge after approval

---

## 📚 Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Clean Code Principles](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)

---

## 📞 Questions?

For coding standard questions, refer to this document or discuss with the team.
