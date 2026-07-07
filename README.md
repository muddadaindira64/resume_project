# Resume RAG (Retrieval-Augmented Generation) Project

## Project Overview

This project implements a **Resume Retrieval-Augmented Generation (RAG)** system that processes and retrieves information from resume documents using advanced NLP techniques. The system combines document ingestion, text chunking, and embedding generation to create a vector database for efficient resume retrieval.

## 🎯 Project Goal

Build an intelligent resume search and analysis system that can:
- Ingest multiple resume documents in PDF format
- Extract and process text content efficiently
- Generate semantic embeddings for resume chunks
- Enable retrieval of relevant resume information based on queries
- Support analysis and evaluation of candidate qualifications

---

## 📁 Folder Structure

```
Resume-RAG-Project/
│
├── resumes/                    # Input folder: contains all PDF resume files
├── src/                        # Backend source code (Backend Module)
│   ├── __init__.py            # Python package initialization
│   ├── ingest.py              # PDF ingestion and text extraction
│   ├── chunking.py            # Text chunking using RecursiveCharacterTextSplitter
│   └── embedding.py           # Embedding generation and vector DB storage
│
├── vector_db/                 # ChromaDB vector database storage
├── docs/                      # Project documentation and design docs
├── evaluation/                # Evaluation scripts (Assigned to Team Member 3)
│
├── README.md                  # This file
├── AGENTS.md                  # Coding standards and guidelines
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── .env.example              # Environment variables template
```

---

## 🔧 Backend Module Description

The **Backend (Document Ingestion) Module** is responsible for:

### 1. **ingest.py**
- Reads all PDF resume files from the `resumes/` directory
- Extracts text content using PyPDFLoader
- Handles multiple resume processing with error handling
- Logs all ingestion activities for debugging
- Returns processed resume data in memory

### 2. **chunking.py**
- Splits extracted text into manageable chunks using `RecursiveCharacterTextSplitter`
- Configurable chunk size (default: 1000 tokens) and overlap (default: 200 tokens)
- Maintains context between chunks for semantic coherence
- Returns list of text chunks ready for embedding

### 3. **embedding.py**
- Generates semantic embeddings using Sentence Transformers
- Configurable embedding model (default: `all-MiniLM-L6-v2`)
- Stores embeddings into ChromaDB vector database
- Persists vector database in `vector_db/` directory
- Provides methods to query and retrieve similar resumes

---

## 👥 Team Module Responsibilities

| Module | Responsibility | Team Member |
|--------|-----------------|-------------|
| **Backend (Document Ingestion)** | PDF ingestion, text extraction, chunking, embedding generation | Team Member 1 |
| **Retrieval & RAG Pipeline** | Query processing, vector similarity search, context building | Team Member 2 |
| **LLM Integration & Streamlit UI** | LLM responses, frontend interface, user interaction | Team Member 3 |

---

## 📦 Installation Steps

### 1. Clone Repository
```bash
cd /path/to/project
git clone <repository-url>
cd resume_project
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Add Resume Files
Place your PDF resume files in the `resumes/` directory:
```bash
cp /path/to/resumes/*.pdf resumes/
```

---

## 🚀 How to Run Backend

### Step 1: Ingest Resumes
```python
from src.ingest import load_resumes_from_directory

# Load all resumes from the resumes/ folder
resume_documents = load_resumes_from_directory(directory_path="resumes")
print(f"Loaded {len(resume_documents)} resumes")
```

### Step 2: Chunk Text
```python
from src.chunking import chunk_text

# Split documents into chunks
chunks = chunk_text(
    documents=resume_documents,
    chunk_size=1000,
    chunk_overlap=200
)
print(f"Created {len(chunks)} chunks")
```

### Step 3: Generate Embeddings & Store in Vector DB
```python
from src.embedding import EmbeddingService

# Initialize embedding service
embedding_service = EmbeddingService(
    model_name="all-MiniLM-L6-v2",
    db_path="vector_db/chroma_db"
)

# Generate embeddings and store in ChromaDB
embedding_service.embed_and_store(chunks)
print("Embeddings stored in vector database")
```

### Complete Example Script
```python
from src.ingest import load_resumes_from_directory
from src.chunking import chunk_text
from src.embedding import EmbeddingService

def main():
    # Step 1: Load resumes
    documents = load_resumes_from_directory("resumes")
    
    # Step 2: Create chunks
    chunks = chunk_text(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Generate embeddings and store
    embedding_service = EmbeddingService(db_path="vector_db/chroma_db")
    embedding_service.embed_and_store(chunks)

if __name__ == "__main__":
    main()
```

---

## 📋 Dependencies

- **langchain** - LLM framework and document processing
- **pypdf** - PDF text extraction
- **sentence-transformers** - Semantic embeddings
- **chroma-client** - Vector database
- **python-dotenv** - Environment variable management
- **pydantic** - Data validation

---

## 🤝 Contributing

Follow the guidelines in [AGENTS.md](AGENTS.md) for:
- Coding standards
- Naming conventions
- Error handling
- Logging practices
- Git commit message format

---

## 📝 License

[Specify your license here]

---

## 📞 Support

For backend-related issues, contact Team Member 1 (Backend Developer)