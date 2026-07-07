# Resume Vector DB Pipeline

Run the pipeline from the project root:

```bash
python src/create_vector_db.py
```

This loads all PDF resumes from the resumes folder, cleans each document, attaches metadata, creates embeddings, and writes the local FAISS files to the vector_db folder.
