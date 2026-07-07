"""Retrieval and generation components for the resume RAG application."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.llm_prompt import build_chat_prompt

load_dotenv()

logger = logging.getLogger(__name__)


class RagPipelineError(Exception):
    """Raised when the resume RAG pipeline cannot be initialized or run."""


class ResumeRagPipeline:
    """Load a prebuilt FAISS index and answer questions from the matching resume."""

    def __init__(self, vector_db_dir: str = "vector_db") -> None:
        self.vector_db_dir = Path(vector_db_dir)
        self.index_path = self.vector_db_dir / "index.faiss"
        self.metadata_path = self.vector_db_dir / "index.pkl"
        self.documents: List[Dict[str, Any]] = []
        self.index: faiss.Index | None = None
        self._load_index()
        self._load_documents()
        self.llm = self._build_llm()
        self.chain = self._build_chain()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            raise RagPipelineError(f"FAISS index not found at {self.index_path}")
        if not self.metadata_path.exists():
            raise RagPipelineError(f"FAISS metadata not found at {self.metadata_path}")

        self.index = faiss.read_index(str(self.index_path))
        if self.index is None:
            raise RagPipelineError("Failed to load FAISS index")

    def _load_documents(self) -> None:
        with self.metadata_path.open("rb") as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, list):
            raise RagPipelineError("FAISS metadata payload is not a list")

        self.documents = payload

    def _build_llm(self):
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No LLM API key found; using a placeholder fallback")
            return None

        if provider == "openai":
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=api_key)

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            google_api_key=api_key,
        )

    def _build_chain(self):
        prompt = build_chat_prompt()
        if self.llm is None:
            return None

        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RagPipelineError("sentence-transformers is required for retrieval") from exc

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode([query], convert_to_tensor=False)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = np.asarray(embedding, dtype="float32")
        if embedding.shape[1] != self.index.d:
            padding = self.index.d - embedding.shape[1]
            if padding > 0:
                embedding = np.pad(embedding, ((0, 0), (0, padding)), mode="constant")
            else:
                embedding = embedding[:, : self.index.d]
        return embedding

    def retrieve(self, question: str, person_name: str | None = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most relevant resume documents for a question and optional person filter."""
        if self.index is None:
            raise RagPipelineError("FAISS index has not been loaded")

        query_embedding = self._embed_query(question)
        distances, indices = self.index.search(query_embedding, top_k)
        candidates: List[Dict[str, Any]] = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            document = self.documents[int(idx)]
            metadata = document.get("metadata", {}) or {}
            if person_name:
                person_value = str(metadata.get("person_name", "")).lower()
                if person_name.lower() not in person_value:
                    continue
            candidates.append(
                {
                    "id": int(idx),
                    "text": document.get("text", ""),
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

        if not candidates and person_name:
            logger.warning("No candidate matched person '%s'; falling back to top results", person_name)
            return self.retrieve(question, person_name=None, top_k=top_k)

        return candidates

    def answer_question(self, question: str, person_name: str | None = None) -> Dict[str, Any]:
        """Retrieve the best matching resume and generate an answer using the LLM."""
        candidates = self.retrieve(question, person_name=person_name, top_k=3)
        if not candidates:
            return {
                "answer": "The requested information is not available in the selected resume.",
                "context": "",
                "source": None,
                "retrieval_results": [],
            }

        selected = candidates[0]
        context = selected["text"]
        question_to_llm = question
        if self.chain is None or self.llm is None:
            answer = "The requested information is not available in the selected resume."
        else:
            answer = self.chain.invoke({"context": context, "question": question_to_llm})

        return {
            "answer": answer.strip() or "The requested information is not available in the selected resume.",
            "context": context,
            "source": selected["metadata"].get("source", None),
            "retrieval_results": candidates,
        }
