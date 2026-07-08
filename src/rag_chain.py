"""Retrieval and generation components for the resume RAG application."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List
import chromadb
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
    """Use a ChromaDB collection to retrieve resumes and answer questions."""

    def __init__(self, vector_db_dir: str = "vector_db", collection_name: str = "resume_collection") -> None:
        self.vector_db_dir = Path(vector_db_dir)
        self.collection_name = collection_name
        self.documents: List[Dict[str, Any]] = []
        self.client = None
        self.collection = None

        # Connect to ChromaDB collection
        self._connect_collection()
        # Load documents metadata from collection
        self._load_documents_from_collection()

        self.llm = self._build_llm()
        self.chain = self._build_chain()

    def _connect_collection(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=str(self.vector_db_dir))
            # get_collection will raise if not found; prefer get_collection to avoid creating unintended collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                # Fall back to get_or_create to be permissive in local dev
                self.collection = self.client.get_or_create_collection(name=self.collection_name)

            if self.collection is None:
                raise RagPipelineError(f"ChromaDB collection '{self.collection_name}' not available in {self.vector_db_dir}")

            logger.info("Connected to ChromaDB collection '%s' at %s", self.collection_name, self.vector_db_dir)
        except Exception as exc:
            logger.error("Failed to connect to ChromaDB: %s", exc)
            raise RagPipelineError(f"Failed to connect to ChromaDB: {exc}") from exc

    def _load_documents_from_collection(self) -> None:
        try:
            payload = self.collection.get()
            ids = payload.get("ids", []) or []
            metadatas = payload.get("metadatas", []) or []
            documents = payload.get("documents", []) or []

            self.documents = []
            for i, _id in enumerate(ids):
                md = metadatas[i] if i < len(metadatas) else {}
                doc_text = documents[i] if i < len(documents) else ""
                self.documents.append({
                    "id": _id,
                    "text": doc_text,
                    "metadata": md,
                })

            logger.info("Loaded %s document records from ChromaDB collection '%s'", len(self.documents), self.collection_name)
        except Exception as exc:
            logger.error("Failed to load documents from ChromaDB: %s", exc)
            raise RagPipelineError(f"Failed to load documents from ChromaDB: {exc}") from exc

    def _build_llm(self):
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No LLM API key found; using a placeholder fallback")
            return None

        if provider == "openai":
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=api_key)

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
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
        return embedding

    def retrieve(self, question: str, person_name: str | None = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most relevant resume documents for a question and optional person filter."""
        if self.collection is None:
            raise RagPipelineError("ChromaDB collection has not been loaded")
        logger.debug("Retrieval requested. person_name=%s, question='%s', top_k=%s", person_name, question, top_k)

        # If a person_name is provided, try to restrict the retrieval to documents
        # that belong to that person (case-insensitive exact match). This ensures
        # the returned results are strictly from the selected person.
        if person_name:
            # Use ChromaDB metadata filtering (where) to restrict retrieval to this person_name
            try:
                query_embedding = self._embed_query(question)
                # Chroma expects plain python lists
                embedding_list = query_embedding.tolist()[0]
                # Perform metadata-filtered query
                results = self.collection.query(query_embeddings=[embedding_list], n_results=top_k, where={"person_name": person_name})

                formatted_results: List[Dict[str, Any]] = []
                if results.get("ids") and results["ids"][0]:
                    for index, _id in enumerate(results["ids"][0]):
                        formatted_results.append(
                            {
                                "id": _id,
                                "text": results.get("documents", [[""]])[0][index] if results.get("documents") else "",
                                "metadata": results.get("metadatas", [[{}]])[0][index] if results.get("metadatas") else {},
                                "distance": results.get("distances", [[0]])[0][index] if results.get("distances") else 0,
                            }
                        )

                logger.debug("Person-filtered retrieval returned %s results for '%s'", len(formatted_results), person_name)
                return formatted_results
            except Exception as exc:
                logger.error("Error during Chroma person-filtered retrieval: %s", exc)
                return []

        # Default: global retrieval across the whole index
        # Global (unfiltered) retrieval using ChromaDB
        try:
            query_embedding = self._embed_query(question)
            embedding_list = query_embedding.tolist()[0]
            results = self.collection.query(query_embeddings=[embedding_list], n_results=top_k)

            formatted_results: List[Dict[str, Any]] = []
            if results.get("ids") and results["ids"][0]:
                for index, _id in enumerate(results["ids"][0]):
                    formatted_results.append(
                        {
                            "id": _id,
                            "text": results.get("documents", [[""]])[0][index] if results.get("documents") else "",
                            "metadata": results.get("metadatas", [[{}]])[0][index] if results.get("metadatas") else {},
                            "distance": results.get("distances", [[0]])[0][index] if results.get("distances") else 0,
                        }
                    )

            logger.debug("Global Chroma retrieval returned %s candidates", len(formatted_results))
            return formatted_results
        except Exception as exc:
            logger.error("Error during Chroma global retrieval: %s", exc)
            return []

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
