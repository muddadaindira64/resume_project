"""Retrieval and generation components for the resume RAG application."""

from __future__ import annotations

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List
import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI

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
        # Keep the old index alias for backward compatibility with existing tests and workflow
        self.index = self.collection
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

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _extract_person_names(self) -> List[str]:
        names = []
        for document in self.documents:
            metadata = document.get("metadata", {}) or {}
            person_name = str(metadata.get("person_name", "")).strip()
            if person_name and person_name.lower() not in {name.lower() for name in names}:
                names.append(person_name)
        return sorted(names, key=lambda value: -len(value))

    def _match_person_name(self, question: str) -> List[str]:
        normalized_question = self._normalize_text(question)
        candidate_names = self._extract_person_names()

        # Full name exact match first
        exact_matches: List[str] = []
        for person_name in candidate_names:
            pattern = rf"\b{re.escape(person_name.lower())}\b"
            if re.search(pattern, normalized_question):
                exact_matches.append(person_name)

        if exact_matches:
            return exact_matches

        # If no full-name match, use unique token-based candidate matching
        question_tokens = set(re.findall(r"\b\w+\b", normalized_question))
        token_matches: Dict[str, List[str]] = {}
        for person_name in candidate_names:
            name_tokens = set(re.findall(r"\b\w+\b", person_name.lower()))
            for token in name_tokens & question_tokens:
                token_matches.setdefault(token, []).append(person_name)

        unique_matches: List[str] = []
        for token, names in token_matches.items():
            if len(names) == 1 and names[0] not in unique_matches:
                unique_matches.append(names[0])

        return unique_matches

    def _is_comparison_query(self, question: str) -> bool:
        normalized_question = self._normalize_text(question)
        return bool(re.search(r"\b(compare|versus|vs|between)\b", normalized_question))

    def _format_query_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        return formatted_results

    def _query_collection(self, query_embedding: np.ndarray, top_k: int = 1, where: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        try:
            embedding_list = query_embedding.tolist()[0]
            results = self.collection.query(query_embeddings=[embedding_list], n_results=top_k, where=where)
            return self._format_query_results(results)
        except Exception as exc:
            logger.error("Error querying ChromaDB collection: %s", exc)
            return []

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

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most relevant resume documents for a question."""
        if self.collection is None:
            raise RagPipelineError("ChromaDB collection has not been loaded")

        logger.debug("Retrieval requested for question='%s' top_k=%s", question, top_k)
        if not question.strip():
            return []

        query_embedding = self._embed_query(question)
        matched_names = self._match_person_name(question)

        if matched_names and not self._is_comparison_query(question):
            if len(matched_names) == 1:
                selected_person = matched_names[0]
                logger.debug("Detected candidate name in question: %s", selected_person)
                return self._query_collection(query_embedding, top_k=1, where={"person_name": selected_person})

            logger.debug("Ambiguous name match detected; falling back to semantic retrieval")

        if matched_names and self._is_comparison_query(question):
            logger.debug("Comparison query detected for candidates: %s", matched_names)
            candidates: List[Dict[str, Any]] = []
            for person_name in matched_names:
                candidates.extend(self._query_collection(query_embedding, top_k=1, where={"person_name": person_name}))
            return sorted(candidates, key=lambda item: item.get("distance", 0))[:top_k]

        return self._query_collection(query_embedding, top_k=top_k)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieve the best matching resume and generate an answer using the LLM."""
        candidates = self.retrieve(question, top_k=3)
        if not candidates:
            return {
                "answer": "The requested information is not available.",
                "context": "",
                "source": None,
                "retrieval_results": [],
            }

        selected = candidates[0]
        context = selected["text"]
        if self.chain is None or self.llm is None:
            answer = "The requested information is not available."
        else:
            answer = self.chain.invoke({"context": context, "question": question})

        return {
            "answer": str(answer).strip() if answer else "The requested information is not available.",
            "context": context,
            "source": selected["metadata"].get("source", None),
            "retrieval_results": candidates,
        }
