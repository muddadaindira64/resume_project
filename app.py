"""Streamlit app for the Resume RAG assistant."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.evaluation import evaluate_answer, load_ground_truth, save_results
from src.rag_chain import ResumeRagPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Resume RAG Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> ResumeRagPipeline:
    """Load the existing ChromaDB-backed RAG pipeline once for the app session."""
    return ResumeRagPipeline(vector_db_dir="vector_db")


def normalize_ground_truth_key(value: str) -> str:
    """Normalize a candidate or ground truth name for robust matching."""
    if not value:
        return ""

    normalized = str(value).strip().lower()
    normalized = re.sub(r"\.pdf$", "", normalized)
    normalized = re.sub(r"[_\-]", " ", normalized)
    normalized = re.sub(r"\b(resume|updated|cv|profile)\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9 ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def match_ground_truth_row(ground_truth, person_name: str | None, source_filename: str | None):
    """Find the best matching ground truth row based on resume metadata."""
    candidates = [candidate for candidate in [person_name, source_filename] if candidate]
    logger.info("Retrieved metadata person_name=%s source_filename=%s", person_name, source_filename)

    if not candidates:
        logger.warning("No candidate metadata available for ground truth matching.")
        return None

    candidate_norms = [normalize_ground_truth_key(candidate) for candidate in candidates if normalize_ground_truth_key(candidate)]
    if not candidate_norms:
        logger.warning("No valid normalized candidate metadata available for ground truth matching.")
        return None

    best_score = 0.0
    best_index = None
    best_name = None

    for index, row in ground_truth.iterrows():
        row_name = str(row.get("Name", ""))
        row_norm = normalize_ground_truth_key(row_name)
        if not row_norm:
            continue

        for candidate_norm in candidate_norms:
            if candidate_norm == row_norm:
                logger.info("Exact ground truth name match found: %s", row_name)
                return row

            candidate_tokens = set(candidate_norm.split())
            row_tokens = set(row_norm.split())
            common_tokens = candidate_tokens & row_tokens
            if not common_tokens:
                continue

            # Compute a score that rewards both token coverage and overlap
            coverage = len(common_tokens) / max(len(candidate_tokens), len(row_tokens))
            jaccard = len(common_tokens) / len(candidate_tokens | row_tokens)
            score = max(coverage, jaccard)

            if score > best_score or (score == best_score and row_norm and best_name and len(row_norm) > len(normalize_ground_truth_key(best_name))):
                best_score = score
                best_index = index
                best_name = row_name

    if best_index is not None and best_score > 0:
        logger.info("Best ground truth match found: %s (score=%s)", best_name, best_score)
        return ground_truth.loc[best_index]

    logger.warning("No matching ground-truth answer was found after normalization.")
    return None


st.sidebar.header("Resume RAG Assistant")
st.sidebar.markdown("AI Powered Resume Question Answering using RAG")
st.sidebar.subheader("Project Overview")
st.sidebar.write("This app retrieves the most relevant resume from the existing vector store and uses an LLM to answer questions grounded only in that resume.")
st.sidebar.subheader("RAG Workflow")
st.sidebar.markdown("Resume PDF → Embedding → Vector Database → Retriever → LLM → Answer")

st.title("Resume RAG Assistant")
st.caption("AI Powered Resume Question Answering using RAG")

question = st.text_area(
    "Ask a question",
    value="What technologies does Indira know?",
    height=120,
)

if st.button("Generate Answer", type="primary"):
    with st.spinner("Retrieving the best resume match and generating an answer..."):
        pipeline = load_pipeline()
        result = pipeline.answer_question(question)

    st.subheader("Retrieved Candidate")
    retrieved_person = None
    if result.get("retrieval_results"):
        retrieved_person = result["retrieval_results"][0].get("metadata", {}).get("person_name")
    st.write(retrieved_person or "No candidate could be retrieved")

    st.subheader("Resume Source")
    st.write(result.get("source") or "No source metadata available")
    st.subheader("Retrieved Resume Context")
    with st.expander("View retrieved context", expanded=False):
        st.text_area("Context", result.get("context", ""), height=220)
    st.subheader("Generated Answer")
    st.write(result.get("answer", ""))

    ground_truth_path = Path("ground_truth") / "Rag_Resumes (Responses).xlsx"
    if ground_truth_path.exists():
        try:
            ground_truth = load_ground_truth(ground_truth_path)
            source_filename = result.get("source") or result.get("retrieval_results", [])[0].get("metadata", {}).get("source_filename") if result.get("retrieval_results") else None
            best_match = match_ground_truth_row(ground_truth, retrieved_person, source_filename)

            if best_match is not None:
                expected_answer = str(best_match.get("GroundTruth_Answer", "")).strip()
                if expected_answer:
                    evaluation = evaluate_answer(result.get("answer", ""), expected_answer)

                    st.subheader("Ground Truth Answer")
                    st.write(expected_answer)

                    st.subheader("Evaluation")
                    st.write(f"Similarity Score: {evaluation['similarity_score']}")
                    st.write(f"Answer Relevance: {evaluation['answer_relevance']}")
                    st.write(f"Retrieval Accuracy: {evaluation['retrieval_accuracy']}")

                    results_output = Path("results") / "generated_results.xlsx"
                    results_payload = [{
                        "Person Name": retrieved_person or "Unknown",
                        "Question": question,
                        "Generated Answer": result.get("answer", ""),
                        "Expected Answer": expected_answer,
                        "Similarity Score": evaluation["similarity_score"],
                    }]
                    save_results(results_payload, results_output)
                    st.success(f"Evaluation saved to {results_output}")
                else:
                    st.info("No expected answer was found for the retrieved candidate in the ground truth workbook.")
            else:
                st.info("No matching ground-truth answer was found for the retrieved candidate.")
        except Exception as exc:
            st.warning(f"Evaluation could not be completed: {exc}")
