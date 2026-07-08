"""Streamlit app for the Resume RAG assistant."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src.evaluation import evaluate_answer, load_ground_truth, save_results
from src.rag_chain import ResumeRagPipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Resume RAG Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> ResumeRagPipeline:
    """Load the existing ChromaDB-backed RAG pipeline once for the app session."""
    return ResumeRagPipeline(vector_db_dir="vector_db")


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
            if retrieved_person:
                matching_rows = ground_truth[ground_truth["Name"].astype(str).str.lower() == retrieved_person.lower()]
            else:
                matching_rows = ground_truth.iloc[0:0]

            if not matching_rows.empty:
                expected_answer = str(matching_rows.iloc[0].get("GroundTruth_Answer", "")).strip()
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
