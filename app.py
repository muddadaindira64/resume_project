"""Streamlit app for the Resume RAG assistant."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.evaluation import evaluate_answer, load_ground_truth, save_results
from src.rag_chain import ResumeRagPipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Resume RAG Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> ResumeRagPipeline:
    """Load the existing FAISS-backed RAG pipeline once for the app session."""
    return ResumeRagPipeline(vector_db_dir="vector_db")


@st.cache_data(show_spinner=False)
def load_person_names() -> List[str]:
    """Infer person names from the vector store metadata for the dropdown."""
    pipeline = load_pipeline()
    names = []
    seen = set()
    for record in pipeline.documents:
        metadata = record.get("metadata", {}) or {}
        person_name = str(metadata.get("person_name", "")).strip()
        if person_name and person_name.lower() not in seen:
            seen.add(person_name.lower())
            names.append(person_name)
    return sorted(names)


st.sidebar.header("Resume RAG Assistant")
st.sidebar.markdown("AI Powered Resume Question Answering using RAG")
st.sidebar.subheader("Project Overview")
st.sidebar.write("This app retrieves the most relevant resume from the existing vector store and uses an LLM to answer questions grounded only in that resume.")
st.sidebar.subheader("RAG Workflow")
st.sidebar.markdown("Resume PDF → Embedding → Vector Database → Retriever → LLM → Answer")

st.title("Resume RAG Assistant")
st.caption("AI Powered Resume Question Answering using RAG")

person_names = load_person_names()
if not person_names:
    st.error("No person names were found in the existing vector database metadata.")
    st.stop()

selected_person = st.selectbox("Step 1: Select Person", person_names)
question = st.text_area(
    "Step 2: Ask a question",
    value="What technologies does this person know?",
    height=120,
)

if st.button("Generate Answer", type="primary"):
    with st.spinner("Retrieving the best resume match and generating an answer..."):
            pipeline = load_pipeline()

            # Perform a retrieval restricted to the selected person and log results
            candidates = pipeline.retrieve(question, person_name=selected_person, top_k=5)
            logging.getLogger(__name__).info("Selected person: %s; Retrieved %s candidate(s)", selected_person, len(candidates))
            for c in candidates:
                logging.getLogger(__name__).debug("Candidate id=%s source=%s distance=%s", c.get("id"), c.get("metadata", {}).get("source"), c.get("distance"))

            if not candidates:
                st.error(f"No documents found for '{selected_person}' or no matching content in that resume.")
                result = {
                    "answer": "The requested information is not available in the selected resume.",
                    "context": "",
                    "source": None,
                    "retrieval_results": [],
                }
            else:
                # Use the top candidate for generation
                top = candidates[0]
                pipeline_candidate_context = top.get("text", "")
                if pipeline.chain is None or pipeline.llm is None:
                    answer = "The requested information is not available in the selected resume."
                else:
                    answer = pipeline.chain.invoke({"context": pipeline_candidate_context, "question": question})

                result = {
                    "answer": answer.strip() if isinstance(answer, str) else str(answer),
                    "context": pipeline_candidate_context,
                    "source": top.get("metadata", {}).get("source") or top.get("metadata", {}).get("source_filename"),
                    "retrieval_results": candidates,
                }

    st.subheader("Selected Person")
    st.write(selected_person)
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
            matching_rows = ground_truth[ground_truth["Name"].astype(str).str.lower() == selected_person.lower()]
            if not matching_rows.empty:
                expected_answer = str(matching_rows.iloc[0].get("GroundTruth_Answer", "")).strip()
                if expected_answer:
                    evaluation = evaluate_answer(result.get("answer", ""), expected_answer)
                    st.subheader("Evaluation")
                    st.write(f"Similarity Score: {evaluation['similarity_score']}")
                    st.write(f"Answer Relevance: {evaluation['answer_relevance']}")
                    st.write(f"Retrieval Accuracy: {evaluation['retrieval_accuracy']}")

                    results_output = Path("results") / "generated_results.xlsx"
                    results_payload = [{
                        "Person Name": selected_person,
                        "Question": question,
                        "Generated Answer": result.get("answer", ""),
                        "Expected Answer": expected_answer,
                        "Similarity Score": evaluation["similarity_score"],
                    }]
                    save_results(results_payload, results_output)
                    st.success(f"Evaluation saved to {results_output}")
                else:
                    st.info("No expected answer was found for the selected person in the ground truth workbook.")
            else:
                st.info("No matching ground-truth answer was found for the selected person.")
        except Exception as exc:
            st.warning(f"Evaluation could not be completed: {exc}")
