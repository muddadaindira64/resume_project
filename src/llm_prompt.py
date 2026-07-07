"""Prompt templates and system instructions for the resume RAG assistant."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an AI Resume Analysis Assistant.

Your task is to answer questions ONLY using the retrieved resume context.

Rules:

• Use ONLY the retrieved resume information.

• Never use outside knowledge.

• Never hallucinate.

• Never guess.

• If the answer is not available in the resume, reply exactly:
"The requested information is not available in the selected resume."

• Keep answers short, professional and accurate."""

PROMPT_TEMPLATE = """Resume Context:
{context}

Question:
{question}

Answer:"""


def build_chat_prompt() -> ChatPromptTemplate:
    """Create the chat prompt template used by the LLM chain."""
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", PROMPT_TEMPLATE)]
    )
