"""Grounded Gemini chat responses."""

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from services.embedding_service import get_google_api_key
from services.retrieval_service import retrieve_documents


# Build and cache the Gemini chat model used to write grounded answers.
@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """Create and cache the Gemini chat model."""
    # Reusing the model client avoids rebuilding it on every Streamlit rerun.
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key=get_google_api_key(),
    )


# Retrieve supporting chunks and use them to answer one user question.
def answer_question(
    question: str, store: Chroma, messages: list[dict[str, str]]
) -> tuple[str, list[Document]]:
    """Answer a question using only retrieved document context."""
    # Retrieval happens before generation so Gemini receives relevant document evidence.
    retrieved = retrieve_documents(question, store)
    # Add source labels to the prompt so the model can cite PDF page numbers.
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source')}"
        + (f", page {doc.metadata['page']}]" if "page" in doc.metadata else "]")
        + f"\n{doc.page_content}"
        for doc in retrieved
    )
    # Recent messages help resolve follow-up questions without sending the entire chat.
    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in messages[-6:]
    ) or "(No previous conversation.)"
    # The system instruction is the grounding boundary: answers must come from context.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions about uploaded documents. Use only the supplied context. "
                "If the answer is not in the context, say you cannot find it in the document. "
                "Cite page numbers inline when they are available. Do not invent citations.\n\n"
                "Context:\n{context}\n\n"
                "Previous conversation (use only to resolve references in the new question):\n{conversation}",
            ),
            ("human", "{question}"),
        ]
    )
    # LangChain pipes the formatted prompt into Gemini and returns its response object.
    response = (prompt | get_llm()).invoke(
        {"context": context, "conversation": conversation, "question": question}
    )
    answer = response.content
    # Some model versions return structured content blocks instead of a plain string.
    if not isinstance(answer, str):
        answer = "".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in answer
        )
    return answer, retrieved
