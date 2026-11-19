"""Streamlit entry point for the document assistant."""

from __future__ import annotations

import hashlib

import streamlit as st
from dotenv import load_dotenv

from services.chat_service import answer_question
from services.embedding_service import get_google_api_key
from services.retrieval_service import build_vector_store

load_dotenv()

APP_TITLE = "Document Assistant"


# Reset the chat messages stored for the current browser session.
def reset_chat() -> None:
    """Clear the current conversation."""
    # Replacing the list removes the visible chat history from this session.
    st.session_state.messages = []


st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
st.title("Document Assistant")
st.caption("A focused workspace for understanding your documents with grounded AI answers.")

with st.container(border=True):
    st.subheader("Ask your document")
    st.write(
        "Upload a technical PDF or text file, then explore its topics, processes, "
        "security guidance, and key details."
    )

if not get_google_api_key():
    st.warning("Add GOOGLE_API_KEY to a .env file or Streamlit secrets before asking questions.")

with st.sidebar:
    st.markdown("## Document Assistant")
    st.caption("Private, document-grounded Q&A")
    st.divider()
    st.subheader("Workspace")
    uploaded_file = st.file_uploader(
        "Add a document",
        type=["pdf", "txt"],
        help="Supported formats: PDF and plain text (.txt).",
    )
    if st.button("Clear conversation", width="stretch"):
        reset_chat()
        st.rerun()
    st.divider()
    st.markdown("**Processing pipeline**")
    st.caption(
        "Extract → chunk → embed → retrieve → answer"
    )
    st.caption("Answers are generated from retrieved document context.")

if "messages" not in st.session_state:
    # Session state survives Streamlit reruns caused by widget interactions.
    st.session_state.messages = []

if uploaded_file is not None:
    # Hashing detects a genuinely new upload and avoids re-indexing the same file.
    file_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Reading and indexing your document…"):
            try:
                st.session_state.store, chunk_count = build_vector_store(uploaded_file)
                # Keep the vector store and file identity available for future reruns.
                st.session_state.file_key = file_key
                st.session_state.file_name = uploaded_file.name
                st.session_state.chunk_count = chunk_count
                st.session_state.messages = []
            except Exception as error:
                st.error(f"Could not process the file: {error}")
                st.stop()

if st.session_state.get("file_name"):
    st.subheader("Current document")
    metric_file, metric_chunks, metric_mode = st.columns(3)
    with metric_file:
        st.metric("Document", st.session_state.file_name)
    with metric_chunks:
        st.metric("Searchable chunks", st.session_state.get("chunk_count", "—"))
    with metric_mode:
        st.metric("Answer mode", "Grounded")
    st.success(f"{st.session_state.file_name} is ready for questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.get("store"):
    with st.container(border=True):
        st.subheader("Start with a document")
        st.write("Upload a PDF or TXT file from the sidebar to begin a grounded conversation.")
        st.markdown(
            "**Try asking:** \n"
            "- What are the main topics covered?\n"
            "- Explain the authentication process.\n"
            "- Where is API integration discussed?"
        )

question = st.chat_input(
    "Ask a question about your document…",
    disabled=not st.session_state.get("store"),
)
if question:
    # Store the user message before generating so it is available for follow-up context.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document…"):
            try:
                answer, sources = answer_question(
                    question, st.session_state.store, st.session_state.messages
                )
                st.markdown(answer)
                # Deduplicate source labels because several chunks may come from one page.
                source_labels = sorted(
                    {
                        f"{doc.metadata['source']} — page {doc.metadata['page']}"
                        if "page" in doc.metadata
                        else doc.metadata["source"]
                        for doc in sources
                    }
                )
                with st.expander("Sources"):
                    st.write("\n".join(f"- {label}" for label in source_labels))
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as error:
                st.error(f"The answer could not be generated: {error}")
