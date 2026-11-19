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
st.title("📄 Document Assistant")
st.caption("Upload a PDF or text file, then ask questions grounded in its contents.")

if not get_google_api_key():
    st.warning("Add GOOGLE_API_KEY to a .env file or Streamlit secrets before asking questions.")

with st.sidebar:
    st.header("Your document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if st.button("Clear conversation", width="stretch"):
        reset_chat()
        st.rerun()
    st.divider()
    st.markdown("**How it works**")
    st.caption(
        "Text is extracted, split into chunks, embedded with Gemini, and searched in "
        "ChromaDB before Gemini writes an answer."
    )

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
                st.session_state.messages = []
                st.success(f"Indexed {uploaded_file.name} ({chunk_count} searchable chunks).")
            except Exception as error:
                st.error(f"Could not process the file: {error}")
                st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.get("store"):
    st.info("Upload a document to get started. Try asking: ‘What are the main topics covered?’")

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
