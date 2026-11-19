"""A small, grounded document Q&A app built with Streamlit and Gemini."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

APP_TITLE = "Document Assistant"
PERSIST_DIRECTORY = Path(".chroma")


def get_google_api_key() -> str | None:
    """Read the API key locally from .env or from Streamlit Cloud secrets."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def extract_uploaded_file(uploaded_file) -> list[Document]:
    """Extract text and page metadata from a PDF or TXT upload."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": page_number},
                    )
                )
        return documents

    text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata={"source": uploaded_file.name})] if text else []


@st.cache_resource(show_spinner=False)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=get_google_api_key()
    )


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key=get_google_api_key(),
    )


def build_vector_store(uploaded_file) -> tuple[Chroma, int]:
    raw_documents = extract_uploaded_file(uploaded_file)
    if not raw_documents:
        raise ValueError("No readable text was found in that file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(raw_documents)
    file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    collection_name = f"document_{file_id}"

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )
    store.add_documents(chunks)
    return store, len(chunks)


def answer_question(
    question: str, store: Chroma, messages: list[dict[str, str]]
) -> tuple[str, list[Document]]:
    retrieved = store.similarity_search(question, k=5)
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source')}"
        + (f", page {doc.metadata['page']}]" if "page" in doc.metadata else "]")
        + f"\n{doc.page_content}"
        for doc in retrieved
    )
    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in messages[-6:]
    ) or "(No previous conversation.)"
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
    response = (prompt | get_llm()).invoke(
        {"context": context, "conversation": conversation, "question": question}
    )
    answer = response.content
    if not isinstance(answer, str):
        answer = "".join(block.get("text", str(block)) if isinstance(block, dict) else str(block) for block in answer)
    return answer, retrieved


def reset_chat() -> None:
    st.session_state.messages = []


st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
st.title("📄 Document Assistant")
st.caption("Upload a PDF or text file, then ask questions grounded in its contents.")

if not get_google_api_key():
    st.warning("Add GOOGLE_API_KEY to a .env file before asking questions.")

with st.sidebar:
    st.header("Your document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if st.button("Clear conversation", use_container_width=True):
        reset_chat()
        st.rerun()
    st.divider()
    st.markdown("**How it works**")
    st.caption("Text is extracted, split into chunks, embedded with Gemini, and searched in ChromaDB before Gemini writes an answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    file_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Reading and indexing your document…"):
            try:
                st.session_state.store, chunk_count = build_vector_store(uploaded_file)
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

question = st.chat_input("Ask a question about your document…", disabled=not st.session_state.get("store"))
if question:
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
