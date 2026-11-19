"""Gemini API key and embedding model helpers."""

import os

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load the Gemini credential from local development or Streamlit Cloud.
def get_google_api_key() -> str | None:
    """Read the API key locally from .env or from Streamlit Cloud secrets."""
    # Local development uses .env, while Community Cloud provides st.secrets.
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Returning None lets the UI show a helpful configuration warning.
        return None


# Build the Gemini embedding client once and reuse it across Streamlit reruns.
@st.cache_resource(show_spinner=False)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and cache the Gemini embedding model."""
    # Embedding model initialization is expensive, so Streamlit reuses it across reruns.
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=get_google_api_key(),
    )
