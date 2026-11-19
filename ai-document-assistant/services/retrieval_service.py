"""Vector-store construction and document retrieval."""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from services.document_processor import extract_uploaded_file, split_documents
from services.embedding_service import get_embeddings

PERSIST_DIRECTORY = Path(".chroma")


# Convert one uploaded file into a searchable ChromaDB vector collection.
def build_vector_store(uploaded_file) -> tuple[Chroma, int]:
    """Extract, chunk, embed, and persist one uploaded document."""
    raw_documents = extract_uploaded_file(uploaded_file)
    if not raw_documents:
        raise ValueError("No readable text was found in that file.")

    chunks = split_documents(raw_documents)
    # A content hash gives each file a stable collection name without using its filename.
    file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    collection_name = f"document_{file_id}"
    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIRECTORY),
    )
    # Chroma asks the embedding model to convert each chunk into a searchable vector.
    store.add_documents(chunks)
    return store, len(chunks)


# Find the document chunks that are most relevant to the user's question.
def retrieve_documents(question: str, store: Chroma, count: int = 5):
    """Return the chunks most relevant to a question."""
    # The question is embedded and compared with stored chunk vectors.
    return store.similarity_search(question, k=count)
