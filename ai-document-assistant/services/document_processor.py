"""Document extraction and text chunking."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Extract readable text and source metadata from an uploaded PDF or TXT file.
def extract_uploaded_file(uploaded_file) -> list[Document]:
    """Extract text and page metadata from a PDF or TXT upload."""
    # The suffix determines which local parser should handle the upload.
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages, start=1):
            # Keep the page number so answers can point users back to the PDF.
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": page_number},
                    )
                )
        return documents

    # UploadedFile behaves like an in-memory file, so no temporary file is needed.
    text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata={"source": uploaded_file.name})] if text else []


# Divide extracted text into overlapping pieces for embedding and retrieval.
def split_documents(documents: list[Document]) -> list[Document]:
    """Split extracted documents into overlapping searchable chunks."""
    # Overlap prevents context from being lost when a sentence crosses a chunk boundary.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents(documents)
