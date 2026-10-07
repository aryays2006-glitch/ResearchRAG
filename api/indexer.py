from pathlib import Path

import ollama
import chromadb
from pypdf import PdfReader

from src.ingest import chunk_text


EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "research_papers"


def process_pdf(pdf_path: Path):
    """
    Process one PDF at a time.

    The uploaded PDF becomes the only active research paper.
    Previous PDFs and their Chroma embeddings are removed.
    """

    # Remove all previous PDFs
    papers_directory = pdf_path.parent

    for existing_pdf in papers_directory.glob("*.pdf"):
        if existing_pdf != pdf_path:
            existing_pdf.unlink()

    # Connect to Chroma
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Clear previous embeddings
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(
            ids=existing_ids
        )

    # Read the new PDF
    reader = PdfReader(pdf_path)

    chunks_to_store = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        chunks = chunk_text(page_text)

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            chunks_to_store.append(
                {
                    "chunk_id": (
                        f"{pdf_path.stem}"
                        f"_p{page_number}"
                        f"_c{chunk_number}"
                    ),
                    "text": chunk,
                    "page": page_number,
                    "source": pdf_path.name,
                }
            )

    if not chunks_to_store:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    # Create embeddings and store them
    for chunk in chunks_to_store:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk["text"]
        )

        embedding = response["embeddings"][0]

        collection.upsert(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                }
            ],
        )

    return {
        "filename": pdf_path.name,
        "pages": len(reader.pages),
        "chunks": len(chunks_to_store),
    }