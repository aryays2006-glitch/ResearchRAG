from pathlib import Path
from pypdf import PdfReader

PAPERS_DIR = Path("data/papers")


def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_chunks():
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    all_chunks = []

    for pdf_path in pdf_files:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text()

            if not page_text:
                continue

            chunks = chunk_text(page_text)

            for chunk_number, chunk in enumerate(chunks, start=1):

                chunk_data = {
                    "chunk_id": f"{pdf_path.stem}_p{page_number}_c{chunk_number}",
                    "text": chunk,
                    "page": page_number,
                    "source": pdf_path.name,
                }

                all_chunks.append(chunk_data)

    return all_chunks


if __name__ == "__main__":

    chunks = load_chunks()

    print(f"Created {len(chunks)} total chunks.")

    first_chunk = chunks[0]

    print("\n--- First Chunk ---")
    print("Chunk ID:", first_chunk["chunk_id"])
    print("Page:", first_chunk["page"])
    print("Source:", first_chunk["source"])
    print("Text:", first_chunk["text"][:500])