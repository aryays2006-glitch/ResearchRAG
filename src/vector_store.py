import chromadb
import ollama

from ingest import load_chunks


# Connect to Chroma
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="research_papers"
)


# Load chunks from the research papers
chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")


# Embed and store each chunk
for i, chunk in enumerate(chunks):

    response = ollama.embed(
        model="nomic-embed-text",
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
                "source": chunk["source"]
            }
        ]
    )

    print(
        f"Stored {i + 1}/{len(chunks)}: "
        f"{chunk['chunk_id']}"
    )


print("\nDone!")
print("Total documents in Chroma:", collection.count())