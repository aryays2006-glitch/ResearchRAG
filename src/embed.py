import ollama
from ingest import load_chunks


chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")

embeddings = []

for i, chunk in enumerate(chunks):

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )

    embedding = response["embeddings"][0]

    embeddings.append(embedding)

    print(
        f"Embedded {i + 1}/{len(chunks)}: "
        f"{chunk['chunk_id']}"
    )


print("\nEmbedding complete!")

print("Total embeddings:", len(embeddings))
print("Dimensions:", len(embeddings[0]))

print("\nFirst embedding, first 10 numbers:")
print(embeddings[0][:10])
print(embedding[:10])