from rank_bm25 import BM25Okapi

from ingest import load_chunks


# Load all research-paper chunks
chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")


# Prepare text for BM25
documents = [
    chunk["text"]
    for chunk in chunks
]


# Tokenize each document
tokenized_documents = [
    document.lower().split()
    for document in documents
]


# Create BM25 index
bm25 = BM25Okapi(tokenized_documents)


# User question
query = "Why is self-attention better than recurrent networks?"


# Tokenize the query
tokenized_query = query.lower().split()


# Search using BM25
scores = bm25.get_scores(tokenized_query)


# Get the top 3 results
top_indices = scores.argsort()[-3:][::-1]


print("\n========== BM25 Search Results ==========\n")


for rank, index in enumerate(top_indices, start=1):

    chunk = chunks[index]

    print(f"--- Result {rank} ---")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Score: {scores[index]:.4f}")
    print("\nText:")
    print(chunk["text"][:500])
    print()