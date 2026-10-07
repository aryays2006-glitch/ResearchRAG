import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import load_chunks


# ============================================================
# 1. LOAD DOCUMENT CHUNKS
# ============================================================

chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# 2. PREPARE BM25 SEARCH
# ============================================================

documents = [
    chunk["text"]
    for chunk in chunks
]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# ============================================================
# 3. CONNECT TO CHROMA
# ============================================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="research_papers"
)


# ============================================================
# 4. USER QUERY
# ============================================================

query = "Why is self-attention better than recurrent networks?"


# ============================================================
# 5. VECTOR SEARCH
# ============================================================

response = ollama.embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = response["embeddings"][0]

vector_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

vector_chunk_ids = vector_results["ids"][0]


# ============================================================
# 6. BM25 SEARCH
# ============================================================

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_query
)

bm25_top_indices = (
    bm25_scores.argsort()[-10:][::-1]
)

bm25_chunk_ids = [
    chunks[index]["chunk_id"]
    for index in bm25_top_indices
]


# ============================================================
# 7. RECIPROCAL RANK FUSION (RRF)
# ============================================================

rrf_scores = {}

k = 60

# Add vector search rankings
for rank, chunk_id in enumerate(
    vector_chunk_ids,
    start=1
):
    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + 1 / (k + rank)
    )


# Add BM25 rankings
for rank, chunk_id in enumerate(
    bm25_chunk_ids,
    start=1
):
    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + 1 / (k + rank)
    )


# Sort by RRF score
ranked_chunks = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)


# ============================================================
# 8. SELECT RRF CANDIDATES FOR RERANKING
# ============================================================

candidate_count = 10

candidate_ids = [
    chunk_id
    for chunk_id, score in ranked_chunks[
        :candidate_count
    ]
]


candidate_chunks = [
    chunk
    for chunk in chunks
    if chunk["chunk_id"] in candidate_ids
]


print(
    f"\nSelected {len(candidate_chunks)} "
    "candidates for reranking."
)


# ============================================================
# 9. LOAD CROSS-ENCODER RERANKER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker loaded successfully.")


# ============================================================
# 10. CREATE QUERY-DOCUMENT PAIRS
# ============================================================

pairs = [
    [
        query,
        chunk["text"]
    ]
    for chunk in candidate_chunks
]


# ============================================================
# 11. CALCULATE RERANKER SCORES
# ============================================================

reranker_scores = reranker.predict(
    pairs
)


# ============================================================
# 12. COMBINE CHUNKS WITH SCORES
# ============================================================

reranked_results = []

for chunk, score in zip(
    candidate_chunks,
    reranker_scores
):

    reranked_results.append(
        {
            "chunk": chunk,
            "score": float(score)
        }
    )


# ============================================================
# 13. SORT BY RERANKER SCORE
# ============================================================

reranked_results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# ============================================================
# 14. DISPLAY FINAL RESULTS
# ============================================================

print(
    "\n========== FINAL RERANKED RESULTS ==========\n"
)

for rank, result in enumerate(
    reranked_results,
    start=1
):

    chunk = result["chunk"]
    score = result["score"]

    print(
        f"--- Final Rank {rank} ---"
    )

    print(
        f"Chunk ID: {chunk['chunk_id']}"
    )

    print(
        f"Page: {chunk['page']}"
    )

    print(
        f"Reranker Score: {score:.4f}"
    )

    print("\nText:")

    print(
        chunk["text"][:500]
    )

    print("\n")