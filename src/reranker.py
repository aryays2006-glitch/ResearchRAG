from sentence_transformers import CrossEncoder

from ingest import load_chunks


# ============================================================
# 1. LOAD RESEARCH PAPER CHUNKS
# ============================================================

chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# 2. LOAD THE RERANKER MODEL
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker loaded successfully.")


# ============================================================
# 3. USER QUESTION
# ============================================================

query = "Why is self-attention better than recurrent networks?"


# ============================================================
# 4. SELECT CANDIDATE CHUNKS
# ============================================================
# For this first test, we use a few chunks from the paper.
# Later, these will come from our hybrid search + RRF pipeline.

candidate_chunks = chunks[:8]


# ============================================================
# 5. CREATE QUESTION + CHUNK PAIRS
# ============================================================

pairs = [
    [query, chunk["text"]]
    for chunk in candidate_chunks
]


# ============================================================
# 6. SCORE THE CANDIDATE CHUNKS
# ============================================================

scores = reranker.predict(pairs)


# ============================================================
# 7. COMBINE CHUNKS WITH THEIR SCORES
# ============================================================

ranked_results = []

for chunk, score in zip(candidate_chunks, scores):

    ranked_results.append(
        {
            "chunk": chunk,
            "score": float(score)
        }
    )


# ============================================================
# 8. SORT BY RELEVANCE
# ============================================================

ranked_results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n========== Reranker Results ==========\n")

for rank, result in enumerate(
    ranked_results,
    start=1
):

    chunk = result["chunk"]
    score = result["score"]

    print(f"--- Rank {rank} ---")

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