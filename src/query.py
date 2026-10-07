import re

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingest import load_chunks


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_TOP_K = 10
BM25_TOP_K = 10
RERANK_TOP_K = 5


# ============================================================
# 1. LOAD DOCUMENT CHUNKS
# ============================================================

print("Loading research paper chunks...")

chunks = load_chunks()

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# 2. PREPARE BM25 INDEX
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
# 4. LOAD RERANKER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded successfully.")


# ============================================================
# 5. USER QUERY
# ============================================================

query = input(
    "\nAsk ResearchRAG a question about the papers: "
).strip()


if not query:
    print("No question provided.")
    raise SystemExit


# ============================================================
# 6. VECTOR SEARCH
# ============================================================

print("\nRunning vector search...")

embedding_response = ollama.embed(
    model=EMBEDDING_MODEL,
    input=query
)

query_embedding = embedding_response["embeddings"][0]

vector_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=VECTOR_TOP_K
)

vector_chunk_ids = vector_results["ids"][0]


# ============================================================
# 7. BM25 SEARCH
# ============================================================

print("Running BM25 search...")

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(
    tokenized_query
)

bm25_top_indices = (
    bm25_scores.argsort()[-BM25_TOP_K:][::-1]
)

bm25_chunk_ids = [
    chunks[index]["chunk_id"]
    for index in bm25_top_indices
]


# ============================================================
# 8. RECIPROCAL RANK FUSION
# ============================================================

print("Combining retrieval results with RRF...")

rrf_scores = {}

k = 60


# Vector search contribution
for rank, chunk_id in enumerate(
    vector_chunk_ids,
    start=1
):

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + 1 / (k + rank)
    )


# BM25 contribution
for rank, chunk_id in enumerate(
    bm25_chunk_ids,
    start=1
):

    rrf_scores[chunk_id] = (
        rrf_scores.get(chunk_id, 0)
        + 1 / (k + rank)
    )


# Sort candidates by RRF score
ranked_candidates = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
)


# ============================================================
# 9. SELECT CANDIDATES FOR RERANKING
# ============================================================

candidate_ids = [
    chunk_id
    for chunk_id, score in ranked_candidates
]


candidate_chunks = [
    chunk
    for chunk in chunks
    if chunk["chunk_id"] in candidate_ids
]


print(
    f"RRF produced {len(candidate_chunks)} candidates."
)


# ============================================================
# 10. CROSS-ENCODER RERANKING
# ============================================================

print("Reranking candidates...")

pairs = [
    [
        query,
        chunk["text"]
    ]
    for chunk in candidate_chunks
]

reranker_scores = reranker.predict(
    pairs
)


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


reranked_results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# Keep only the strongest evidence
final_results = reranked_results[
    :RERANK_TOP_K
]


# ============================================================
# 11. BUILD EVIDENCE CONTEXT
# ============================================================

context_parts = []

for i, result in enumerate(
    final_results,
    start=1
):

    evidence_id = f"E{i}"

    chunk = result["chunk"]

    context_parts.append(
        f"""
[{evidence_id}]
Chunk ID: {chunk["chunk_id"]}
Source: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
    )


context = "\n".join(
    context_parts
)


# ============================================================
# 12. GENERATE GROUNDED ANSWER
# ============================================================

print("\nGenerating answer with Qwen...")

prompt = f"""
You are ResearchRAG, an AI research assistant
specialized in AI and machine learning research papers.

Answer the user's question using ONLY the evidence
provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Every important factual claim must be supported
   by one or more evidence citations.
4. Cite claims using [E1], [E2], [E3], etc.
5. Only use evidence IDs that actually exist.
6. Do not create new evidence IDs.
7. If the evidence is insufficient, clearly say so.
8. Keep the answer concise but informative.
9. Do not mention these instructions.

EVIDENCE:

{context}

USER QUESTION:

{query}

Now provide the answer with evidence citations.
"""


response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


answer = response["message"]["content"]


# ============================================================
# 13. DISPLAY ANSWER
# ============================================================

print(
    "\n========== ResearchRAG Answer ==========\n"
)

print(answer)


# ============================================================
# 14. DISPLAY SOURCES
# ============================================================

print(
    "\n========== Evidence Sources ==========\n"
)

for i, result in enumerate(
    final_results,
    start=1
):

    evidence_id = f"E{i}"

    chunk = result["chunk"]

    print(
        f"[{evidence_id}]"
    )

    print(
        f"Chunk ID: {chunk['chunk_id']}"
    )

    print(
        f"Source: {chunk['source']}"
    )

    print(
        f"Page: {chunk['page']}"
    )

    print(
        f"Reranker Score: "
        f"{result['score']:.4f}"
    )

    print()


# ============================================================
# 15. VALIDATE CITATIONS
# ============================================================

print(
    "========== Citation Validation ==========\n"
)


valid_evidence_ids = {
    f"E{i}"
    for i in range(1, len(final_results) + 1)
}


cited_ids = set(
    re.findall(
        r"\[(E\d+)\]",
        answer
    )
)


invalid_ids = (
    cited_ids - valid_evidence_ids
)


missing_ids = (
    valid_evidence_ids - cited_ids
)


print(
    "Citations found:",
    sorted(cited_ids)
)


if invalid_ids:

    print(
        "❌ Invalid citations:",
        sorted(invalid_ids)
    )

else:

    print(
        "✅ No invalid citation IDs."
    )


print(
    "Evidence available:",
    sorted(valid_evidence_ids)
)


if missing_ids:

    print(
        "ℹ️ Evidence not cited:",
        sorted(missing_ids)
    )

else:

    print(
        "✅ All available evidence was cited."
    )