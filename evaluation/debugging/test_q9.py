import sys
from pathlib import Path

# Add ResearchRAG project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer


QUESTION = "What is one advantage of self-attention over recurrent layers?"


print("=" * 60)
print("Q9 CITATION DIAGNOSTIC")
print("=" * 60)

print("\nQuestion:")
print(QUESTION)


# ---------------------------------------------------------
# Load retriever
# ---------------------------------------------------------

print("\nLoading retriever...")

retriever = Retriever()

print("Retriever loaded successfully.")


# ---------------------------------------------------------
# Retrieve evidence
# ---------------------------------------------------------

print("\nRetrieving evidence...")

results = retriever.retrieve(QUESTION)

print(f"Retrieved {len(results)} evidence chunks.")


# ---------------------------------------------------------
# Build evidence context
# ---------------------------------------------------------

context_parts = []

for i, result in enumerate(results, start=1):

    evidence_id = f"E{i}"
    chunk = result["chunk"]

    context_parts.append(
        f"""
[{evidence_id}]
Source: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
    )

context = "\n".join(context_parts)


# ---------------------------------------------------------
# Display retrieved evidence
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVED EVIDENCE")
print("=" * 60)

for i, result in enumerate(results, start=1):

    evidence_id = f"E{i}"
    chunk = result["chunk"]

    print(f"\n[{evidence_id}]")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Reranker score: {result['score']:.4f}")

    print("-" * 60)
    print(chunk["text"])


# ---------------------------------------------------------
# Generate answer
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GENERATING ANSWER")
print("=" * 60)

answer = generate_answer(
    question=QUESTION,
    context=context
)

print("\nGenerated answer:")
print("-" * 60)
print(answer)


# ---------------------------------------------------------
# Final diagnostic
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Q9 DIAGNOSTIC COMPLETE")
print("=" * 60)