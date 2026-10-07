import sys
from pathlib import Path
import re

# Add ResearchRAG project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer


QUESTION = "Why use multiple attention heads?"


print("=" * 60)
print("Q14 HALLUCINATION DIAGNOSTIC")
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
# Citation validation
# ---------------------------------------------------------

valid_evidence_ids = {
    f"E{i}"
    for i in range(1, len(results) + 1)
}

cited_ids = set(
    re.findall(
        r"\[(E\d+)\]",
        answer
    )
)

invalid_ids = cited_ids - valid_evidence_ids


print("\n" + "=" * 60)
print("CITATION VALIDATION")
print("=" * 60)

print("Detected ResearchRAG citations:")
print(sorted(cited_ids))

print("\nInvalid ResearchRAG citations:")
print(sorted(invalid_ids))

if invalid_ids:
    print("\n❌ Invalid ResearchRAG citations detected.")
else:
    print("\n✅ No invalid ResearchRAG citations detected.")


# ---------------------------------------------------------
# Detect academic paper citations
# ---------------------------------------------------------

paper_citations = re.findall(
    r"\[(\d+)\]",
    answer
)

print("\n" + "=" * 60)
print("ACADEMIC CITATION CHECK")
print("=" * 60)

print("Paper-style numeric citations:")
print(paper_citations)

if paper_citations:
    print("\n⚠️ Paper-style citations detected.")
else:
    print("\n✅ No paper-style citations detected.")


# ---------------------------------------------------------
# Detect separate citation sections
# ---------------------------------------------------------

citation_section_patterns = [
    "Evidence citations",
    "Evidence Citations",
    "Citations:",
    "References:",
    "Sources:"
]

found_sections = [
    pattern
    for pattern in citation_section_patterns
    if pattern in answer
]

print("\n" + "=" * 60)
print("SEPARATE CITATION SECTION CHECK")
print("=" * 60)

print("Detected sections:")
print(found_sections)

if found_sections:
    print("\n❌ Separate citation/reference section detected.")
else:
    print("\n✅ No separate citation/reference section detected.")


# ---------------------------------------------------------
# Final diagnostic summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 60)

print("\nQuestion:")
print(QUESTION)

print("\nGenerated answer:")
print(answer)

print("\nResearchRAG citations:")
print(sorted(cited_ids))

print("\nInvalid citations:")
print(sorted(invalid_ids))

print("\nPaper-style citations:")
print(paper_citations)

print("\nSeparate citation sections:")
print(found_sections)

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)