import sys
from pathlib import Path

# Add ResearchRAG project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer


QUESTION = "What is the computational complexity of self-attention compared with recurrent layers?"


print("=" * 60)
print("Q20 CITATION DIAGNOSTIC")
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
# Citation-format checks
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CITATION FORMAT CHECK")
print("=" * 60)

import re


# Correct ResearchRAG citation format
valid_citations = re.findall(r"\[E\d+\]", answer)

print("\nValid evidence citations found:")
print(valid_citations)


# Detect academic-style citations such as [1], [29]
academic_citations = re.findall(
    r"(?<!E)\[\d+\]",
    answer
)

print("\nAcademic-style citations found:")
print(academic_citations)


# Detect markdown-style citations such as [E2](#e2)
markdown_citations = re.findall(
    r"\[E\d+\]\([^)]+\)",
    answer
)

print("\nMarkdown-style citations found:")
print(markdown_citations)


# Detect separate citation/reference sections
citation_section_patterns = [
    r"(?im)^\s*#{0,6}\s*(citations?|references?|sources?|evidence citations?)\s*:?\s*$",
    r"(?im)^\s*(citations?|references?|sources?|evidence citations?)\s*:?\s*$"
]

citation_section_found = False

for pattern in citation_section_patterns:
    if re.search(pattern, answer):
        citation_section_found = True
        break

print("\nSeparate citation/reference section found:")
print(citation_section_found)


# ---------------------------------------------------------
# Final verdict
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Q20 DIAGNOSTIC VERDICT")
print("=" * 60)

if (
    valid_citations
    and not academic_citations
    and not markdown_citations
    and not citation_section_found
):
    print("\nPASS: Citation format is clean.")
else:
    print("\nFAIL: Citation-format issue detected.")

print("\n" + "=" * 60)
print("Q20 DIAGNOSTIC COMPLETE")
print("=" * 60)