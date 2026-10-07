import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer
from evaluation.evaluate_generation import judge_answer


QUESTION = "What does the paper show about English constituency parsing?"

REFERENCE_ANSWER = (
    "The paper shows that the Transformer can be applied successfully "
    "to English constituency parsing, including settings with both "
    "large and limited amounts of training data."
)


print("\n" + "=" * 70)
print("Q20 CITATION FORMAT DIAGNOSTIC TEST")
print("=" * 70)

print("\nQuestion:")
print(QUESTION)

print("\nReference answer:")
print(REFERENCE_ANSWER)


# ============================================================
# STEP 1 — LOAD RESEARCHRAG
# ============================================================

print("\n" + "-" * 70)
print("STEP 1: Loading ResearchRAG")
print("-" * 70)

retriever = Retriever()

print("\nResearchRAG loaded.")


# ============================================================
# STEP 2 — RETRIEVE EVIDENCE
# ============================================================

print("\n" + "-" * 70)
print("STEP 2: Retrieving evidence")
print("-" * 70)

results = retriever.retrieve(QUESTION)

print(f"\nRetrieved {len(results)} evidence chunks.")

for i, result in enumerate(results, start=1):

    chunk = result["chunk"]

    print("\n" + "=" * 70)
    print(f"EVIDENCE E{i}")
    print("=" * 70)

    print("Chunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page"])
    print("Reranker score:", result["score"])

    print("\nText:")
    print(chunk["text"])


# ============================================================
# STEP 3 — BUILD CONTEXT
# ============================================================

context_parts = []

for i, result in enumerate(results, start=1):

    chunk = result["chunk"]

    context_parts.append(
        f"""
[E{i}]
Source: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
    )

context = "\n".join(context_parts)


# ============================================================
# STEP 4 — GENERATE ANSWER
# ============================================================

print("\n" + "-" * 70)
print("STEP 3: Generating answer")
print("-" * 70)

generated_answer = generate_answer(
    question=QUESTION,
    context=context
)

print("\nGenerated answer:")
print(generated_answer)


# ============================================================
# STEP 5 — STRICT CITATION FORMAT CHECK
# ============================================================

print("\n" + "-" * 70)
print("STEP 4: Checking citation format")
print("-" * 70)

print("\nAll bracketed references found:")

all_bracket_references = re.findall(
    r"\[[^\]]+\]",
    generated_answer
)

if all_bracket_references:
    for reference in all_bracket_references:
        print(reference)
else:
    print("None")


print("\nValid ResearchRAG citations:")

research_rag_citations = re.findall(
    r"\[E\d+\]",
    generated_answer
)

if research_rag_citations:
    print(research_rag_citations)
else:
    print("NO [E-number] CITATIONS FOUND")


print("\nPotential paper-style citations:")

paper_citations = []

for reference in all_bracket_references:

    if not re.fullmatch(
        r"\[E\d+\]",
        reference
    ):
        paper_citations.append(reference)


if paper_citations:
    print(paper_citations)
else:
    print("None")


# ============================================================
# STEP 6 — CHECK INVALID E-NUMBERS
# ============================================================

valid_ids = {
    f"E{i}"
    for i in range(1, len(results) + 1)
}

invalid_citations = []

for citation in research_rag_citations:

    citation_id = citation[1:-1]

    if citation_id not in valid_ids:
        invalid_citations.append(citation)


print("\nInvalid ResearchRAG citations:")

if invalid_citations:
    print(invalid_citations)
else:
    print("None")


# ============================================================
# STEP 7 — CHECK FOR MARKDOWN LINK CITATIONS
# ============================================================

markdown_citations = re.findall(
    r"\[E\d+\]\([^)]+\)",
    generated_answer
)

print("\nMarkdown-style citations:")

if markdown_citations:
    print(markdown_citations)
else:
    print("None")


# ============================================================
# STEP 8 — JUDGE
# ============================================================

print("\n" + "-" * 70)
print("STEP 5: Judging answer")
print("-" * 70)

print("\nSending the answer to the evaluator...")

judge_output = judge_answer(
    question=QUESTION,
    reference_answer=REFERENCE_ANSWER,
    generated_answer=generated_answer,
    evidence_results=results
)

print("\nJudge output:")
print(judge_output)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("Q20 DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\nWhat we are investigating:")
print("1. Does the answer correctly answer the question?")
print("2. Are ResearchRAG citations formatted as [E1], [E2], etc.?")
print("3. Does the answer accidentally use paper citation numbers?")
print("4. Does the answer use markdown-style citations?")
print("5. Are there invalid evidence IDs?")
print("6. Does the judge correctly evaluate the answer?")