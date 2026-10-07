import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer
from evaluation.evaluate_generation import judge_answer


QUESTION = "What is multi-head attention?"

REFERENCE_ANSWER = (
    "Multi-head attention consists of several attention layers running "
    "in parallel and allows the model to jointly attend to information "
    "from different representation subspaces at different positions."
)


print("\n" + "=" * 70)
print("Q4 CITATION DIAGNOSTIC TEST")
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
# STEP 3 — BUILD EVIDENCE CONTEXT
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
# STEP 5 — CHECK CITATION FORMAT
# ============================================================

print("\n" + "-" * 70)
print("STEP 4: Checking citation format")
print("-" * 70)

citations = re.findall(
    r"\[(E\d+)\]",
    generated_answer
)

print("\nDetected ResearchRAG citations:")

if citations:
    print(citations)
else:
    print("NO [E-number] CITATIONS FOUND")


invalid_citations = []

valid_ids = {
    f"E{i}"
    for i in range(1, len(results) + 1)
}

for citation in citations:

    if citation not in valid_ids:
        invalid_citations.append(citation)


if invalid_citations:

    print("\nInvalid citations:")
    print(invalid_citations)

else:

    print("\nInvalid citations:")
    print("None")


# ============================================================
# STEP 6 — JUDGE ANSWER
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
print("Q4 DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\nWhat we are investigating:")
print("1. Does the answer correctly explain multi-head attention?")
print("2. Does every factual claim have a ResearchRAG citation?")
print("3. Are citations in the required [E1] format?")
print("4. Does each citation actually support the claim?")
print("5. Does the judge correctly evaluate the generated answer?")