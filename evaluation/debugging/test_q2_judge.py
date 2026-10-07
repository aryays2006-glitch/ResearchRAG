import sys
from pathlib import Path

# Allow imports from the ResearchRAG project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from api.retriever import Retriever
from api.generator import generate_answer
from evaluation.evaluate_generation import judge_answer


QUESTION = "What is the main architecture proposed in the paper?"

REFERENCE_ANSWER = (
    "The paper proposes the Transformer architecture, which uses "
    "stacked self-attention and point-wise fully connected layers "
    "for the encoder and decoder."
)


print("\n" + "=" * 70)
print("Q2 DIAGNOSTIC TEST")
print("=" * 70)

print("\nQuestion:")
print(QUESTION)

print("\nReference answer:")
print(REFERENCE_ANSWER)


# ---------------------------------------------------------
# STEP 1: Load ResearchRAG
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 1: Loading ResearchRAG")
print("-" * 70)

retriever = Retriever()

print("\nResearchRAG loaded.")


# ---------------------------------------------------------
# STEP 2: Retrieve evidence
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# STEP 3: Build evidence context
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# STEP 4: Generate answer
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 3: Generating answer")
print("-" * 70)

generated_answer = generate_answer(
    question=QUESTION,
    context=context
)

print("\nGenerated answer:")
print(generated_answer)


# ---------------------------------------------------------
# STEP 5: Judge the answer
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 4: Judging answer")
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


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("Q2 DIAGNOSTIC COMPLETE")
print("=" * 70)