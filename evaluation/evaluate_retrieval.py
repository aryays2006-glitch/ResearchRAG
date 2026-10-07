import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from api.retriever import Retriever

# ============================================================
# PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

import json


print("Loading golden dataset...")

with open(
    PROJECT_ROOT / "evaluation" / "golden_dataset.json",
    "r",
    encoding="utf-8"
) as file:

    golden_dataset = json.load(file)


print(
    f"Loaded {len(golden_dataset)} evaluation questions."
)


# ============================================================
# LOAD RESEARCHRAG RETRIEVER
# ============================================================

print("\nLoading ResearchRAG retriever...")

retriever = Retriever()

print("Retriever ready.")


# ============================================================
# EVALUATE EACH QUESTION
# ============================================================

results = []

print(
    "\n=================================================="
)

print(
    "STARTING RETRIEVAL EVALUATION"
)

print(
    "==================================================\n"
)


for question_number, item in enumerate(
    golden_dataset,
    start=1
):

    question = item["question"]

    expected_chunk_ids = {
        chunk_id.lower()
        for chunk_id in item["expected_chunk_ids"]
    }


    print(
        f"Question {question_number}: {question}"
    )


    # ========================================================
    # RUN THE SAME RETRIEVAL PIPELINE USED BY THE API
    # ========================================================

    retrieved_results = retriever.retrieve(
        question
    )


    retrieved_chunk_ids = [
        result["chunk"]["chunk_id"].lower()
        for result in retrieved_results
    ]


    # ========================================================
    # CALCULATE RECALL@5
    # ========================================================

    top_k_chunk_ids = set(
        retrieved_chunk_ids[:5]
    )


    retrieved_expected = (
        expected_chunk_ids
        & top_k_chunk_ids
    )


    recall_at_5 = (
        len(retrieved_expected)
        / len(expected_chunk_ids)
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    result = {
        "question": question,
        "expected_chunks": list(
            expected_chunk_ids
        ),
        "retrieved_chunks": retrieved_chunk_ids[:5],
        "recall_at_5": recall_at_5
    }


    results.append(result)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        f"Expected chunks: "
        f"{list(expected_chunk_ids)}"
    )


    print(
        f"Retrieved top 5: "
        f"{retrieved_chunk_ids[:5]}"
    )


    print(
        f"Recall@5: "
        f"{recall_at_5:.2f}"
    )


    if recall_at_5 == 1.0:

        print(
            "✅ All expected chunks retrieved."
        )

    elif recall_at_5 > 0:

        print(
            "⚠️ Some expected chunks retrieved."
        )

    else:

        print(
            "❌ No expected chunks retrieved."
        )


    print()


# ============================================================
# CALCULATE OVERALL SCORE
# ============================================================

average_recall = (
    sum(
        result["recall_at_5"]
        for result in results
    )
    / len(results)
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "=================================================="
)

print(
    "FINAL RETRIEVAL EVALUATION"
)

print(
    "=================================================="
)


print(
    f"Questions evaluated: {len(results)}"
)


print(
    f"Average Recall@5: {average_recall:.2f}"
)


if average_recall >= 0.80:

    print(
        "✅ Retrieval quality is currently acceptable."
    )

else:

    print(
        "⚠️ Retrieval quality needs improvement."
    )