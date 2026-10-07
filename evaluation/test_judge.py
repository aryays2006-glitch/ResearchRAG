import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from evaluation.evaluate_generation import judge_answer


# ============================================================
# TEST DATA
# ============================================================

TESTS = [

    {
        "name": "Question 9",

        "question":
            "How does the Transformer use residual connections?",

        "reference_answer":
            "Residual connections are used around each of the two "
            "sub-layers, followed by layer normalization.",

        "generated_answer":
            "The provided evidence does not explicitly mention "
            "how the Transformer uses residual connections."
    },

    {
        "name": "Question 18",

        "question":
            "What dropout rate is used in the Transformer base model?",

        "reference_answer":
            "The Transformer base model uses a dropout rate of 0.1.",

        "generated_answer":
            "The dropout rate used in the Transformer base model "
            "is 0.1. [E4]"
    }
]


# ============================================================
# FAKE EVIDENCE
# ============================================================

EVIDENCE_RESULTS = [

    {
        "chunk": {
            "chunk_id": "test_p3_c1",
            "source": "Attention Is All You Need.pdf",
            "page": 3,
            "text":
                "Each of the two sub-layers has a residual connection "
                "around it followed by layer normalization."
        },
        "score": 1.0
    },

    {
        "chunk": {
            "chunk_id": "test_p8_c1",
            "source": "Attention Is All You Need.pdf",
            "page": 8,
            "text":
                "Table 3 shows the variations on the Transformer "
                "architecture. The base model uses a dropout rate "
                "of 0.1."
        },
        "score": 1.0
    }
]


# ============================================================
# RUN TEST
# ============================================================

print("\n" + "=" * 60)
print("FAST JUDGE-ONLY TEST")
print("=" * 60)


for test in TESTS:

    print("\n" + "-" * 60)

    print(
        f"Testing: {test['name']}"
    )

    print("-" * 60)

    print(
        "\nQuestion:"
    )

    print(
        test["question"]
    )

    print(
        "\nGenerated answer:"
    )

    print(
        test["generated_answer"]
    )

    print(
        "\nRunning judge..."
    )

    raw_output = judge_answer(

        question=test["question"],

        reference_answer=test["reference_answer"],

        generated_answer=test["generated_answer"],

        evidence_results=EVIDENCE_RESULTS
    )

    print(
        "\nJudge output:"
    )

    print(
        raw_output
    )


print("\n" + "=" * 60)
print("FAST JUDGE TEST COMPLETE")
print("=" * 60)