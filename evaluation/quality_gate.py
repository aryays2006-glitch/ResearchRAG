QUALITY_THRESHOLDS = {
    "retrieval_recall_at_5": 0.80,
    "faithfulness": 0.80,
    "answer_relevance": 0.80,
    "citation_correctness": 0.80,
}

def check_quality_gate(
    average_recall_at_5,
    average_faithfulness,
    average_answer_relevance,
    average_citation_correctness,
    invalid_citations
):
    checks = {
        "retrieval_recall_at_5": average_recall_at_5 >= QUALITY_THRESHOLDS["retrieval_recall_at_5"],
        "faithfulness": average_faithfulness >= QUALITY_THRESHOLDS["faithfulness"],
        "answer_relevance": average_answer_relevance >= QUALITY_THRESHOLDS["answer_relevance"],
        "citation_correctness": average_citation_correctness >= QUALITY_THRESHOLDS["citation_correctness"],
        "no_invalid_citations": invalid_citations == 0,
    }

    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks,
        "thresholds": QUALITY_THRESHOLDS,
    }


if __name__ == "__main__":
    print("ResearchRAG Quality Gate")
    print("========================")

    result = check_quality_gate(
        average_recall_at_5=1.00,
        average_faithfulness=1.00,
        average_answer_relevance=1.00,
        average_citation_correctness=1.00,
        invalid_citations=0,
    )

    print("\nQuality Checks:")

    for check_name, passed in result["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")

    print("\nFinal Quality Gate:")

    if result["passed"]:
        print("PASS")
    else:
        print("FAIL")