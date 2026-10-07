import json
import re


# ============================================================
# CITATION VALIDATION
# ============================================================

def extract_citation_ids(answer):
    """
    Extract citation IDs such as [E1], [E2], [E3]
    from a generated answer.
    """

    return sorted(
        set(
            re.findall(
                r"\[(E\d+)\]",
                answer
            )
        )
    )


def validate_citations(
    answer,
    evidence_count
):
    """
    Check whether every citation used by the answer
    refers to an evidence item that actually exists.
    """

    valid_evidence_ids = {
        f"E{i}"
        for i in range(
            1,
            evidence_count + 1
        )
    }

    cited_ids = set(
        extract_citation_ids(answer)
    )

    invalid_ids = (
        cited_ids
        - valid_evidence_ids
    )

    return {
        "cited_ids": sorted(cited_ids),
        "invalid_ids": sorted(invalid_ids),
        "citation_valid": len(invalid_ids) == 0
    }


# ============================================================
# EXTRACT JSON FROM LLM RESPONSE
# ============================================================

def extract_json(text):
    """
    Extract a JSON object from an LLM response.

    Handles responses where the model returns JSON
    inside a markdown code block.
    """

    # --------------------------------------------------------
    # First try the entire response
    # --------------------------------------------------------

    try:

        return json.loads(
            text.strip()
        )

    except json.JSONDecodeError:
        pass


    # --------------------------------------------------------
    # Try to find a JSON object inside the response
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:

        return None


    try:

        return json.loads(
            match.group()
        )

    except json.JSONDecodeError:

        return None


# ============================================================
# VALIDATE JUDGE OUTPUT
# ============================================================

def validate_judge_output(
    evaluation
):
    """
    Validate the structure and values returned
    by the LLM evaluator.
    """

    required_fields = [
        "faithfulness",
        "answer_relevance",
        "citation_correctness",
        "reason"
    ]


    # --------------------------------------------------------
    # Check that all fields exist
    # --------------------------------------------------------

    for field in required_fields:

        if field not in evaluation:

            return {
                "valid": False,
                "error": (
                    f"Missing required field: {field}"
                )
            }


    # --------------------------------------------------------
    # Check score values
    # --------------------------------------------------------

    score_fields = [
        "faithfulness",
        "answer_relevance",
        "citation_correctness"
    ]

    for field in score_fields:

        value = evaluation[field]

        if not isinstance(
            value,
            (int, float)
        ):

            return {
                "valid": False,
                "error": (
                    f"{field} must be a number."
                )
            }


        if not 0.0 <= float(value) <= 1.0:

            return {
                "valid": False,
                "error": (
                    f"{field} must be between "
                    "0.0 and 1.0."
                )
            }


    # --------------------------------------------------------
    # Check reason
    # --------------------------------------------------------

    if not isinstance(
        evaluation["reason"],
        str
    ):

        return {
            "valid": False,
            "error": "reason must be a string."
        }


    # --------------------------------------------------------
    # Everything is valid
    # --------------------------------------------------------

    return {
        "valid": True,
        "error": None
    }


# ============================================================
# CALCULATE OVERALL SCORE
# ============================================================

def calculate_overall_score(
    faithfulness,
    answer_relevance,
    citation_correctness
):
    """
    Calculate the average of the three main
    generation quality metrics.
    """

    return (
        faithfulness
        + answer_relevance
        + citation_correctness
    ) / 3


# ============================================================
# DETERMINE PASS / FAIL
# ============================================================

def determine_pass(
    faithfulness,
    answer_relevance,
    citation_correctness,
    citation_valid,
    threshold=0.80
):
    """
    Determine whether an answer passes the
    generation quality threshold.

    Every quality score must meet the threshold,
    and all citation IDs must be valid.
    """

    return (
        faithfulness >= threshold
        and answer_relevance >= threshold
        and citation_correctness >= threshold
        and citation_valid
    )


# ============================================================
# BUILD EVALUATION RESULT
# ============================================================

def build_evaluation_result(
    question,
    generated_answer,
    citation_result,
    evaluation,
    threshold=0.80
):
    """
    Build one standardized evaluation result.
    """

    faithfulness = float(
        evaluation["faithfulness"]
    )

    answer_relevance = float(
        evaluation["answer_relevance"]
    )

    citation_correctness = float(
        evaluation["citation_correctness"]
    )

    overall_score = calculate_overall_score(
        faithfulness,
        answer_relevance,
        citation_correctness
    )

    passed = determine_pass(
        faithfulness,
        answer_relevance,
        citation_correctness,
        citation_result["citation_valid"],
        threshold
    )

    return {
        "question": question,
        "generated_answer": generated_answer,
        "cited_ids": citation_result[
            "cited_ids"
        ],
        "invalid_citations": citation_result[
            "invalid_ids"
        ],
        "citation_valid": citation_result[
            "citation_valid"
        ],
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "citation_correctness": citation_correctness,
        "overall_score": overall_score,
        "passed": passed,
        "reason": evaluation["reason"]
    }


# ============================================================
# CALCULATE DATASET SUMMARY
# ============================================================

def calculate_dataset_summary(
    results,
    threshold=0.80
):
    """
    Calculate overall evaluation statistics
    across the entire golden dataset.
    """

    if not results:

        return {
            "questions_evaluated": 0,
            "average_faithfulness": 0.0,
            "average_answer_relevance": 0.0,
            "average_citation_correctness": 0.0,
            "average_overall_score": 0.0,
            "passed_questions": 0,
            "failed_questions": 0,
            "pass_rate": 0.0
        }


    question_count = len(results)


    average_faithfulness = (
        sum(
            result["faithfulness"]
            for result in results
        )
        / question_count
    )


    average_relevance = (
        sum(
            result["answer_relevance"]
            for result in results
        )
        / question_count
    )


    average_citation_correctness = (
        sum(
            result["citation_correctness"]
            for result in results
        )
        / question_count
    )


    average_overall_score = (
        sum(
            result["overall_score"]
            for result in results
        )
        / question_count
    )


    passed_questions = sum(
        1
        for result in results
        if result["passed"]
    )


    failed_questions = (
        question_count
        - passed_questions
    )


    pass_rate = (
        passed_questions
        / question_count
    )


    return {
        "questions_evaluated": question_count,
        "average_faithfulness": average_faithfulness,
        "average_answer_relevance": average_relevance,
        "average_citation_correctness": (
            average_citation_correctness
        ),
        "average_overall_score": (
            average_overall_score
        ),
        "passed_questions": passed_questions,
        "failed_questions": failed_questions,
        "pass_rate": pass_rate,
        "threshold": threshold
    }


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    summary
):
    """
    Print a clean final evaluation report.
    """

    print(
        "\n=================================================="
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=================================================="
    )

    print(
        f"Questions evaluated: "
        f"{summary['questions_evaluated']}"
    )

    print(
        f"Average Faithfulness: "
        f"{summary['average_faithfulness']:.2f}"
    )

    print(
        f"Average Answer Relevance: "
        f"{summary['average_answer_relevance']:.2f}"
    )

    print(
        f"Average Citation Correctness: "
        f"{summary['average_citation_correctness']:.2f}"
    )

    print(
        f"Average Overall Score: "
        f"{summary['average_overall_score']:.2f}"
    )

    print(
        f"Passed Questions: "
        f"{summary['passed_questions']}"
    )

    print(
        f"Failed Questions: "
        f"{summary['failed_questions']}"
    )

    print(
        f"Pass Rate: "
        f"{summary['pass_rate'] * 100:.1f}%"
    )

    print(
        f"Quality Threshold: "
        f"{summary['threshold']:.2f}"
    )


    if (
        summary["failed_questions"] == 0
        and summary["average_overall_score"]
        >= summary["threshold"]
    ):

        print(
            "\n✅ EVALUATION PASSED"
        )

    else:

        print(
            "\n❌ EVALUATION FAILED"
        )