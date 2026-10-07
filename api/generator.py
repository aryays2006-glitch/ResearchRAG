import re

import ollama


LLM_MODEL = "qwen3:8b"


def get_evidence_ids(context: str) -> list[str]:
    """
    Get all valid evidence IDs from the retrieved context.
    """

    evidence_ids = []

    for match in re.findall(
        r"\[(E\d+)\]",
        context
    ):

        if match not in evidence_ids:
            evidence_ids.append(match)

    return evidence_ids


def extract_evidence_ids(text: str) -> list[str]:
    """
    Extract evidence IDs such as E1, E2, E3 from text.
    """

    evidence_ids = []

    for match in re.findall(
        r"\[(E\d+)\]",
        text
    ):

        if match not in evidence_ids:
            evidence_ids.append(match)

    return evidence_ids


def generate_answer(
    question: str,
    context: str
) -> str:

    valid_evidence_ids = get_evidence_ids(
        context
    )

    if not valid_evidence_ids:

        return (
            "The provided evidence is insufficient "
            "to answer this question."
        )

    prompt = f"""
You are an AI research assistant answering questions about a research paper.

Answer the user's question using ONLY the evidence provided below.

Rules:

1. Use only the provided evidence.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Give a concise, direct answer.
5. After the answer, list the evidence IDs that support your answer.
6. Evidence IDs look like [E1], [E2], [E3], etc.
7. Only use evidence IDs that actually appear in the provided evidence.
8. If multiple evidence blocks support the answer, you may cite multiple IDs.

Use this format:

ANSWER:
<your answer>

EVIDENCE:
[E1] [E2]

====================
EVIDENCE
====================

{context}

====================
QUESTION
====================

{question}

====================
RESPONSE
====================
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output = response[
        "message"
    ][
        "content"
    ].strip()

    if not output:

        return (
            "The provided evidence is insufficient "
            "to answer this question."
        )

    # ---------------------------------------------------------
    # Try to extract the ANSWER section
    # ---------------------------------------------------------

    answer_match = re.search(
        r"ANSWER:\s*(.*?)(?:\n\s*EVIDENCE:|$)",
        output,
        re.IGNORECASE | re.DOTALL
    )

    if answer_match:

        answer = answer_match.group(1).strip()

    else:

        # If Qwen did not follow the exact format,
        # use the whole response as the answer.
        answer = output

    # ---------------------------------------------------------
    # Extract evidence IDs from the generated response
    # ---------------------------------------------------------

    cited_evidence = extract_evidence_ids(
        output
    )

    valid_citations = [
        evidence_id
        for evidence_id in cited_evidence
        if evidence_id in valid_evidence_ids
    ]

    # ---------------------------------------------------------
    # If Qwen did not provide citations,
    # use the top retrieved evidence.
    # ---------------------------------------------------------

    if not valid_citations:

        valid_citations = [
            valid_evidence_ids[0]
        ]

    # ---------------------------------------------------------
    # Remove evidence IDs from the answer itself
    # ---------------------------------------------------------

    answer = re.sub(
        r"\[(E\d+)\]",
        "",
        answer
    ).strip()

    # Remove accidental ANSWER / EVIDENCE headings
    answer = re.sub(
        r"^ANSWER:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    answer = re.sub(
        r"\s*EVIDENCE:\s*$",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    if not answer:

        return (
            "The provided evidence is insufficient "
            "to answer this question."
        )

    citations = " ".join(
        f"[{evidence_id}]"
        for evidence_id in valid_citations
    )

    return f"{answer} {citations}"