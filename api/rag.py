import re

from api.generator import generate_answer
from api.retriever import Retriever


print("Loading ResearchRAG...")

retriever = Retriever()

print("ResearchRAG loaded successfully.")


def ask_research_rag(question: str):

    # --------------------------------------------------
    # 1. Retrieve and rerank evidence
    # --------------------------------------------------

    final_results = retriever.retrieve(
        question
    )


    # --------------------------------------------------
    # 2. Build evidence context
    # --------------------------------------------------

    context_parts = []

    for i, result in enumerate(
        final_results,
        start=1
    ):

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

    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------
    # 3. Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        question=question,
        context=context
    )


    # --------------------------------------------------
    # 4. Validate citations
    # --------------------------------------------------

    valid_evidence_ids = {
        f"E{i}"
        for i in range(
            1,
            len(final_results) + 1
        )
    }

    cited_ids = set(
        re.findall(
            r"\[(E\d+)\]",
            answer
        )
    )

    invalid_ids = (
        cited_ids
        - valid_evidence_ids
    )


    # --------------------------------------------------
    # 5. Build source information
    # --------------------------------------------------

    sources = []

    for i, result in enumerate(
        final_results,
        start=1
    ):

        chunk = result["chunk"]

        sources.append(
            {
                "evidence_id": f"E{i}",
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "reranker_score": result["score"]
            }
        )


    # --------------------------------------------------
    # 6. Return final response
    # --------------------------------------------------

    return {
        "question": question,
        "answer": answer,
        "citations": sorted(cited_ids),
        "invalid_citations": sorted(invalid_ids),
        "sources": sources
    }