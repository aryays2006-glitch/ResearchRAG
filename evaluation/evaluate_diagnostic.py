import json
import re
import sys
from pathlib import Path

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)

sys.path.append(
    str(PROJECT_ROOT / "evaluation")
)


from ingest import load_chunks

from evaluation_utils import (
    build_evaluation_result,
    extract_json,
    validate_citations,
    validate_judge_output,
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_TOP_K = 10
BM25_TOP_K = 10
RERANK_TOP_K = 5

RRF_K = 60

QUALITY_THRESHOLD = 0.80


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

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
# DIAGNOSTIC QUESTIONS
# ============================================================

# We deliberately test three different questions:
#
# Q4  -> previously failed because of citation correctness
# Q12 -> factual/numeric question
# Q19 -> comparison/performance question

QUESTION_NUMBERS = [4, 12, 19]


print("\n==================================================")
print("3-QUESTION GENERATION DIAGNOSTIC")
print("==================================================")
print(
    f"Questions being tested: {QUESTION_NUMBERS}"
)
print("==================================================\n")


# ============================================================
# LOAD DOCUMENT CHUNKS
# ============================================================

print("Loading document chunks...")

chunks = load_chunks()

print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# PREPARE BM25
# ============================================================

documents = [
    chunk["text"]
    for chunk in chunks
]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(
    tokenized_documents
)


# ============================================================
# CONNECT TO CHROMA
# ============================================================

print("\nConnecting to Chroma...")

client = chromadb.PersistentClient(
    path=str(PROJECT_ROOT / "chroma_db")
)

collection = client.get_collection(
    name="research_papers"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded successfully.")


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(query):

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = (
        embedding_response["embeddings"][0]
    )

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=VECTOR_TOP_K
    )

    vector_chunk_ids = (
        vector_results["ids"][0]
    )


    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    tokenized_query = (
        query.lower().split()
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_top_indices = (
        bm25_scores
        .argsort()[-BM25_TOP_K:][::-1]
    )

    bm25_chunk_ids = [
        chunks[index]["chunk_id"]
        for index in bm25_top_indices
    ]


    # --------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    rrf_scores = {}

    for rank, chunk_id in enumerate(
        vector_chunk_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    for rank, chunk_id in enumerate(
        bm25_chunk_ids,
        start=1
    ):

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )


    ranked_candidates = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # --------------------------------------------------------
    # TOP 10 CANDIDATES
    # --------------------------------------------------------

    candidate_ids = [
        chunk_id
        for chunk_id, score
        in ranked_candidates[:10]
    ]

    candidate_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] in candidate_ids
    ]


    # --------------------------------------------------------
    # CROSS-ENCODER RERANKING
    # --------------------------------------------------------

    pairs = [
        [
            query,
            chunk["text"]
        ]
        for chunk in candidate_chunks
    ]

    reranker_scores = reranker.predict(
        pairs
    )

    reranked_results = []

    for chunk, score in zip(
        candidate_chunks,
        reranker_scores
    ):

        reranked_results.append(
            {
                "chunk": chunk,
                "score": float(score)
            }
        )


    reranked_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    return reranked_results[:RERANK_TOP_K]


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    evidence_results
):

    context_parts = []

    for i, result in enumerate(
        evidence_results,
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


    prompt = f"""
You are ResearchRAG, an AI research assistant
specialized in AI and machine learning research papers.

Your task is to answer the user's question using
ONLY the evidence provided below.

IMPORTANT:
The evidence is the ONLY source of truth.
Do not use your own knowledge, training knowledge,
or assumptions.

STRICT RULES:

1. Use ONLY information explicitly stated in the evidence.

2. Do NOT add facts that are not present in the evidence.

3. Do NOT explain concepts using outside knowledge.

4. Every factual statement must have a citation.

5. A citation must directly support the statement
   immediately before it.

6. Choose the citation based on the actual text of
   the evidence.

7. NEVER cite an evidence item merely because it is
   related to the topic.

8. Do NOT combine claims from different evidence items
   under one citation unless that evidence actually
   supports all of those claims.

9. If a claim cannot be supported by the evidence,
   DO NOT make the claim.

10. If the evidence does not contain enough information
    to answer the question, say:

    "The provided evidence is insufficient to answer
    this question."

11. Citations MUST use exactly this format:

    [E1]
    [E2]
    [E3]

12. Only use evidence IDs that actually appear in the
    provided evidence.

13. NEVER create new evidence IDs.

14. Do NOT create a separate citations section.

15. Put each citation immediately after the factual
    statement it supports.

16. Keep the answer concise.

17. Before producing the final answer, internally check
    every factual statement against the evidence.

EVIDENCE:

{context}

USER QUESTION:

{question}

Now answer the question using ONLY the evidence above.
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

    return response["message"]["content"]


# ============================================================
# JUDGE ANSWER
# ============================================================

def judge_answer(
    question,
    reference_answer,
    generated_answer,
    evidence_results
):

    context_parts = []

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        evidence_id = f"E{i}"

        chunk = result["chunk"]

        context_parts.append(
            f"""
[{evidence_id}]
{chunk["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )


    judge_prompt = f"""
You are a strict evaluator for a
Retrieval-Augmented Generation system.

Your job is to evaluate ONLY the generated answer
against the provided evidence.

Do NOT use outside knowledge.

QUESTION:

{question}


REFERENCE ANSWER:

{reference_answer}


GENERATED ANSWER:

{generated_answer}


EVIDENCE:

{context}


Evaluate these three dimensions.


--------------------------------------------------
1. FAITHFULNESS
--------------------------------------------------

Check every factual claim in the generated answer.

A claim is faithful only if it is directly supported
by the provided evidence.

Do NOT give credit simply because a claim is generally
true according to your own knowledge.


--------------------------------------------------
2. ANSWER RELEVANCE
--------------------------------------------------

Check whether the generated answer directly answers
the user's question.

Penalize:

- unrelated information
- unnecessary explanations
- claims that do not help answer the question
- failure to address the actual question


--------------------------------------------------
3. CITATION CORRECTNESS
--------------------------------------------------

For EVERY citation such as [E1]:

1. Find the exact evidence identified by that citation.

2. Identify the claim associated with that citation.

3. Determine whether that evidence actually supports
   that specific claim.

A citation being topically related is NOT enough.

If a claim is supported by E1 but the answer cites E2,
that citation is incorrect.

If even one important citation is clearly incorrect,
reduce the citation correctness score.


--------------------------------------------------
SCORING
--------------------------------------------------

Give each score from 0.0 to 1.0.

1.0 = excellent
0.8 = strong
0.6 = partially correct
0.4 = significant problems
0.2 = mostly incorrect
0.0 = completely incorrect


Return ONLY valid JSON.

Use exactly this structure:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "citation_correctness": 0.0,
    "reason": "brief explanation of the evaluation"
}}
"""


    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": judge_prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# RUN DIAGNOSTIC QUESTIONS
# ============================================================

results = []


for question_number in QUESTION_NUMBERS:

    item = golden_dataset[
        question_number - 1
    ]

    question = item["question"]

    reference_answer = (
        item["reference_answer"]
    )


    print("\n==================================================")
    print(
        f"QUESTION {question_number}"
    )
    print("==================================================")

    print(
        f"\nQuestion: {question}"
    )


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    print("\nRetrieving evidence...")

    evidence_results = retrieve_evidence(
        question
    )


    print("\nRetrieved evidence:")

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        chunk = result["chunk"]

        print(
            f"E{i}: "
            f"{chunk['chunk_id']} | "
            f"Page {chunk['page']} | "
            f"Score {result['score']:.4f}"
        )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print("\nGenerating answer...")

    generated_answer = generate_answer(
        question,
        evidence_results
    )


    print("\nGenerated answer:")
    print(generated_answer)


    # --------------------------------------------------------
    # VALIDATE CITATIONS
    # --------------------------------------------------------

    citation_result = validate_citations(
        generated_answer,
        len(evidence_results)
    )


    print("\nCitation validation:")

    print(
        "Cited IDs:",
        citation_result["cited_ids"]
    )

    print(
        "Invalid IDs:",
        citation_result["invalid_ids"]
    )

    print(
        "Citation valid:",
        citation_result["citation_valid"]
    )


    # --------------------------------------------------------
    # JUDGE
    # --------------------------------------------------------

    print(
        "\nEvaluating answer with Qwen judge..."
    )

    judge_output = judge_answer(
        question,
        reference_answer,
        generated_answer,
        evidence_results
    )


    # --------------------------------------------------------
    # PARSE JUDGE
    # --------------------------------------------------------

    evaluation = extract_json(
        judge_output
    )


    if evaluation is None:

        print(
            "\n❌ Judge returned invalid JSON."
        )

        print(
            "\nRaw judge output:"
        )

        print(judge_output)

        evaluation = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_correctness": 0.0,
            "reason": "Invalid judge output."
        }


    # --------------------------------------------------------
    # VALIDATE JUDGE STRUCTURE
    # --------------------------------------------------------

    validation = validate_judge_output(
        evaluation
    )


    if not validation["valid"]:

        print(
            "\n❌ Invalid evaluation structure:"
        )

        print(
            validation["error"]
        )

        evaluation = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "citation_correctness": 0.0,
            "reason": validation["error"]
        }


    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = build_evaluation_result(
        question=question,
        generated_answer=generated_answer,
        citation_result=citation_result,
        evaluation=evaluation,
        threshold=QUALITY_THRESHOLD
    )


    results.append(result)


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("\nScores:")

    print(
        f"Faithfulness: "
        f"{result['faithfulness']:.2f}"
    )

    print(
        f"Answer relevance: "
        f"{result['answer_relevance']:.2f}"
    )

    print(
        f"Citation correctness: "
        f"{result['citation_correctness']:.2f}"
    )

    print(
        f"Overall score: "
        f"{result['overall_score']:.2f}"
    )

    print(
        "Question result:",
        "✅ PASS"
        if result["passed"]
        else "❌ FAIL"
    )

    print("\nReason:")
    print(result["reason"])


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================

print("\n\n==================================================")
print("3-QUESTION DIAGNOSTIC SUMMARY")
print("==================================================")


for result in results:

    status = (
        "✅ PASS"
        if result["passed"]
        else "❌ FAIL"
    )

    print(
        f"\n{status} | "
        f"{result['question']}"
    )

    print(
        f"Faithfulness: "
        f"{result['faithfulness']:.2f}"
    )

    print(
        f"Answer relevance: "
        f"{result['answer_relevance']:.2f}"
    )

    print(
        f"Citation correctness: "
        f"{result['citation_correctness']:.2f}"
    )

    print(
        f"Overall: "
        f"{result['overall_score']:.2f}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )


print("\n==================================================")
print("DIAGNOSTIC COMPLETE")
print("==================================================")

print(
    f"\nQuestions tested: {len(results)}"
)

print(
    "Passed:",
    sum(
        1
        for result in results
        if result["passed"]
    )
)

print(
    "Failed:",
    sum(
        1
        for result in results
        if not result["passed"]
    )
)

print(
    "\nDo NOT run the 20-question evaluation yet."
)