import json
import re
import sys
from pathlib import Path

import chromadb
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT PRODUCTION GENERATOR
# ============================================================

from api.generator import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_dataset.json"
)

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"

RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_TOP_K = 10

BM25_TOP_K = 10

RERANK_CANDIDATE_K = 10

RERANK_TOP_K = 5

RRF_K = 60

QUALITY_THRESHOLD = 0.80


# ============================================================
# DATASET
# ============================================================

def load_golden_dataset():

    with open(
        GOLDEN_DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# DOCUMENT CHUNKS
# ============================================================

def load_document_chunks():

    from src.ingest import load_chunks

    return load_chunks()


# ============================================================
# BM25
# ============================================================

def prepare_bm25(chunks):

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


# ============================================================
# CHROMA
# ============================================================

def load_chroma_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client.get_collection(
        name="research_papers"
    )


# ============================================================
# RERANKER
# ============================================================

def load_reranker():

    print("\nLoading BGE reranker model...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print("BGE reranker loaded.")

    return reranker


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(
    query,
    chunks,
    bm25,
    collection,
    reranker
):

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

    tokenized_query = query.lower().split()

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

    candidate_ids = [
        chunk_id
        for chunk_id, score
        in ranked_candidates[:RERANK_CANDIDATE_K]
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
        [query, chunk["text"]]
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
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_evidence_context(evidence_results):

    context_parts = []

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        chunk = result["chunk"]

        context_parts.append(
            f"""
[E{i}]
Source: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# JUDGE
# ============================================================

def judge_answer(
    question,
    reference_answer,
    generated_answer,
    evidence_results
):

    evidence_parts = []

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        chunk = result["chunk"]

        evidence_parts.append(
            f"""
[E{i}]
Source: {chunk["source"]}
Page: {chunk["page"]}

{chunk["text"]}
"""
        )

    evidence = "\n".join(
        evidence_parts
    )

    prompt = f"""
You are the QUALITY EVALUATOR for ResearchRAG.

Evaluate ONLY the generated answer.

Do not evaluate the research paper itself.
Do not evaluate the evidence as if it were the answer.
Do not evaluate the reference answer as if it were the answer.

============================================================
QUESTION
============================================================

{question}

============================================================
REFERENCE ANSWER
============================================================

{reference_answer}

============================================================
GENERATED ANSWER
============================================================

--- BEGIN GENERATED ANSWER ---

{generated_answer}

--- END GENERATED ANSWER ---

============================================================
EVIDENCE
============================================================

{evidence}

============================================================
SCORING
============================================================

FAITHFULNESS:

Score 1.0 if every factual claim in the generated answer
is directly supported by the provided evidence.

Reduce the score if the generated answer contains:
- unsupported claims
- additional interpretations
- unsupported explanations
- facts not present in the evidence

ANSWER RELEVANCE:

Score 1.0 if the generated answer directly answers the
question asked.

The answer does not need to use the exact wording of the
reference answer.

Equivalent wording is acceptable if it preserves the same
meaning.

If the evidence contains the answer but the generated answer
incorrectly says that the evidence is insufficient, reduce
answer relevance.

CITATION CORRECTNESS:

ResearchRAG citations must use:

[E1]
[E2]
[E3]
[E4]
[E5]

For every factual claim that has a citation:

1. Check that the E-number exists.
2. Check that the cited evidence directly supports the claim.

Reduce the score when:
- a citation points to unrelated evidence
- a factual claim has no citation
- the citation format is invalid
- paper citation numbers such as [1] or [37] are used as
  ResearchRAG citations

============================================================
IMPORTANT
============================================================

Judge the GENERATED ANSWER itself.

Do not invent problems that are not present in the generated
answer.

Do not claim that the generated answer contains a statement
unless that statement actually appears in the generated answer.

Base the scores only on the question, reference answer,
generated answer, and provided evidence.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{{
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "citation_correctness": 0.0,
    "reason": "Brief explanation."
}}

All scores must be between 0.0 and 1.0.
"""

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    try:

        return json.loads(
            text.strip()
        )

    except json.JSONDecodeError:

        pass

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

    if not isinstance(
        evaluation,
        dict
    ):

        return False

    required_fields = [
        "faithfulness",
        "answer_relevance",
        "citation_correctness",
        "reason"
    ]

    for field in required_fields:

        if field not in evaluation:

            return False

    for field in [
        "faithfulness",
        "answer_relevance",
        "citation_correctness"
    ]:

        value = evaluation[field]

        if not isinstance(
            value,
            (int, float)
        ):

            return False

        if not 0.0 <= float(value) <= 1.0:

            return False

    return isinstance(
        evaluation["reason"],
        str
    )


# ============================================================
# RUN FULL EVALUATION
# ============================================================

def run_full_evaluation():

    print("\n" + "=" * 60)

    print(
        "RESEARCHRAG GENERATION EVALUATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LOAD EVERYTHING ONCE
    # --------------------------------------------------------

    print("\nLoading golden dataset...")

    golden_dataset = (
        load_golden_dataset()
    )

    print(
        f"Loaded {len(golden_dataset)} questions."
    )

    print("\nLoading document chunks...")

    chunks = load_document_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print("\nPreparing BM25...")

    bm25 = prepare_bm25(
        chunks
    )

    print("BM25 ready.")

    print("\nLoading Chroma...")

    collection = (
        load_chroma_collection()
    )

    print("Chroma ready.")

    print("\nLoading reranker...")

    reranker = load_reranker()

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    results = []

    total_questions = len(
        golden_dataset
    )

    for index, item in enumerate(
        golden_dataset,
        start=1
    ):

        question = item["question"]

        reference_answer = (
            item["reference_answer"]
        )

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"QUESTION {index}/{total_questions}"
        )

        print(
            "-" * 60
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        print(
            "Retrieving + reranking..."
        )

        evidence_results = retrieve_evidence(
            question,
            chunks,
            bm25,
            collection,
            reranker
        )

        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_evidence_context(
            evidence_results
        )

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        print(
            "Generating answer using production generator..."
        )

        generated_answer = generate_answer(
            question=question,
            context=context
        )

        print(
            f"Answer: {generated_answer}"
        )

        # ----------------------------------------------------
        # JUDGE
        # ----------------------------------------------------

        print(
            "Judging answer..."
        )

        judge_output = judge_answer(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            evidence_results=evidence_results
        )

        evaluation = extract_json(
            judge_output
        )

        if not validate_judge_output(
            evaluation
        ):

            print(
                "❌ Invalid judge output."
            )

            continue

        print(
            "Scores:"
        )

        print(
            f"  Faithfulness: "
            f"{evaluation['faithfulness']:.2f}"
        )

        print(
            f"  Relevance: "
            f"{evaluation['answer_relevance']:.2f}"
        )

        print(
            f"  Citation: "
            f"{evaluation['citation_correctness']:.2f}"
        )

        print(
            f"  Reason: "
            f"{evaluation['reason']}"
        )

        results.append(
            {
                "question": question,
                "generated_answer": generated_answer,
                "evaluation": evaluation
            }
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL GENERATION EVALUATION"
    )

    print(
        "=" * 60
    )

    if not results:

        print(
            "No valid evaluation results."
        )

        return

    count = len(results)

    average_faithfulness = (
        sum(
            result["evaluation"]["faithfulness"]
            for result in results
        )
        / count
    )

    average_relevance = (
        sum(
            result["evaluation"]["answer_relevance"]
            for result in results
        )
        / count
    )

    average_citation = (
        sum(
            result["evaluation"]["citation_correctness"]
            for result in results
        )
        / count
    )

    average_overall = (
        average_faithfulness
        + average_relevance
        + average_citation
    ) / 3

    print(
        f"Questions evaluated: {count}"
    )

    print(
        f"Average Faithfulness: "
        f"{average_faithfulness:.2f}"
    )

    print(
        f"Average Answer Relevance: "
        f"{average_relevance:.2f}"
    )

    print(
        f"Average Citation Correctness: "
        f"{average_citation:.2f}"
    )

    print(
        f"Average Overall Score: "
        f"{average_overall:.2f}"
    )

    print(
        f"Quality Threshold: "
        f"{QUALITY_THRESHOLD:.2f}"
    )

    if average_overall >= QUALITY_THRESHOLD:

        print(
            "\n✅ GENERATION QUALITY PASSED"
        )

    else:

        print(
            "\n❌ GENERATION QUALITY FAILED"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_full_evaluation()