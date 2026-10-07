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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "Individual_Evaluation"
    / "results"
)

EMBEDDING_MODEL = "nomic-embed-text"

RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_TOP_K = 10

BM25_TOP_K = 10

RERANK_CANDIDATE_K = 10

RERANK_TOP_K = 5

RRF_K = 60


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

    print("\nRunning vector search...")

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

    print("Running BM25 search...")

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

    print("Combining vector and BM25 results...")

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

    print("Running BGE reranker...")

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

def build_evidence_context(
    evidence_results
):

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

    evidence = build_evidence_context(
        evidence_results
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
IMPORTANT CITATION RULE
============================================================

ResearchRAG citations are ONLY citations in the generated
answer that have this format:

[E1]
[E2]
[E3]
[E4]
[E5]

ONLY inspect these E-number citations when evaluating
citation correctness.

The research paper itself may contain ordinary academic
citations such as:

[1]
[2]
[11]
[30]

These academic citations may appear INSIDE THE EVIDENCE.

IGNORE those academic citations completely.

They are NOT citations made by the generated answer.

Never penalize the generated answer because an academic
citation such as [30] appears inside the evidence.

============================================================
FAITHFULNESS
============================================================

Score 1.0 if every factual claim in the generated answer
is directly supported by the provided evidence.

Reduce the score if the generated answer contains:
- unsupported claims
- additional interpretations
- unsupported explanations
- facts not present in the evidence

============================================================
ANSWER RELEVANCE
============================================================

Score 1.0 if the generated answer directly answers the
question asked.

Equivalent wording is acceptable if it preserves the same
meaning.

Reduce the score for:
- unrelated information
- unnecessary explanations
- failure to answer the question
- claiming the evidence is insufficient when the evidence
  actually contains the answer

============================================================
CITATION CORRECTNESS
============================================================

For every [E-number] citation appearing in the GENERATED
ANSWER:

1. Check that the E-number exists.
2. Identify the factual claim associated with it.
3. Check whether that evidence directly supports the claim.

Reduce the score when:
- an E-number points to unrelated evidence
- an important factual claim has no citation
- the citation format is invalid
- the generated answer uses academic citation numbers as
  ResearchRAG citations

Again:

IGNORE [1], [2], [11], [30], etc. appearing ONLY inside
the evidence.

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

Return ONLY valid JSON.

Do not use Markdown code fences.

Return exactly:

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
# SAVE RESULT
# ============================================================

def save_result(
    question_number,
    question,
    reference_answer,
    generated_answer,
    evidence_results,
    judge_output,
    evaluation
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    evidence_data = []

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        chunk = result["chunk"]

        evidence_data.append(
            {
                "evidence_id": f"E{i}",
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "reranker_score": result["score"],
                "text": chunk["text"]
            }
        )

    result_data = {
        "question_number": question_number,
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "evidence": evidence_data,
        "raw_judge_output": judge_output,
        "judge_evaluation": evaluation,
        "judge_valid": validate_judge_output(
            evaluation
        )
    }

    result_path = (
        RESULTS_DIR
        / f"q{question_number:02d}.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResult saved to:"
    )

    print(
        result_path
    )


# ============================================================
# RUN ONE QUESTION
# ============================================================

def run_one_question(
    question_number
):

    print("\n" + "=" * 70)

    print(
        f"RESEARCHRAG INDIVIDUAL EVALUATION — Q{question_number}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    golden_dataset = (
        load_golden_dataset()
    )

    if question_number < 1 or question_number > len(
        golden_dataset
    ):

        print(
            f"Invalid question number."
        )

        print(
            f"Choose between 1 and {len(golden_dataset)}."
        )

        return

    item = golden_dataset[
        question_number - 1
    ]

    question = item["question"]

    reference_answer = (
        item["reference_answer"]
    )

    print(
        f"\nQuestion {question_number}:"
    )

    print(
        question
    )

    print(
        "\nReference answer:"
    )

    print(
        reference_answer
    )

    # --------------------------------------------------------
    # LOAD RETRIEVAL COMPONENTS
    # --------------------------------------------------------

    print(
        "\nLoading document chunks..."
    )

    chunks = load_document_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print(
        "\nPreparing BM25..."
    )

    bm25 = prepare_bm25(
        chunks
    )

    print(
        "BM25 ready."
    )

    print(
        "\nLoading Chroma..."
    )

    collection = (
        load_chroma_collection()
    )

    print(
        "Chroma ready."
    )

    reranker = load_reranker()

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    evidence_results = retrieve_evidence(
        question,
        chunks,
        bm25,
        collection,
        reranker
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "TOP 5 RETRIEVED EVIDENCE"
    )

    print(
        "-" * 70
    )

    for i, result in enumerate(
        evidence_results,
        start=1
    ):

        chunk = result["chunk"]

        print(
            f"\n[E{i}]"
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            f"Reranker score: "
            f"{result['score']:.6f}"
        )

        print(
            f"Source: {chunk['source']}"
        )

        print(
            "\nEvidence preview:"
        )

        print(
            chunk["text"][:700]
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_evidence_context(
        evidence_results
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "GENERATING ANSWER"
    )

    print(
        "-" * 70
    )

    generated_answer = generate_answer(
        question=question,
        context=context
    )

    print(
        "\nGenerated answer:"
    )

    print(
        generated_answer
    )

    # --------------------------------------------------------
    # EXTRACT CITATIONS
    # --------------------------------------------------------

    cited_ids = sorted(
        set(
            re.findall(
                r"\[(E\d+)\]",
                generated_answer
            )
        )
    )

    valid_evidence_ids = {
        f"E{i}"
        for i in range(
            1,
            len(evidence_results) + 1
        )
    }

    invalid_citations = sorted(
        set(cited_ids)
        - valid_evidence_ids
    )

    print(
        "\nCitations found in answer:"
    )

    print(
        cited_ids
    )

    print(
        "Invalid citations:"
    )

    print(
        invalid_citations
    )

    # --------------------------------------------------------
    # JUDGE
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "JUDGING ANSWER"
    )

    print(
        "-" * 70
    )

    judge_output = judge_answer(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        evidence_results=evidence_results
    )

    print(
        "\nRaw judge output:"
    )

    print(
        judge_output
    )

    evaluation = extract_json(
        judge_output
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print(
        "\nJudge validation:"
    )

    if not validate_judge_output(
        evaluation
    ):

        print(
            "❌ INVALID JUDGE OUTPUT"
        )

        print(
            "\nThe RAG answer itself is NOT being marked "
            "as a failure yet."
        )

        print(
            "We first need to determine why the judge "
            "returned invalid output."
        )

    else:

        print(
            "✅ Judge output is valid."
        )

        print(
            f"\nFaithfulness: "
            f"{evaluation['faithfulness']:.2f}"
        )

        print(
            f"Relevance: "
            f"{evaluation['answer_relevance']:.2f}"
        )

        print(
            f"Citation correctness: "
            f"{evaluation['citation_correctness']:.2f}"
        )

        print(
            f"Reason: "
            f"{evaluation['reason']}"
        )

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------

    save_result(
        question_number=question_number,
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        evidence_results=evidence_results,
        judge_output=judge_output,
        evaluation=evaluation
    )

    # --------------------------------------------------------
    # FINAL DIAGNOSTIC
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INDIVIDUAL EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    if not validate_judge_output(
        evaluation
    ):

        print(
            "Status: ⚠️ JUDGE OUTPUT NEEDS INVESTIGATION"
        )

    elif invalid_citations:

        print(
            "Status: ⚠️ INVALID CITATION FOUND"
        )

    else:

        print(
            "Status: Judge completed successfully."
        )

    print(
        "\nThe process is finished."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python evaluation/Individual_Evaluation/evaluate_one.py <question_number>"
        )

        print(
            "\nExample:"
        )

        print(
            "python evaluation/Individual_Evaluation/evaluate_one.py 9"
        )

        sys.exit(1)

    try:

        question_number = int(
            sys.argv[1]
        )

    except ValueError:

        print(
            "Question number must be an integer."
        )

        sys.exit(1)

    run_one_question(
        question_number
    )