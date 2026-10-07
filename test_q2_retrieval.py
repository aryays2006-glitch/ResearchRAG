from evaluation.evaluate_generation import (
    load_golden_dataset,
    load_document_chunks,
    prepare_bm25,
    load_chroma_collection,
    load_reranker,
    retrieve_evidence,
)

dataset = load_golden_dataset()

question_data = dataset[1]

chunks = load_document_chunks()
bm25 = prepare_bm25(chunks)
collection = load_chroma_collection()
reranker = load_reranker()

evidence = retrieve_evidence(
    question_data["question"],
    chunks,
    bm25,
    collection,
    reranker,
)

print("\nQUESTION:")
print(question_data["question"])

print("\nREFERENCE ANSWER:")
print(question_data["reference_answer"])

print("\n" + "=" * 80)
print("RETRIEVED EVIDENCE")
print("=" * 80)

for i, result in enumerate(evidence, start=1):
    chunk = result["chunk"]

    print(f"\n--- E{i} ---")
    print("Chunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page"])
    print("Reranker score:", result["score"])
    print("\nTEXT:")
    print(chunk["text"])