import chromadb
import ollama

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ingest import load_chunks


EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-base"

VECTOR_TOP_K = 10
BM25_TOP_K = 10
RERANK_CANDIDATE_K = 10
RERANK_TOP_K = 5
RRF_K = 60


class Retriever:

    def __init__(self):
        print("Loading retriever...")

        self.client = chromadb.PersistentClient(
            path="chroma_db"
        )

        self.collection = self.client.get_collection(
            name="research_papers"
        )

        print("Loading reranker model...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        self.refresh()

        print("Retriever loaded successfully.")

    def refresh(self):
        """
        Reload the active PDF chunks and rebuild
        the BM25 retrieval index.
        """

        self.chunks = load_chunks()

        # Refresh the Chroma collection reference.
        self.collection = self.client.get_collection(
            name="research_papers"
        )

        print(
            f"Retriever refreshed with "
            f"{len(self.chunks)} chunks."
        )

        documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        tokenized_documents = [
            document.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def retrieve(self, question: str):

        # -----------------------------------------------------
        # 1. Vector retrieval
        # -----------------------------------------------------

        embedding_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question
        )

        query_embedding = (
            embedding_response["embeddings"][0]
        )

        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=VECTOR_TOP_K
        )

        vector_chunk_ids = (
            vector_results["ids"][0]
        )

        # -----------------------------------------------------
        # 2. BM25 keyword retrieval
        # -----------------------------------------------------

        tokenized_query = (
            question.lower().split()
        )

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_top_indices = (
            bm25_scores
            .argsort()[-BM25_TOP_K:][::-1]
        )

        bm25_chunk_ids = [
            self.chunks[index]["chunk_id"]
            for index in bm25_top_indices
        ]

        # -----------------------------------------------------
        # 3. Reciprocal Rank Fusion
        # -----------------------------------------------------

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
            in ranked_candidates[
                :RERANK_CANDIDATE_K
            ]
        ]

        # -----------------------------------------------------
        # 4. Get candidate chunks
        # -----------------------------------------------------

        candidate_chunks = [
            chunk
            for chunk in self.chunks
            if chunk["chunk_id"] in candidate_ids
        ]

        # -----------------------------------------------------
        # 5. BGE cross-encoder reranking
        # -----------------------------------------------------

        pairs = [
            [question, chunk["text"]]
            for chunk in candidate_chunks
        ]

        reranker_scores = self.reranker.predict(
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

        # -----------------------------------------------------
        # 6. Return the best results
        # -----------------------------------------------------

        return reranked_results[
            :RERANK_TOP_K
        ]