from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized_documents = [
            self.tokenize(document)
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def tokenize(self, text):

        return text.lower().split()

    def search(self, query, top_k=20):

        tokenized_query = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1
        ):

            results.append({
                "text": self.documents[index],
                "index": index,
                "bm25_score": float(
                    scores[index]
                ),
                "rank": rank
            })

        return results


class HybridRetriever:

    def __init__(
        self,
        vector_store,
        embedding_manager,
        documents
    ):

        self.vector_store = vector_store

        self.embedding_manager = (
            embedding_manager
        )

        self.documents = documents

        # BM25
        self.bm25_retriever = BM25Retriever(
            documents
        )

        # Cross Encoder
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    # ==========================================
    # Semantic Search
    # ==========================================

    def semantic_search(
        self,
        query,
        top_k=20
    ):

        query_embedding = (
            self.embedding_manager
            .generate_query_embedding(
                query
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

    # ==========================================
    # BM25 Search
    # ==========================================

    def bm25_search(
        self,
        query,
        top_k=20
    ):

        return self.bm25_retriever.search(
            query,
            top_k=top_k
        )

    # ==========================================
    # RRF
    # ==========================================

    def reciprocal_rank_fusion(
        self,
        semantic_results,
        bm25_results,
        k=60
    ):

        fused_scores = {}

        documents = {}

        # --------------------------------------
        # Semantic results
        # --------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            index = result["index"]

            documents[index] = result

            fused_scores[index] = (
                fused_scores.get(
                    index,
                    0
                )
                + 1 / (k + rank)
            )

        # --------------------------------------
        # BM25 results
        # --------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            index = result["index"]

            if index not in documents:
                documents[index] = result

            fused_scores[index] = (
                fused_scores.get(
                    index,
                    0
                )
                + 1 / (k + rank)
            )

        # --------------------------------------
        # Sort
        # --------------------------------------

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in ranked:

            result = documents[index]

            results.append({
                "text": result["text"],
                "index": index,
                "metadata": result.get("metadata", {}),
                "rrf_score": score
            })

        return results

    # ==========================================
    # Cross Encoder Reranking
    # ==========================================

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        if not candidates:
            return []

        pairs = [
            [
                query,
                candidate["text"]
            ]
            for candidate in candidates
        ]

        scores = self.cross_encoder.predict(
            pairs
        )

        for candidate, score in zip(
            candidates,
            scores
        ):

            candidate[
                "rerank_score"
            ] = float(score)

        candidates.sort(
            key=lambda x: x[
                "rerank_score"
            ],
            reverse=True
        )

        return candidates[:top_k]

    # ==========================================
    # COMPLETE RETRIEVAL PIPELINE
    # ==========================================

    def retrieve(
            self,
            query,
            candidate_k=20,
            final_k=5
    ):

        print()
        print("============================================")
        print("RETRIEVAL DEBUG")
        print("============================================")

        print()
        print("QUERY:")
        print(query)

        # ========================================================
        # 1. Semantic Search
        # ========================================================

        semantic_results = (
            self.semantic_search(
                query,
                top_k=candidate_k
            )
        )

        print()
        print("--------------------------------------------")
        print("SEMANTIC SEARCH")
        print("--------------------------------------------")

        for result in semantic_results:
            print(
                f"Index: {result['index']} | "
                f"Distance: {result.get('distance')}"
            )

            print(
                f"Text: {result['text'][:150].replace(chr(10), ' ')}"
            )

            print()

        # ========================================================
        # 2. BM25 Search
        # ========================================================

        bm25_results = (
            self.bm25_search(
                query,
                top_k=candidate_k
            )
        )

        print()
        print("--------------------------------------------")
        print("BM25 SEARCH")
        print("--------------------------------------------")

        for result in bm25_results:
            print(
                f"Index: {result['index']} | "
                f"BM25 Score: {result['bm25_score']}"
            )

            print(
                f"Text: {result['text'][:150].replace(chr(10), ' ')}"
            )

            print()

        # ========================================================
        # 3. RRF
        # ========================================================

        hybrid_results = (
            self.reciprocal_rank_fusion(
                semantic_results,
                bm25_results
            )
        )

        print()
        print("--------------------------------------------")
        print("RRF RESULTS")
        print("--------------------------------------------")

        for result in hybrid_results[:candidate_k]:
            print(
                f"Index: {result['index']} | "
                f"RRF Score: {result['rrf_score']}"
            )

            print(
                f"Text: {result['text'][:150].replace(chr(10), ' ')}"
            )

            print()

        # ========================================================
        # 4. Cross Encoder
        # ========================================================

        rerank_candidates = (
            hybrid_results[:candidate_k]
        )

        final_results = self.rerank(
            query,
            rerank_candidates,
            top_k=final_k
        )

        print()
        print("--------------------------------------------")
        print("CROSS ENCODER RESULTS")
        print("--------------------------------------------")

        for result in final_results:
            print(
                f"Index: {result['index']} | "
                f"RRF: {result['rrf_score']} | "
                f"Cross Encoder: {result['rerank_score']}"
            )

            print(
                f"Text: {result['text'][:150].replace(chr(10), ' ')}"
            )

            print()

        # ========================================================
        # Final
        # ========================================================

        print("============================================")
        print("END RETRIEVAL DEBUG")
        print("============================================")

        return final_results