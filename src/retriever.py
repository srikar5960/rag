###RAG Retriever

# Type hints for better code readability
from typing import List, Dict, Any

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(
        self,
        vector_store,
        embedding_manager
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings.
            embedding_manager: Manager for generating query embeddings.
        """

        # Store the VectorStore object
        self.vector_store = vector_store

        # Store the EmbeddingManager object
        self.embedding_manager = embedding_manager

    def retrieve(self,query: str,top_k: int = 5,score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User search query.
            top_k: Number of top similar documents to retrieve.
            score_threshold: Minimum similarity score required.

        Returns:
            List of dictionaries containing retrieved documents and metadata.
        """

        # Display the query information
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")

        # Generate embedding for the user query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            # Search the ChromaDB vector store
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # print(results)

            # Store retrieved documents
            retrieved_docs = []

            # Check whether documents were found
            if results["documents"] and results["documents"][0]:

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                # Process each retrieved document
                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):

                    # Convert cosine distance into similarity score
                    # similarity_score = 1 - distance

                    # Apply similarity threshold
                    if distance >= score_threshold:

                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            # "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
                print(retrieved_docs)

            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
