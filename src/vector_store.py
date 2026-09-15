from pathlib import Path
import shutil

import chromadb


class VectorStore:

    def __init__(
        self,
        collection_name="rag_documents",
        fresh_start=False
    ):

        # ----------------------------------------
        # Project root
        # ----------------------------------------

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        self.persist_directory = (
            project_root / "chroma_db"
        )

        self.collection_name = collection_name

        # ----------------------------------------
        # Delete complete ChromaDB if requested
        # ----------------------------------------

        if fresh_start:

            if self.persist_directory.exists():

                shutil.rmtree(
                    self.persist_directory
                )

                print(
                    "Old ChromaDB directory deleted."
                )

            else:

                print(
                    "No previous ChromaDB found."
                )

        # ----------------------------------------
        # Create persistent Chroma client
        # ----------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # ----------------------------------------
        # Create collection
        # ----------------------------------------

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

        print(
            "ChromaDB location:"
        )

        print(
            self.persist_directory
        )

    # --------------------------------------------
    # Add documents
    # --------------------------------------------

    def add_documents(
        self,
        documents,
        embeddings
    ):

        ids = []
        texts = []
        metadatas = []

        for i, document in enumerate(
            documents
        ):

            ids.append(
                f"chunk_{i}"
            )

            texts.append(
                document.page_content
            )

            metadata = (
                document.metadata.copy()
            )

            metadata["chunk_id"] = i

            metadatas.append(
                metadata
            )

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"{len(documents)} chunks added to ChromaDB."
        )

    # --------------------------------------------
    # Semantic search
    # --------------------------------------------

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        results = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        output = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            output.append({
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "index": metadata["chunk_id"]
            })

        return output