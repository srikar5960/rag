###Embedding and Vector DB imports

# Numerical computations and vector operations
import numpy as np


# ChromaDB vector database
import chromadb


# Generates unique IDs for each document chunk
import uuid

# Type hints for better code readability
from typing import List, Dict, Any, Tuple


import os

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data1/vector_store"
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory where the vector database is stored.
        """

        # Store the collection name
        self.collection_name = collection_name

        # Store the directory where ChromaDB will persist its data
        self.persist_directory = persist_directory

        # Initialize the ChromaDB client (currently None)
        self.client = None

        # Initialize the ChromaDB collection (currently None)
        self.collection = None

        # Automatically initialize the vector store
        self._initialize_store()
    def _initialize_store(self):
        """
        Initialize the ChromaDB client and collection.
        """

        try:
            # Create the directory for storing the vector database
            os.makedirs(self.persist_directory, exist_ok=True)

            # Create a persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            # Get the existing collection or create a new one
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG"
                }
            )

            # Display initialization details
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain Document objects.
            embeddings: Corresponding embeddings for the documents.
        """

        # Ensure every document has a corresponding embedding
        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(f"Adding {len(documents)} documents to vector store...")

        # Lists that will be inserted into ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        # Process each document and its embedding
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):

            # Generate a unique ID for the document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            # Store the document text
            documents_text.append(doc.page_content)

            # Convert NumPy array to Python list
            embeddings_list.append(embedding.tolist())

        # Add everything to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(
                f"Successfully added {len(documents)} documents to vector store"
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def clear_collection(self):
        """
        Delete the existing collection and create a new empty one.
        """

        try:
            # Delete existing collection
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")

        except Exception:
            print("Collection does not exist or is already empty.")

        # Create a fresh collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document embeddings for RAG"
            }
        )

        print("Created new empty collection.")