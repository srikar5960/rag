RAG PDF Question Answering System
  A Retrieval-Augmented Generation (RAG) application that answers questions from uploaded PDF
  documents using semantic search, hybrid retrieval, and Groq LLMs.

Features
  Upload and process PDF documents
  Automatic text chunking
  Semantic embeddings for retrieval
  Hybrid search (Vector + BM25)
  Context-aware question answering
  RAG evaluation with a golden dataset
  GitHub Actions CI for automatic testing

Tech Stack
  Python
  LangChain
  ChromaDB
  Sentence Transformers
  Groq API
  Flask
  GitHub Actions

CI/CD
  This project uses GitHub Actions.
    Every Pull Request automatically:
    Installs dependencies
    Loads the PDF
    Generates embeddings
    Runs the RAG evaluation
    Passes or fails the Quality Gate
      The evaluation measures:
      Retrieval Score
      Faithfulness Score
      Correctness Score
      Overall Quality Gate (PASS / FAIL)
