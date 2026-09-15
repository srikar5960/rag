import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)

from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import HybridRetriever


# ============================================================
# Paths
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data1"
    / "pdfs"
    / "paper.pdf"
)


# ============================================================
# Build pipeline
# ============================================================

loader = DocumentLoader()

documents = loader.load_pdf(
    str(PDF_PATH)
)

chunks = loader.split_documents(
    documents
)

texts = [
    document.page_content
    for document in chunks
]


embedding_manager = EmbeddingManager()

embeddings = (
    embedding_manager
    .generate_embeddings(
        texts
    )
)


vector_store = VectorStore(
    fresh_start=True
)

vector_store.add_documents(
    chunks,
    embeddings
)


retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    documents=texts
)


# ============================================================
# Questions that previously failed retrieval
# ============================================================

questions = [

    "What programme did Karnataka plan to strengthen for stray dogs?",

    "What vaccination programme was Karnataka planning for stray dogs?",

    "What additional budget did Tamil Nadu present for 2026-27?",

    "What areas were highlighted in Tamil Nadu's supplementary budget?",

    "What was the purpose of the additional allocation in Tamil Nadu's supplementary budget?"

]


# ============================================================
# Run debugging
# ============================================================

for question in questions:

    results = retriever.retrieve(
        query=question,
        candidate_k=20,
        final_k=5
    )