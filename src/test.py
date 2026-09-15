from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import HybridRetriever
from chatbot import ChatBot


# ============================================
# 1. Load PDF
# ============================================

loader = DocumentLoader()

chunks = loader.load_and_split(
    "../data1/pdfs/paper.pdf"
)

print(
    "Number of chunks:",
    len(chunks)
)


# ============================================
# 2. Extract text
# ============================================

texts = [
    chunk.page_content
    for chunk in chunks
]


# ============================================
# 3. Generate embeddings
# ============================================

embedding_manager = EmbeddingManager()

embeddings = (
    embedding_manager
    .generate_embeddings(texts)
)


# ============================================
# 4. Create fresh ChromaDB
# ============================================

vector_store = VectorStore(
    fresh_start=True
)

vector_store.add_documents(
    chunks,
    embeddings
)


# ============================================
# 5. Create Hybrid Retriever
# ============================================

retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    documents=texts
)


# ============================================
# 6. Create ChatBot
# ============================================

chatbot = ChatBot(
    retriever=retriever
)


# ============================================
# 7. Ask question
# ============================================

question = (
    "What happened to India's urea production?"
)

answer = chatbot.ask(
    question
)


# ============================================
# 8. Display answer
# ============================================

print()
print("============================================")
print("QUESTION")
print("============================================")

print(question)

print()
print("============================================")
print("ANSWER")
print("============================================")

print(answer)