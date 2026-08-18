from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from document_loader import load_pdf, split_documents
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import RAGRetriever
from chatbot import ChatBot

app = Flask(__name__)
CORS(app)

# Folder to store uploaded PDFs
UPLOAD_FOLDER = os.path.join("data1", "pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize once when the server starts
embedding_manager = EmbeddingManager()
vector_store = VectorStore()

# Initialize retriever
retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager
)

# Initialize chatbot
chatbot = ChatBot(retriever)


@app.route("/")
def home():
    return "PDF Chatbot Backend Running"


@app.route("/upload", methods=["POST"])
def upload_pdf():

    if "file" not in request.files:
        return jsonify({
            "message": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "message": "No file selected"
        }), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "message": "Only PDF files are allowed"
        }), 400

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    print(f"PDF saved at: {filepath}")

    # Load PDF
    documents = load_pdf(filepath)

    # Split into chunks
    chunks = split_documents(documents)

    # Extract text
    texts = [doc.page_content for doc in chunks]

    # Generate embeddings
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store.clear_collection()

    # Store embeddings
    vector_store.add_documents(
        chunks,
        embeddings
    )

    return jsonify({
        "message": "PDF uploaded and indexed successfully",
        "filename": file.filename,
        "chunks": len(chunks)
    }), 200


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "message": "Question is required"
        }), 400

    question = data["question"]

    response = chatbot.ask(question)

    return jsonify({
        "answer": response
    }), 200


if __name__ == "__main__":
    app.run(debug=True)