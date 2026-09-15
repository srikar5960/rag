from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import HybridRetriever
from chatbot import ChatBot


# ============================================================
# Flask setup
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# Upload folder
# ============================================================

UPLOAD_FOLDER = os.path.join(
    "data1",
    "pdfs"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# Initialize embedding manager
# ============================================================

embedding_manager = EmbeddingManager()


# ============================================================
# RAG components
# ============================================================

vector_store = None
retriever = None
chatbot = None


# ============================================================
# Home
# ============================================================

@app.route("/")
def home():

    return "PDF Chatbot Backend Running"


# ============================================================
# Upload PDF
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    global vector_store
    global retriever
    global chatbot

    # --------------------------------------------------------
    # Check whether file exists
    # --------------------------------------------------------

    if "file" not in request.files:

        return jsonify({
            "message": "No file uploaded"
        }), 400


    file = request.files["file"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return jsonify({
            "message": "No file selected"
        }), 400


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "message": "Only PDF files are allowed"
        }), 400


    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    print(
        f"PDF saved at: {filepath}"
    )


    try:

        # ====================================================
        # 1. Load PDF
        # ====================================================

        loader = DocumentLoader()

        documents = loader.load_pdf(
            filepath
        )

        print(
            f"Pages loaded: {len(documents)}"
        )


        # ====================================================
        # 2. Split PDF into chunks
        # ====================================================

        chunks = loader.split_documents(
            documents
        )

        print(
            f"Chunks created: {len(chunks)}"
        )


        # ====================================================
        # 3. Extract chunk text
        # ====================================================

        texts = [
            document.page_content
            for document in chunks
        ]


        # ====================================================
        # 4. Generate embeddings
        # ====================================================

        embeddings = (
            embedding_manager
            .generate_embeddings(
                texts
            )
        )

        print(
            f"Embeddings created: {len(embeddings)}"
        )


        # ====================================================
        # 5. Delete old ChromaDB
        # ====================================================

        vector_store = VectorStore(
            fresh_start=True
        )


        # ====================================================
        # 6. Store new chunks and embeddings
        # ====================================================

        vector_store.add_documents(
            chunks,
            embeddings
        )


        # ====================================================
        # 7. Create Hybrid Retriever
        #
        # Semantic Search
        #       +
        # BM25
        #       ↓
        # RRF
        #       ↓
        # Cross Encoder
        # ====================================================

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            documents=texts
        )


        # ====================================================
        # 8. Create ChatBot
        # ====================================================

        chatbot = ChatBot(
            retriever=retriever
        )


        print(
            "RAG pipeline initialized successfully."
        )


        # ====================================================
        # 9. Return upload information
        # ====================================================

        return jsonify({

            "message":
                "PDF uploaded and indexed successfully",

            "filename":
                file.filename,

            "pages":
                len(documents),

            "chunks":
                len(chunks)

        }), 200


    except Exception as e:

        print(
            "Error while processing PDF:",
            str(e)
        )

        return jsonify({

            "message":
                "Error while processing PDF",

            "error":
                str(e)

        }), 500


# ============================================================
# Chat
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    global chatbot


    # --------------------------------------------------------
    # Check whether a PDF has been uploaded
    # --------------------------------------------------------

    if chatbot is None:

        return jsonify({

            "message":
                "Please upload a PDF before asking questions."

        }), 400


    # --------------------------------------------------------
    # Read JSON request
    # --------------------------------------------------------

    data = request.get_json()


    if not data:

        return jsonify({

            "message":
                "Request body is required"

        }), 400


    # --------------------------------------------------------
    # Check question
    # --------------------------------------------------------

    if "question" not in data:

        return jsonify({

            "message":
                "Question is required"

        }), 400


    question = data["question"]


    # --------------------------------------------------------
    # Validate question type
    # --------------------------------------------------------

    if not isinstance(question, str):

        return jsonify({

            "message":
                "Question must be a string"

        }), 400


    question = question.strip()


    # --------------------------------------------------------
    # Check empty question
    # --------------------------------------------------------

    if not question:

        return jsonify({

            "message":
                "Question cannot be empty"

        }), 400


    # --------------------------------------------------------
    # Ask chatbot
    # --------------------------------------------------------

    try:

        response = chatbot.ask(
            question
        )


        # ====================================================
        # ChatBot now returns:
        #
        # {
        #     "answer": "...",
        #     "sources": [...]
        # }
        #
        # So return it directly.
        # ====================================================

        return jsonify(
            response
        ), 200


    except Exception as e:

        print(
            "Chat error:",
            str(e)
        )

        return jsonify({

            "message":
                "Error while generating answer",

            "error":
                str(e)

        }), 500


# ============================================================
# Start Flask server
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )