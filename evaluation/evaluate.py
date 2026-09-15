import json
import sys
from langchain_groq import ChatGroq


# ============================================================
# Load environment variables
# ============================================================

from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Make src importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# Import RAG components
# ============================================================

from document_loader import DocumentLoader
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from retriever import HybridRetriever


# ============================================================
# Paths
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "golden_dataset.json"
)

PDF_PATH = (
    PROJECT_ROOT
    / "data1"
    / "pdfs"
    / "paper.pdf"
)

# ============================================================
# Quality Gate Thresholds
# ============================================================

MIN_RETRIEVAL_SCORE = 80.0

MIN_FAITHFULNESS_SCORE = 90.0

MIN_CORRECTNESS_SCORE = 90.0


# ============================================================
# Load golden dataset
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    dataset = json.load(file)


print(
    f"Golden questions: {len(dataset)}"
)


# ============================================================
# Build RAG pipeline
# ============================================================

print()
print("Building RAG evaluation pipeline...")


# ------------------------------------------------------------
# Load PDF
# ------------------------------------------------------------

loader = DocumentLoader()

documents = loader.load_pdf(
    str(PDF_PATH)
)

print(
    f"Pages loaded: {len(documents)}"
)


# ------------------------------------------------------------
# Split documents
# ------------------------------------------------------------

chunks = loader.split_documents(
    documents
)

print(
    f"Chunks created: {len(chunks)}"
)


# ------------------------------------------------------------
# Extract text
# ------------------------------------------------------------

texts = [
    document.page_content
    for document in chunks
]


# ------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------

embedding_manager = EmbeddingManager()

embeddings = (
    embedding_manager
    .generate_embeddings(
        texts
    )
)

print(
    f"Embeddings created: {len(embeddings)}"
)


# ------------------------------------------------------------
# Create fresh ChromaDB
# ------------------------------------------------------------

vector_store = VectorStore(
    fresh_start=True
)

vector_store.add_documents(
    chunks,
    embeddings
)


# ------------------------------------------------------------
# Create Hybrid Retriever
# ------------------------------------------------------------

retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_manager=embedding_manager,
    documents=texts
)

print(
    "Hybrid retriever ready."
)


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500,
    reasoning_effort="low"
)


# ============================================================
# Faithfulness LLM
# ============================================================

faithfulness_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500,
    reasoning_effort="low"
)


# ============================================================
# Faithfulness Evaluation
# ============================================================

def evaluate_faithfulness(
    question,
    context,
    answer
):

    prompt = f"""
You are a strict evaluator for a Retrieval Augmented
Generation system.

Your job is to determine whether the answer is completely
supported by the provided context.

Do NOT use outside knowledge.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
{answer}

Rules:

1. If every factual claim in the answer is supported
   by the context, return exactly:

FAITHFUL

2. If any factual claim is not supported by the context,
   return exactly:

NOT_FAITHFUL

3. Do not provide an explanation.

4. Do not use markdown.

5. Return only one of the two labels.
"""

    response = faithfulness_llm.invoke(
        prompt
    )

    result = response.content.strip().upper()

    print()
    print(
        "FAITHFULNESS JUDGE RESPONSE:"
    )

    print(
        repr(result)
    )

    if result == "FAITHFUL":

        return True

    if result == "NOT_FAITHFUL":

        return False

    print(
        "Warning: Faithfulness judge returned "
        "an unexpected response."
    )

    return False


# ============================================================
# Answer Correctness Evaluation
# ============================================================

def evaluate_correctness(
    question,
    golden_answer,
    generated_answer
):

    prompt = f"""
You are evaluating the correctness of an answer
generated by a Retrieval Augmented Generation system.

Compare the generated answer with the verified golden answer.

Question:
{question}

Golden answer:
{golden_answer}

Generated answer:
{generated_answer}

Rules:

1. The generated answer does not need to use the exact
   same wording as the golden answer.

2. Different wording is acceptable if the meaning and
   important facts are correct.

3. If the generated answer correctly answers the question,
   return exactly:

CORRECT

4. If the generated answer contains incorrect,
   contradictory, or missing important information,
   return exactly:

INCORRECT

5. Do not provide an explanation.

6. Do not use markdown.

Return only one label:

CORRECT

or

INCORRECT
"""

    response = faithfulness_llm.invoke(
        prompt
    )

    result = response.content.strip().upper()

    print()
    print(
        "CORRECTNESS JUDGE RESPONSE:"
    )

    print(
        repr(result)
    )

    if result == "CORRECT":

        return True

    if result == "INCORRECT":

        return False

    print(
        "Warning: Correctness judge returned "
        "an unexpected response."
    )

    return False


# ============================================================
# Evaluation Counters
# ============================================================

retrieval_passed = 0
retrieval_failed = 0

faithfulness_passed = 0
faithfulness_failed = 0

correctness_passed = 0
correctness_failed = 0


# ============================================================
# Start Evaluation
# ============================================================

print()
print("============================================")
print("RAG EVALUATION")
print("============================================")


for item in dataset:

    # ========================================================
    # Read golden data
    # ========================================================

    question_id = item["id"]

    question = item["question"]

    golden_answer = item["answer"]

    expected_pages = [
        int(page)
        for page in item["source_pages"]
    ]


    # ========================================================
    # 1. Retrieve documents
    # ========================================================

    results = retriever.retrieve(
        query=question,
        candidate_k=20,
        final_k=3
    )


    # ========================================================
    # 2. Get retrieved pages
    # ========================================================

    retrieved_pages = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        page = metadata.get(
            "page_label"
        )

        if page is not None:

            try:

                page = int(page)

                retrieved_pages.append(
                    page
                )

            except ValueError:

                pass


    retrieved_pages = list(
        dict.fromkeys(
            retrieved_pages
        )
    )


    # ========================================================
    # 3. Retrieval Evaluation
    # ========================================================

    retrieval_ok = any(
        page in retrieved_pages
        for page in expected_pages
    )


    if retrieval_ok:

        retrieval_passed += 1

    else:

        retrieval_failed += 1


    # ========================================================
    # 4. Build context from exact retrieved chunks
    # ========================================================

    context = "\n\n".join(
        result["text"]
        for result in results
    )


    # ========================================================
    # 5. Generate answer
    # ========================================================

    answer_prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I couldn't find that information in the uploaded document."

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    answer_response = llm.invoke(
        answer_prompt
    )

    answer = answer_response.content.strip()


    # ========================================================
    # 6. Faithfulness Evaluation
    # ========================================================

    faithful = evaluate_faithfulness(
        question,
        context,
        answer
    )


    if faithful:

        faithfulness_passed += 1

    else:

        faithfulness_failed += 1


    # ========================================================
    # 7. Correctness Evaluation
    # ========================================================

    correct = evaluate_correctness(
        question,
        golden_answer,
        answer
    )


    if correct:

        correctness_passed += 1

    else:

        correctness_failed += 1


    # ========================================================
    # 8. Print Question Result
    # ========================================================

    print()
    print("--------------------------------------------")

    print(
        f"Question {question_id}"
    )

    print(
        "Question:",
        question
    )

    print()

    print(
        "Expected pages:",
        expected_pages
    )

    print(
        "Retrieved pages:",
        retrieved_pages
    )

    print()

    print(
        "Retrieval:",
        "PASS" if retrieval_ok else "FAIL"
    )

    print()

    print(
        "Generated answer:"
    )

    print(
        answer
    )

    print()

    print(
        "Golden answer:"
    )

    print(
        golden_answer
    )

    print()

    print(
        "Faithfulness:",
        "PASS" if faithful else "FAIL"
    )

    print()

    print(
        "Correctness:",
        "PASS" if correct else "FAIL"
    )


# ============================================================
# Calculate Scores
# ============================================================

total = len(dataset)


retrieval_score = (
    retrieval_passed / total * 100
    if total > 0
    else 0
)


faithfulness_score = (
    faithfulness_passed / total * 100
    if total > 0
    else 0
)


correctness_score = (
    correctness_passed / total * 100
    if total > 0
    else 0
)

# ============================================================
# Quality Gate
# ============================================================

quality_gate_passed = (
    retrieval_score >= MIN_RETRIEVAL_SCORE
    and
    faithfulness_score >= MIN_FAITHFULNESS_SCORE
    and
    correctness_score >= MIN_CORRECTNESS_SCORE
)

# ============================================================
# Final Evaluation Summary
# ============================================================

print()
print("============================================")
print("EVALUATION SUMMARY")
print("============================================")

print(
    f"Total questions       : {total}"
)

print()

print(
    f"Retrieval passed      : {retrieval_passed}"
)

print(
    f"Retrieval failed      : {retrieval_failed}"
)

print(
    f"Retrieval score       : {retrieval_score:.2f}%"
)

print()

print(
    f"Faithfulness passed   : {faithfulness_passed}"
)

print(
    f"Faithfulness failed   : {faithfulness_failed}"
)

print(
    f"Faithfulness score    : {faithfulness_score:.2f}%"
)

print()

print(
    f"Correctness passed    : {correctness_passed}"
)

print(
    f"Correctness failed    : {correctness_failed}"
)

print(
    f"Correctness score     : {correctness_score:.2f}%"
)

print("============================================")

print()

print(
    "============================================"
)

print(
    "QUALITY GATE"
)

print(
    "============================================"
)

print(
    f"Retrieval threshold     : "
    f"{MIN_RETRIEVAL_SCORE:.2f}%"
)

print(
    f"Faithfulness threshold  : "
    f"{MIN_FAITHFULNESS_SCORE:.2f}%"
)

print(
    f"Correctness threshold   : "
    f"{MIN_CORRECTNESS_SCORE:.2f}%"
)

print()

if quality_gate_passed:

    print(
        "QUALITY GATE: PASS"
    )

    sys.exit(0)

else:

    print(
        "QUALITY GATE: FAIL"
    )

    sys.exit(1)