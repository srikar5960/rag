from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()



class ChatBot:

    def __init__(self, retriever):

        self.retriever = retriever

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            max_tokens=1024
        )

    def ask(self, question):

        retrieved_docs = self.retriever.retrieve(
            query=question,
            candidate_k=20,
            final_k=3
        )

        if not retrieved_docs:
            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the uploaded document."
                ),
                "sources": []
            }

        # Build context
        context = "\n\n".join(
            doc["text"]
            for doc in retrieved_docs
        )

        # Generate answer
        prompt = f"""
    You are a helpful PDF assistant.

    Answer ONLY using the provided context.

    Do not use outside knowledge.

    If the answer cannot be found in the context, say:

    "I couldn't find that information in the uploaded document."

    Give a clear and direct answer.

    Context:
    ----------------
    {context}
    ----------------

    Question:
    {question}

    Answer:
    """

        response = self.llm.invoke(prompt)

        answer = response.content

        # Build structured sources
        sources = []

        seen = set()

        for doc in retrieved_docs:

            metadata = doc.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
                "page_label",
                metadata.get("page", "?")
            )

            filename = os.path.basename(
                source
            )

            key = (
                filename,
                str(page)
            )

            if key not in seen:
                seen.add(key)

                sources.append({
                    "file": filename,
                    "page": int(page)
                    if str(page).isdigit()
                    else page
                })

        return {
            "answer": answer,
            "sources": sources
        }