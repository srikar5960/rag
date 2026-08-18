from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class ChatBot:

    def __init__(self, retriever):

        self.retriever = retriever

        self.llm = ChatGroq(
            groq_api_key="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",
            model_name="openai/gpt-oss-20b",
            temperature=0.1,
            max_tokens=1024
        )

    def ask(self, question):

        # Retrieve relevant chunks
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=3
        )

        # No documents found
        if not retrieved_docs:
            return "I couldn't find relevant information in the uploaded document."


        # Build context
        context = "\n\n".join(
            doc["content"] for doc in retrieved_docs
        )

        # Prompt
        prompt = f"""
You are a helpful PDF assistant.

Answer ONLY from the context provided.

If the answer is not available in the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            prompt.format(
                context=context,
                query=question
            )
        )

        return response.content