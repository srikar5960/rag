import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:

    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def load_pdf(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        return documents

    def split_documents(self, documents):

        chunks = self.text_splitter.split_documents(
            documents
        )

        return chunks

    def load_and_split(self, file_path):

        documents = self.load_pdf(file_path)

        chunks = self.split_documents(
            documents
        )

        return chunks