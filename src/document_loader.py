###Document Loader

from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(pdf_path):

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    return documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Split LangChain documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Display an example chunk
    if split_docs:
        print("\nExample Chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs
