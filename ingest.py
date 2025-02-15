import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATASET = "dataset/"
FAISS_INDEX = "vectorstore/"

def embed_all():
    """
    Embed all files in the dataset directory
    """
    documents = []
    
    # Load PDFs manually
    for file in os.listdir(DATASET):
        if file.endswith(".pdf"):
            pdf_loader = PyPDFLoader(os.path.join(DATASET, file))
            documents.extend(pdf_loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(FAISS_INDEX)

if __name__ == "__main__":
    embed_all()
