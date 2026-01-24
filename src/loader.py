from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

def load_documents(data_dir="data"):
    documents = []

    pdf_dir = os.path.join(data_dir, "pdfs")
    txt_dir = os.path.join(data_dir, "texts")

    if os.path.exists(pdf_dir):
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                documents.extend(loader.load())

    if os.path.exists(txt_dir):
        for file in os.listdir(txt_dir):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(txt_dir, file))
                documents.extend(loader.load())

    return documents
