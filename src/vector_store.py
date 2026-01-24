from langchain_community.vectorstores import Chroma
from src.config import CHROMA_PERSIST_DIR

def create_vector_store(chunks, embedding_model):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectordb.persist()
    return vectordb


def load_vector_store(embedding_model):
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model
    )
