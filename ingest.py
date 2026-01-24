from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

docs = load_documents()
chunks = chunk_documents(docs)

embedding_model = get_embedding_model()
create_vector_store(chunks, embedding_model)

print("Documents indexed successfully")
