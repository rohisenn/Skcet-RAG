from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag
from src.memory import ConversationMemory
from src.loader import load_documents

docs=load_documents()
embedding_model = get_embedding_model()
vectordb = load_vector_store(embedding_model)
retriever = get_retriever(vectordb,docs)
memory = ConversationMemory(max_turns=5)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    answer = run_rag(query, retriever, memory)
    print("\nAnswer:", answer)
