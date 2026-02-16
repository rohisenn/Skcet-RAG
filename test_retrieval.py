from src.loader import load_documents
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever

# Test document loading
docs = load_documents()
print(f"Loaded {len(docs)} documents")
if docs:
    print(f"\nFirst document preview:\n{docs[0].page_content[:200]}")

# Test retrieval
embedding_model = get_embedding_model()
vectordb = load_vector_store(embedding_model)
retriever = get_retriever(vectordb, docs)

# Test query
query = "Who is the HOD of CSE?"
results = retriever.invoke(query)
print(f"\n\nQuery: {query}")
print(f"Retrieved {len(results)} documents\n")
for i, doc in enumerate(results[:3]):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
