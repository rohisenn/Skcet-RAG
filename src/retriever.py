import os
from langchain_community.retrievers import BM25Retriever

def get_retriever(vectordb, documents=None, k=5):

    vector_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    if not documents:
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(
        documents=documents,
        k=k
    )

    def hybrid_retrieve(query):
        vector_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        seen = set()
        results = []

        # Prioritize vector results first
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)

        return results[:k]

    class HybridRetriever:
        def invoke(self, query):
            return hybrid_retrieve(query)

    return HybridRetriever()
