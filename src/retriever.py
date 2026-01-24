import os
from langchain_community.retrievers import BM25Retriever

def get_retriever(vectordb, documents=None, k=3):

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
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_retriever.invoke(query)

        seen = set()
        results = []

        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)

        return results[:k]

    class HybridRetriever:
        def invoke(self, query):
            return hybrid_retrieve(query)

    return HybridRetriever()
