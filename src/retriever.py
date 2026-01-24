#from langchain_community.retrievers import BM25Retriever, EnsembleRetriever

#def get_retriever(vectordb, documents):
 #   vector_retriever = vectordb.as_retriever(
 #       search_type="similarity",
  #      search_kwargs={"k": 5}
   # )

    #bm25_retriever = BM25Retriever.from_documents(
     #   documents=documents,
      #  k=5
    #)

    #return EnsembleRetriever(
     #   retrievers=[bm25_retriever, vector_retriever],
      #  weights=[0.4, 0.6]
    #)
from langchain_community.retrievers import BM25Retriever

def get_retriever(vectordb, documents, k=3):

    vector_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

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

