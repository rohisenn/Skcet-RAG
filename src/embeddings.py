#from langchain_huggingface import HuggingFaceEmbeddings
#from src.config import EMBEDDING_MODEL

#def get_embedding_model():
 #   return HuggingFaceEmbeddings(
  #      model_name=EMBEDDING_MODEL
   # )
from fastembed import TextEmbedding
from typing import List
import numpy as np

class FastEmbedWrapper:
    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [[float(x) for x in emb] for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = list(next(self.model.embed([text])))
        return [float(x) for x in embedding]


def get_embedding_model():
    return FastEmbedWrapper()
