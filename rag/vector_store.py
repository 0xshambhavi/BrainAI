from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import numpy as np


class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper around SentenceTransformer.
    """

    def __init__(self, model):
        self.st_model = model
        self.model = "all-MiniLM-L6-v2"

    def embed_documents(self, texts):
        vectors = self.st_model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text):
        vector = self.st_model.encode([text], convert_to_numpy=True)[0]
        return vector.tolist()


def build_vector_store(chunks, sentence_transformer_model):
    """
    Build a FAISS vector store from text chunks.
    """

    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = SentenceTransformerEmbeddings(sentence_transformer_model)

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
