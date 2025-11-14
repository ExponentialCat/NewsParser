import logging
from typing import List
from uuid import uuid4

import chromadb
from langchain_core.documents import Document

from interfaces.store import VectorStore


class ChromaStore(VectorStore):
    def __init__(self, embedding_model):
        self.client = chromadb.Client()
        self.embedding_model = embedding_model
        self.collection = self.client.get_or_create_collection("collection")

    def save_index(self) -> None:
        print("1")

    def load_index(self) -> None:
        print("2")

    def build_index(self, documents: List[Document]) -> None:
        try:
            logging.info("Building Chroma index...")

            embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
            texts = [doc.page_content for doc in documents]
            metadatas = [{"title": doc.metadata["title"], "topics": ", ".join(doc.metadata["topics"]),
                          "text": doc.metadata["text"]} for doc in documents]
            ids = [str(uuid4()) for _ in documents]

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            #self.save_index()

            logging.info("Chroma index successfully built.")
        except Exception as e:
            logging.error(f"Error building Chroma index: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[dict]:
        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results["metadatas"]