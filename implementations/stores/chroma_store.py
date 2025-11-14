import logging
import os
from typing import List
from uuid import uuid4

import chromadb
from langchain_core.documents import Document

from interfaces.store import VectorStore


class ChromaStore(VectorStore):
    def __init__(self, embedding_model, index_path="chroma_index"):
        self.index_path = index_path
        self.client = chromadb.Client()
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=self.index_path)
        self.client.heartbeat()
        self.collection = self.client.get_or_create_collection("collection")
        logging.info(f"Initial collection count after init: {self.collection.count()}")

    def index_exists(self) -> bool:
        if not os.path.exists(self.index_path):
            logging.warning(f"Index directory {self.index_path} does not exist.")
            return False

        collections = [col.name for col in self.client.list_collections()]
        if "collection" not in collections:
            logging.warning("Collection 'collection' not found in list_collections().")
            return False

        count = self.collection.count()
        if count == 0:
            logging.warning(
                "Collection exists, but count() == 0. Possible ChromaDB loading issue. Treating as existing to avoid rebuild.")
            return True

        logging.info(f"Index exists with {count} documents.")
        return True

    def build_index(self, documents: List[Document]) -> None:
        try:
            logging.info("Building Chroma index...")

            collections = [col.name for col in self.client.list_collections()]
            if "collection" in collections:
                logging.info("Deleting existing collection 'collection' before rebuild.")
                self.client.delete_collection("collection")
                self.collection = self.client.get_or_create_collection("collection")

            embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
            texts = [doc.page_content for doc in documents]
            metadatas = [{"title": doc.metadata["title"], "topics": ", ".join(doc.metadata["topics"]),
                          "summary": doc.metadata["summary"], "url": doc.metadata["url"]} for doc in documents]
            ids = [str(uuid4()) for _ in documents]

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            self.client.heartbeat()

            logging.info("Chroma index successfully built.")
        except Exception as e:
            logging.error(f"Error building Chroma index: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[dict]:
        try:
            query_embedding = self.embedding_model.embed_query(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

            return results["metadatas"][0]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

