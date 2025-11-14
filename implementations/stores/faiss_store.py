import faiss
import os
import logging
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from interfaces.store import VectorStore


class FAISSStore(VectorStore):
    def __init__(self, embedding_model, index_path="faiss_index"):
        self.index_path = index_path
        self.index = None
        self.embedding_model = embedding_model

    def build_index(self, documents: List[Document]) -> None:
        try:
            logging.info("Building FAISS index...")

            self.index = FAISS.from_documents(documents=documents, embedding=self.embedding_model)
            #self.save_index()

            logging.info("FAISS index successfully built.")
        except Exception as e:
            logging.error(f"Error building FAISS index: {str(e)}")

    def save_index(self) -> None:
        if self.index is not None:
            try:
                self.index.save_local(self.index_path)
                logging.info(f"Index saved at {self.index_path}")
            except Exception as e:
                logging.error(f"Error saving FAISS index: {str(e)}")
        else:
            logging.warning("Index is None, cannot save.")

    def load_index(self) -> None:
        if os.path.exists(self.index_path):
            try:
                self.index = FAISS.load_local(self.index_path)
                logging.info(f"Index loaded from {self.index_path}")
            except Exception as e:
                logging.error(f"Error loading FAISS index: {str(e)}")
        else:
            logging.warning("No index file found, need to build a new one.")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if self.index is None:
            logging.warning("Index is not loaded. Please load or build the index first.")
            return []

        try:
            results = self.index.similarity_search_with_score(query, k=k)
            return [
                     {
                         "title": r[0].metadata["title"],
                         "summary": r[0].metadata["summary"],
                         "topics": r[0].metadata["topics"],
                     }
                     for r in results
                 ]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []
