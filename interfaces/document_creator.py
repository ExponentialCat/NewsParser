from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List


class DocumentCreator(ABC):
    @abstractmethod
    def create_document(self, title: str, summary: str, topics: List[str], text: str, url: str) -> Document:
        pass