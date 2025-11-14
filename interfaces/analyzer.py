from abc import ABC, abstractmethod

from interfaces.store import VectorStore


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, title: str, text: str) -> tuple[str, list[str]]:
        pass

    @abstractmethod
    def get_embedding_model(self):
        pass

    @abstractmethod
    def perform_rag_search(self, store: VectorStore, query: str, k: int = 3) -> str:
        pass
