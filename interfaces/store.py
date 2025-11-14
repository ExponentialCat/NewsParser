from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class VectorStore(ABC):

    @abstractmethod
    def build_index(self, documents: List[dict]) -> None:
        pass

    @abstractmethod
    def save_index(self) -> None:
        pass

    @abstractmethod
    def load_index(self) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict]:
        pass