from typing import List
from langchain_core.documents import Document

from interfaces.document_creator import DocumentCreator

class BasicDocumentCreator(DocumentCreator):

    def create_document(self, title: str, summary: str, topics: List[str], text: str, url: str) -> Document:
        content = f"Summary: {summary}\nTopics: {', '.join(topics)}"
        return Document(
            page_content=content,
            metadata={"title": title, "summary": summary, "topics": topics, "text": text, "url": url}
        )