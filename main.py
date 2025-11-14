import logging

from implementations.basic_document_creator import BasicDocumentCreator
from implementations.stores.faiss_store import FAISSStore
from implementations.genai_analyser import GenAIAnalyzer
from implementations.html_content_extractor import HTMLContentExtractor
from utils.text_utils import load_urls_from_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

urls = load_urls_from_file("urls.txt")
extractor = HTMLContentExtractor()

analyzer = GenAIAnalyzer()
embedding_model = analyzer.get_embedding_model()

document_creator = BasicDocumentCreator()
documents = []

for url in urls:
    title, text = extractor.extract(url)
    if text:
        summary, topics = analyzer.analyze(title, text)
        document = document_creator.create_document(title, summary, topics, text)
        documents.append(document)

store = FAISSStore(embedding_model=embedding_model)
store.build_index(documents)
