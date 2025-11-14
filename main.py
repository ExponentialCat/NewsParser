import logging
import argparse
from implementations.basic_document_creator import BasicDocumentCreator
from implementations.stores.chroma_store import ChromaStore
from implementations.stores.faiss_store import FAISSStore
from implementations.genai_analyser import GenAIAnalyzer
from implementations.html_content_extractor import HTMLContentExtractor
from utils.text_utils import load_urls_from_file

def setup_logging(enable_logging=True):
    if enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log')
            ]
        )
    else:
        logging.disable(logging.CRITICAL)

def main():
    parser = argparse.ArgumentParser(description="Choose options for the search and indexing process.")
    parser.add_argument('--logging', action='store_true', help="Enable logging")
    parser.add_argument('--index', choices=['FAISS', 'Chroma'], required=True,
                        help="Index type to use (FAISS or Chroma)")
    parser.add_argument('--rebuild', action='store_true',
                        help="Force rebuild of the index")

    args = parser.parse_args()
    setup_logging(args.logging)

    # Инициализация компонентов
    urls = load_urls_from_file("urls.txt")
    if not urls:
        logging.error("No URLs found in urls.txt. Exiting.")
        return

    extractor = HTMLContentExtractor()
    analyzer = GenAIAnalyzer()

    try:
        embedding_model = analyzer.get_embedding_model()
        if embedding_model is None:
            raise ValueError("Embedding model is None")
    except Exception as e:
        logging.error(f"Failed to initialize embedding model: {str(e)}. Exiting.")
        return

    document_creator = BasicDocumentCreator()

    try:
        if args.index == "FAISS":
            store = FAISSStore(embedding_model=embedding_model)
            store.load_index()
        elif args.index == "Chroma":
            store = ChromaStore(embedding_model=embedding_model)
    except Exception as e:
        logging.error(f"Failed to initialize {args.index} store: {str(e)}. Exiting.")
        return

    if not store.index_exists() or args.rebuild:
        logging.info("Index not found or rebuild requested, creating documents...")
        documents = []
        for url in urls:
            try:
                title, text = extractor.extract(url)
                if not text or not title:
                    logging.warning(f"Failed to extract content from {url}: title={title}, text_length={len(text) if text else 0}")
                    continue
                summary, topics = analyzer.analyze(title, text)
                if not summary or not topics:
                    logging.warning(f"Analysis failed for {url}: summary={summary}, topics={topics}")
                    continue
                document = document_creator.create_document(title, summary, topics, text)
                documents.append(document)
                logging.info(f"Document created for {url}: title={title}")
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
                continue

        if documents:
            logging.info("Building index with %d documents...", len(documents))
            try:
                store.build_index(documents)
            except Exception as e:
                logging.error(f"Failed to build index: {str(e)}. Exiting.")
                return
        else:
            logging.error("No valid documents created. Cannot build index. Exiting.")
            return
    else:
        logging.info("Index exists, using existing index.")

    query = "young"
    logging.info("Searching for query: %s", query)
    try:
        results = store.search(query)
        if results:
            logging.info("Search results: %s", results)
        else:
            logging.info("No results found.")
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")

if __name__ == "__main__":
    main()