import logging
import argparse
import json
import datetime
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

def load_history(history_file, max_history=5):
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history[-max_history:]  # Лимит на последние N
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        logging.warning("Invalid history file format. Starting fresh.")
        return []

def save_history(history_file, query):
    history = load_history(history_file)
    history.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'query': query
    })
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Choose options for the search and indexing process.")
    parser.add_argument('--logging', action='store_true', help="Enable logging")
    parser.add_argument('--index', choices=['FAISS', 'Chroma'], required=True,
                        help="Index type to use (FAISS or Chroma)")
    parser.add_argument('--rebuild', action='store_true',
                        help="Force rebuild of the index")
    parser.add_argument('--search', choices=['basic', 'rag', 'contextual'], default='basic',
                        help="Search type: 'basic' for direct search, 'rag' for RAG-based search, 'contextual' for history-based augmentation")
    parser.add_argument('--query', type=str, default="",
                        help="Search query for basic or RAG search")
    parser.add_argument('--history_file', type=str, default="search_history.json",
                        help="File to store search history (for contextual mode)")
    parser.add_argument('--max_history', type=int, default=5,
                        help="Max number of history entries to use (for contextual mode)")

    args = parser.parse_args()
    setup_logging(args.logging)

    if not args.query:
        logging.info("No query provided. Exiting.")
        return

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
                document = document_creator.create_document(title, summary, topics, text, url)
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

    query = args.query

    if args.search == 'contextual':
        logging.info("Contextual mode: Loading search history...")
        history = load_history(args.history_file, args.max_history)
        if history:
            history_queries = [entry['query'] for entry in history]
            augmented_query = analyzer.augment_query_with_history(history_queries, query)
            logging.info(f"Augmented query: {augmented_query}")
            query = augmented_query
        else:
            logging.info("No history found. Falling back to RAG search.")

    if args.search == "basic":
        logging.info("Performing basic search for query: %s", query)
        try:
            results = store.search(query)
            if results:
                best_result = results[0]
                topics = best_result['topics']
                if isinstance(topics, list):
                    topics = ", ".join(topics)
                best_result_str = (
                    f"Best Result:\n"
                    f"Title: {best_result['title'].encode('utf-8', errors='replace').decode('utf-8')}\n"
                    f"URL: {best_result.get('url', 'N/A').encode('utf-8', errors='replace').decode('utf-8')}\n"
                    f"Summary: {best_result['summary'].encode('utf-8', errors='replace').decode('utf-8')}\n"
                    f"Topics: {topics.encode('utf-8', errors='replace').decode('utf-8')}"
                )

                other_results_str = ""
                if len(results) > 1:
                    other_results = [
                        f"Result {i + 1}:\n"
                        f"Title: {result['title'].encode('utf-8', errors='replace').decode('utf-8')}\n"
                        f"URL: {result.get('url', 'N/A').encode('utf-8', errors='replace').decode('utf-8')}\n"
                        f"Summary: {result['summary'].encode('utf-8', errors='replace').decode('utf-8')}\n"
                        f"Topics: {', '.join(result['topics']) if isinstance(result['topics'], list) else result['topics']}"
                        for i, result in enumerate(results[1:])
                    ]
                    other_results_str = "\n\n" + "\n\n".join(other_results)

                formatted_results = best_result_str + other_results_str
                logging.info("Basic search results:\n%s", formatted_results)
            else:
                logging.info("No results found for basic search.")
        except Exception as e:
            logging.error(f"Basic search failed: {str(e)}")
    else:
        logging.info("Performing RAG search for query: %s", query)
        try:
            result = analyzer.perform_rag_search(store, query)
            logging.info("RAG search result: %s", result)
        except Exception as e:
            logging.error(f"RAG search failed: {str(e)}")

    if args.search in ['rag', 'contextual']:
        save_history(args.history_file, args.query)

if __name__ == "__main__":
    main()