# Web Content Search and Indexing

This project is a tool for indexing web content from a list of URLs and performing searches using three modes: `basic` (direct similarity search), `rag` (Retrieval-Augmented Generation), and `contextual` (RAG with query augmentation based on search history). It supports two indexing backends: FAISS and Chroma. The tool extracts content from web pages, analyzes it, and builds a searchable index.

## Features
- **Index Creation**: Extracts titles, text, summaries, topics, and URLs from web pages, building an index using FAISS or Chroma.
- **Search Modes**:
  - **Basic Search**: Performs direct similarity search, returning title, URL, summary, and topics.
  - **RAG Search**: Uses Retrieval-Augmented Generation for contextual responses, including title, URL, summary, and content.
  - **Contextual Search**: Enhances RAG by augmenting queries with search history for better relevance.
- **Command-Line Interface**: Configurable via arguments for index type, search type, query, history file, and more.
- **Logging**: Optional logging to console and `app.log` for debugging, including query augmentation and search results.
- **Search History**: Stores queries in `search_history.json` for contextual mode augmentation.

## Prerequisites
- Python 3.8 or higher
- External dependencies (see [Installation](#installation))
- An LLM API client (e.g., OpenAI) for query augmentation and RAG search

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ExponentialCat/NewsParser.git
   cd repo
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your LLM API (e.g., set OpenAI API key as an environment variable):
   ```bash
   export GENAI_API_KEY='your-api-key'  # On Windows: set GENAI_API_KEY=your-api-key
   ```

## Usage
1. Create a file named `urls.txt` with a list of URLs (one per line) to index:
   ```
   https://example.com/page1
   https://example.com/page2
   ```

2. Run the script with desired options. Use `--logging` for detailed output:
   ```bash
   python main.py --index FAISS --search contextual --query "machine learning" --logging
   ```

   **Command-Line Arguments**:
   - `--index`: Index type (`FAISS` or `Chroma`). Required.
   - `--search`: Search type (`basic`, `rag`, or `contextual`). Default: `basic`.
   - `--query`: Search query string. Default: `\"\"`.
   - `--rebuild`: Force rebuild of the index. Optional.
   - `--logging`: Enable logging to console and `app.log`. Optional.
   - `--history_file`: File to store search history (for contextual mode). Default: `search_history.json`.
   - `--max_history`: Max number of history entries to use (for contextual mode). Default: `5`.

   Examples:
   - Basic search with FAISS:
     ```bash
     python main.py --index FAISS --search basic --query "neural networks" --logging
     ```
   - Contextual search with Chroma and rebuild:
     ```bash
     python main.py --index Chroma --search contextual --query "artificial intelligence" --rebuild --logging
     ```

## Output
- **Logs**: If `--logging` is enabled, logs are written to `app.log` and printed to the console, detailing index creation, query augmentation, search results, and errors.
- **Search Results**:
  - For `basic` search, results include the title, URL, summary, and topics of matching documents. Example:
    ```
    Best Result:
    Title: ML Guide
    URL: https://example.com/ml
    Summary: Intro to ML
    Topics: ML, AI
    ```
  - For `rag` or `contextual` search, results include the response from LLM about the best document.
  - If no results are found, a message like \"No results found for [basic/RAG] search\" is logged.
- **Search History**: For `rag` and `contextual` modes, queries are saved to `search_history.json` for future augmentation.

## Project Structure
```
├── main.py                # Main script with CLI and core logic
├── interfaces/            # Abstract classes for core components
│   ├── analyser.py        # Abstract class for content analysis
│   ├── document_creator.py  # Abstract class for document creation
│   ├── extractor.py       # Abstract class for content extraction
│   └── store.py           # Abstract class for index storage
├── implementations/       # Implementation modules
│   ├── basic_document_creator.py  # Document creation logic
│   ├── stores/            # Index storage backends
│   │   ├── chroma_store.py
│   │   └── faiss_store.py
│   ├── genai_analyser.py  # LLM-based analysis and query augmentation
│   └── html_content_extractor.py  # Web content extraction
├── utils/                 # Utility functions
│   └── text_utils.py      # URL loading and text utilities
├── urls.txt               # Input file with URLs
├── search_history.json    # Search history file (created in rag/contextual modes)
├── app.log                # Log file (created if --logging is used)
└── requirements.txt       # Project dependencies
```

## Troubleshooting
- **No output**: Run with `--logging` to check `app.log` for errors (e.g., invalid URLs, empty index, or LLM API issues).
- **Empty index**: Ensure `urls.txt` contains valid URLs and use `--rebuild` to recreate the index.
- **LLM errors**: Verify your API key and internet connection. Check `app.log` for messages.