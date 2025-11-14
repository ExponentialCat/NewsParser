import logging
import re
from typing import List

def extract_json(text: str) -> str:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        logging.warning("No JSON found in the response text.")
        return "{}"
    except Exception as e:
        logging.error("Error extracting JSON from the response text: %s", e)
        return "{}"

def load_urls_from_file(file_path: str) -> List:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file.readlines()]
            logging.info(f"Successfully loaded {len(urls)} URLs from {file_path}")
            return urls
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []