import json
import os
import openai
import logging
from openai import OpenAI
from interfaces.analyzer import Analyzer
from langchain_openai import OpenAIEmbeddings

from interfaces.store import VectorStore
from utils.text_utils import extract_json

GENAI_API_KEY = os.getenv("GENAI_API_KEY", "")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable not set.")

class GenAIAnalyzer(Analyzer):
    def __init__(self, model="openai/gpt-4o", httpReferer="https://openrouter.ai/api/v1"):
        self.model = model
        self.client = OpenAI(
            base_url=httpReferer,
            api_key=GENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-large",
            openai_api_key=GENAI_API_KEY,
            openai_api_base=httpReferer,
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": httpReferer
                },
                "encoding_format": "float"
            }
        )

    def analyze(self, title: str, text: str) -> tuple:
        prompt = f"""
        You are a news analysis expert. Provide a short summary and a list of key topics for the following news article.

        Title: {title}
        Text: {text}

        Return the result in JSON format:
        {{
            "summary": "Short summary of the news",
            "topics": ["topic1", "topic2", ...]
        }}
        """

        try:
            logging.info("Sending request to GenAI for analysis.")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            answer_text = completion.choices[0].message.content.strip()
            logging.info("Received response from GenAI.")

            json_text = extract_json(answer_text)  # Очищаем текст, если есть лишние символы
            result = json.loads(json_text)

            summary = result.get("summary", "")
            topics = result.get("topics", [])

            logging.info("Analysis completed successfully.")
            return summary, topics

        except json.JSONDecodeError as e:
            logging.error("JSON parsing error: %s", e)
            return "Error", [], f"Failed to parse JSON from the model response: {e}"

        except openai.error.OpenAIError as e:
            logging.error("OpenAI API error: %s", e)
            return "Error", [], f"OpenAI API error: {e}"

        except Exception as e:
            logging.error("Unexpected error: %s", e)
            return "Error", [], f"Unexpected error: {e}"

    def get_embedding_model(self):
        return self.embeddings

    def perform_rag_search(self, store: VectorStore, query: str, k: int = 3) -> str:
        try:
            logging.info("Performing RAG search.")
            results = store.search(query, k)

            context = "\n\n".join([f"{result['title']}: {result['summary']}\nTopics: {result['topics']}" for i, result in enumerate(results)])

            prompt = f"""
                    Based on the following news, please choose the one that best answers the query below.
    
                    {context}
    
                    Question: {query}
    
                    Answer: 
                    """

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            answer_text = completion.choices[0].message.content.strip()
            logging.info("Received response from OpenAI.")
        except Exception as e:
            logging.error("Unexpected error: %s", e)
            return f"Unexpected RAG error: {e}"

        return answer_text
