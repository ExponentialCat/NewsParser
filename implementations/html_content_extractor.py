from interfaces.extractor import Extractor
import requests
from bs4 import BeautifulSoup
import logging

class HTMLContentExtractor(Extractor):
    def extract(self, url: str) -> tuple[str, str]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, 'html.parser')

            title_tag = soup.find('h1')
            if not title_tag:
                title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else "No title"

            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
            else:
                divs = soup.find_all('div')
                max_p_count = 0
                best_div = None
                for div in divs:
                    p_count = len(div.find_all('p'))
                    if p_count > max_p_count:
                        max_p_count = p_count
                        best_div = div
                paragraphs = best_div.find_all('p') if best_div else []

            text = "\n".join(p.get_text(strip=True) for p in paragraphs)

            return title, text

        except requests.exceptions.RequestException as req_error:

            logging.exception(f"Request error occurred while fetching URL {url}: {str(req_error)}")

            return "", ""

        except Exception as e:

            logging.exception(f"Error occurred while extracting content from URL {url}: {str(e)}")

            return "", ""