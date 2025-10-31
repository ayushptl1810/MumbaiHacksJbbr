import requests
import logging
from urllib.parse import urlparse
from typing import Optional, Tuple
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document
import hashlib

logger = logging.getLogger(__name__)


class WebContentScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
        self.timeout = 10
        self.max_content_length = 10000

    def is_scrapeable_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        if 'reddit.com' in url or 'redd.it' in url:
            return False
        media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.mp3', '.pdf']
        if any(url.lower().endswith(ext) for ext in media_extensions):
            return False
        social_domains = ['twitter.com', 'x.com', 'facebook.com', 'instagram.com', 'tiktok.com']
        parsed = urlparse(url)
        if any(domain in parsed.netloc.lower() for domain in social_domains):
            return False
        return True

    def scrape_with_trafilatura(self, url: str) -> Optional[str]:
        try:
            head = self.session.head(url, timeout=5, allow_redirects=True)
            ctype = head.headers.get('Content-Type', '').lower()
            if ctype and not any(t in ctype for t in ('html', 'text', 'xml')):
                logger.debug(f"Skip non-HTML content-type: {ctype} for {url}")
                return None
        except Exception:
            pass

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.warning(f"requests.get failed for {url}: {e}")
            return None

        try:
            article = Article(url)
            article.download()
            article.parse()
            text = f"Title: {article.title}\n\nContent: {article.text}"
            if text and len(text.strip()) > 200:
                cleaned = ' '.join(text.split())
                logger.info(f"scraper: newspaper extracted {len(cleaned)} chars from {url}")
                return cleaned[:self.max_content_length]
        except Exception as e:
            logger.debug(f"newspaper extraction failed for {url}: {e}")

        try:
            doc = Document(html)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, 'html.parser')
            summary_text = ' '.join(soup.get_text(separator=' ').split())
            if summary_text and len(summary_text) > 200:
                logger.info(f"scraper: readability extracted {len(summary_text)} chars from {url}")
                return summary_text[:self.max_content_length]
        except Exception as e:
            logger.debug(f"readability extraction failed for {url}: {e}")

        try:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            body_text = ' '.join(soup.get_text(separator=' ').split())
            if body_text and len(body_text) > 200:
                logger.info(f"scraper: bsoup body extracted {len(body_text)} chars from {url}")
                return body_text[:self.max_content_length]
        except Exception as e:
            logger.debug(f"BeautifulSoup body extraction failed for {url}: {e}")

        logger.info(f"No usable content extracted for {url}")
        return None

    def scrape_content(self, url: str) -> Tuple[Optional[str], str]:
        if not self.is_scrapeable_url(url):
            return None, "not_scrapeable"

        content = self.scrape_with_trafilatura(url)
        if content and len(content.strip()) > 100:
            return content, 'trafilatura'

        return None, 'failed'
