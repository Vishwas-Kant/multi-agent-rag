"""
Web Reader Tool — Extract readable text content from any URL.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_CONTENT_LENGTH = 4000


@tool
def read_webpage(url: str) -> str:
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_links=False,
                include_tables=True,
                favor_recall=True,
            )
            if text and text.strip():
                content = text.strip()
                if len(content) > _MAX_CONTENT_LENGTH:
                    content = content[:_MAX_CONTENT_LENGTH] + "\n\n[...content truncated]"
                return f"Content from {url}:\n\n{content}"
    except ImportError:
        pass
    except Exception as e:
        logger.warning("trafilatura failed for %s: %s", url, e)

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if len(text) > _MAX_CONTENT_LENGTH:
            text = text[:_MAX_CONTENT_LENGTH] + "\n\n[...content truncated]"
        return f"Content from {url}:\n\n{text}"

    except Exception as e:
        logger.exception("Web reader failed for %s", url)
        return f"Error reading webpage '{url}': {str(e)}"
