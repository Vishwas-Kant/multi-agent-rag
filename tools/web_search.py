"""
Web Search Tool — DuckDuckGo-based search (no API key required).
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(r)

        if not results:
            return f"No search results found for: '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", r.get("link", ""))
            snippet = r.get("body", r.get("snippet", ""))
            formatted.append(f"{i}. **{title}**\n   URL: {url}\n   {snippet}")

        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted)

    except ImportError:
        return "Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        logger.exception("Web search failed")
        return f"Error performing web search: {str(e)}"
