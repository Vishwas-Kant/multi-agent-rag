"""
Summarizer Tool — LLM-based text summarization using the shared local model.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def summarize_text(text: str, style: str = "concise") -> str:
    if not text or not text.strip():
        return "Error: No text provided for summarization."

    if len(text) < 100:
        return text

    try:
        from utils.llm import get_llm

        llm = get_llm()

        if style == "detailed":
            prompt = (
                "Provide a detailed summary of the following text. "
                "Cover all key points, facts, and conclusions.\n\n"
                f"TEXT:\n{text[:3000]}\n\nDETAILED SUMMARY:"
            )
        else:
            prompt = (
                "Summarize the following text in 2-3 concise sentences. "
                "Focus on the most important information.\n\n"
                f"TEXT:\n{text[:3000]}\n\nCONCISE SUMMARY:"
            )

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()

    except Exception as e:
        logger.exception("Summarization failed")
        truncated = text[:500].rsplit(".", 1)[0] + "."
        return f"(Extractive summary — LLM unavailable): {truncated}"
