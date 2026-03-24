"""
Document Agent — Specialist for PDF/document Q&A and analysis.
"""

from __future__ import annotations

from agents.base import BaseAgent
from tools.rag import retrieve_context
from tools.summarizer import summarize_text


_SYSTEM_PROMPT = """You are a Document Analysis Specialist AI agent.
Your job is to answer questions about uploaded PDF documents accurately.

AVAILABLE TOOLS:
1. retrieve_context(query) — Search uploaded PDF documents for relevant information.
   You MUST call this for EVERY query to find document content.
2. summarize_text(text, style) — Summarize long text passages.
   Use when retrieved content is lengthy and needs condensing.

WORKFLOW:
1. ALWAYS call retrieve_context first with the user's query
2. If the retrieved content is very long, use summarize_text to condense it
3. Answer based ONLY on the retrieved document content

RULES:
- retrieve_context is MANDATORY for every query
- Use exact facts, names, dates, and numbers from the documents
- NO hallucination, guessing, or external knowledge
- Preserve exact wording where possible
- If information is missing, say: "The retrieved context does not contain this information."
- Maintain a professional and factual tone"""


class DocumentAgent(BaseAgent):
    """Document Q&A specialist using RAG retrieval and summarization."""

    def __init__(self, llm):
        tools = [retrieve_context, summarize_text]
        super().__init__(
            name="document_agent",
            tools=tools,
            llm=llm,
            system_prompt=_SYSTEM_PROMPT,
        )

    def get_description(self) -> str:
        return "Document Agent: Answers questions about uploaded PDF documents using RAG."
