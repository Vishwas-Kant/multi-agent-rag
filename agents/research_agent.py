"""
Research Agent — Specialist for web search and information synthesis.
"""

from __future__ import annotations

from agents.base import BaseAgent
from tools.web_search import web_search
from tools.web_reader import read_webpage
from tools.summarizer import summarize_text


_SYSTEM_PROMPT = """You are a Research Specialist AI agent.
Your job is to find accurate information from the web and provide well-researched answers.

AVAILABLE TOOLS:
1. web_search(query) — Search the web for information. Use this FIRST for any research query.
2. read_webpage(url) — Read the full content of a specific webpage. Use after finding relevant URLs.
3. summarize_text(text, style) — Summarize long text. Use to condense findings.

WORKFLOW:
1. Use web_search to find relevant results
2. If needed, use read_webpage to get full article content
3. Use summarize_text if the content is too long
4. Synthesize findings into a clear, factual answer

RULES:
- Always cite sources (URLs) in your answers
- Do NOT make up information — only use data from tools
- If no results are found, state it clearly
- Be thorough but concise"""


class ResearchAgent(BaseAgent):
    """Web research specialist using search, reading, and summarization tools."""

    def __init__(self, llm):
        tools = [web_search, read_webpage, summarize_text]
        super().__init__(
            name="research_agent",
            tools=tools,
            llm=llm,
            system_prompt=_SYSTEM_PROMPT,
        )

    def get_description(self) -> str:
        return "Research Agent: Searches the web, reads articles, and summarizes findings."
