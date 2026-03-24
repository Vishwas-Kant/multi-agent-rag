"""
Tools package — auto-registers every tool with the MCP server on import.
"""

from tools.rag import retrieve_context
from tools.weather import fetch_weather
from tools.web_search import web_search
from tools.web_reader import read_webpage
from tools.summarizer import summarize_text
from tools.calculator import calculate
from tools.code_analysis import analyze_code

_ALL_TOOLS = [
    retrieve_context,
    fetch_weather,
    web_search,
    read_webpage,
    summarize_text,
    calculate,
    analyze_code,
]


def get_all_tools():
    return list(_ALL_TOOLS)


def get_tools_by_names(names: list[str]):
    name_set = set(names)
    return [t for t in _ALL_TOOLS if t.name in name_set]
