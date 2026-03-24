"""
Supervisor Agent — Orchestrator that classifies queries and routes to specialists.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Sequence, TypedDict

import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agents.base import AgentState
from agents.research_agent import ResearchAgent
from agents.document_agent import DocumentAgent
from agents.data_agent import DataAgent
from tools.weather import fetch_weather
from tools.rag import retrieve_context

logger = logging.getLogger(__name__)

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    route: str  # "document" | "research" | "data" | "weather" | "general"
    active_agent: str

_WEATHER_KEYWORDS = {
    "weather", "temperature", "rain", "humidity", "wind", "forecast",
    "climate", "sunny", "cloudy", "storm", "snow", "hot", "cold",
}

_MATH_KEYWORDS = {
    "calculate", "compute", "solve", "math", "equation", "formula",
    "sum", "product", "average", "mean", "median", "factorial",
    "sqrt", "square root", "logarithm", "sin", "cos", "tan",
    "integral", "derivative", "algebra", "geometry", "trigonometry",
}

_CODE_KEYWORDS = {
    "code", "function", "class", "python", "javascript", "analyze code",
    "debug", "refactor", "algorithm", "programming", "syntax",
    "import", "def ", "async def",
}

_DOCUMENT_KEYWORDS = {
    "document", "pdf", "resume", "uploaded", "file", "paper",
    "article", "report", "content", "extract", "page",
}

_RESEARCH_KEYWORDS = {
    "search", "find", "look up", "google", "latest", "news",
    "current", "today", "recent", "what is", "who is", "how to",
    "explain", "tell me about", "information about",
}


def classify_intent(query: str) -> str:
    """
    Classify the user's intent based on keywords.
    """
    q = query.lower()

    if any(kw in q for kw in _WEATHER_KEYWORDS):
        return "weather"

    if "\n" in query and ("def " in q or "class " in q or "import " in q):
        return "data"

    if re.search(r'\d+\s*[\+\-\*/\^%]', q) or any(kw in q for kw in _MATH_KEYWORDS):
        return "data"

    if any(kw in q for kw in _CODE_KEYWORDS):
        return "data"

    if any(kw in q for kw in _DOCUMENT_KEYWORDS):
        return "document"

    if any(kw in q for kw in _RESEARCH_KEYWORDS):
        return "research"

    return "document"

_SUPERVISOR_SYSTEM = """You are a Supervisor AI Agent that orchestrates multiple specialist agents.
You handle weather queries directly and route other queries to the appropriate specialist.

For WEATHER queries, you have the fetch_weather tool available.
For other queries, specialist agents have already handled the task — just present their results.

RULES:
- For weather: Call fetch_weather and present the data
- For other queries: The specialist agent's response is in the conversation — present it clearly
- Always be professional and helpful
- If a specialist couldn't answer, acknowledge it and suggest alternatives"""


class SupervisorAgent:
    """
    Top-level orchestrator agent.
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.name = "supervisor"

        self._research_agent = None
        self._document_agent = None
        self._data_agent = None
        self.weather_tools = [fetch_weather, retrieve_context]

        self._weather_graph = self._build_weather_graph()

    @property
    def research_agent(self) -> ResearchAgent:
        if self._research_agent is None:
            self._research_agent = ResearchAgent(self.llm)
        return self._research_agent

    @property
    def document_agent(self) -> DocumentAgent:
        if self._document_agent is None:
            self._document_agent = DocumentAgent(self.llm)
        return self._document_agent

    @property
    def data_agent(self) -> DataAgent:
        if self._data_agent is None:
            self._data_agent = DataAgent(self.llm)
        return self._data_agent

    def _build_weather_graph(self):
        system_prompt = SystemMessage(content=_SUPERVISOR_SYSTEM)
        llm_with_tools = self.llm.bind_tools(self.weather_tools)

        def agent_node(state: AgentState):
            messages = state["messages"]
            messages = [m for m in messages if not isinstance(m, SystemMessage)]
            messages = [system_prompt] + list(messages)
            response = llm_with_tools.invoke(messages)

            if not response.tool_calls and "<tool_call>" in str(response.content):
                from agents.base import BaseAgent
                response.tool_calls = BaseAgent._parse_tool_calls(response.content)

            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            return "tools" if last_message.tool_calls else END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.weather_tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]

        user_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        route = classify_intent(user_query)
        logger.info("Supervisor: query='%s...' → route=%s", user_query[:50], route)

        active_agent = "supervisor"

        try:
            if route == "weather" or route == "general":
                active_agent = "supervisor (weather/general)"
                result = self._weather_graph.invoke({"messages": messages})

            elif route == "document":
                active_agent = "document_agent"
                result = self.document_agent.invoke(messages)

            elif route == "research":
                active_agent = "research_agent"
                result = self.research_agent.invoke(messages)

            elif route == "data":
                active_agent = "data_agent"
                result = self.data_agent.invoke(messages)

            else:
                active_agent = "document_agent"
                result = self.document_agent.invoke(messages)

        except Exception as e:
            logger.exception("Specialist agent '%s' failed", active_agent)
            error_msg = AIMessage(
                content=f"I encountered an error while processing your request with the {active_agent}: {str(e)}. "
                f"Please try rephrasing your question."
            )
            result = {"messages": messages + [error_msg]}

        result["_active_agent"] = active_agent
        result["_route"] = route

        return result

    def get_all_traces(self) -> Dict[str, list]:
        """Collect execution traces from all agents."""
        traces = {"supervisor": []}
        if self._research_agent:
            traces["research_agent"] = self._research_agent.get_traces()
        if self._document_agent:
            traces["document_agent"] = self._document_agent.get_traces()
        if self._data_agent:
            traces["data_agent"] = self._data_agent.get_traces()
        return traces
