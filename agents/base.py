"""
Base Agent — Abstract foundation for all specialist agents.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Sequence, TypedDict

import operator

from langchain_core.messages import BaseMessage, SystemMessage, ToolCall
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from mcp.client import MCPClient

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state definition for all agents."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class BaseAgent(ABC):
    """
    Abstract base class for specialist agents.
    """

    def __init__(self, name: str, tools: list, llm: Any, system_prompt: str) -> None:
        self.name = name
        self.tools = tools
        self.llm = llm
        self.system_prompt_text = system_prompt
        self.system_prompt = SystemMessage(content=system_prompt)
        self.mcp_client = MCPClient(agent_name=name)

        self.graph = self._build_graph()

    def _build_graph(self):
        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: AgentState):
            messages = state["messages"]
            messages = [m for m in messages if not isinstance(m, SystemMessage)]
            messages = [self.system_prompt] + list(messages)

            response = llm_with_tools.invoke(messages)

            if not response.tool_calls and "<tool_call>" in str(response.content):
                response.tool_calls = self._parse_tool_calls(response.content)

            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            return "tools" if last_message.tool_calls else END

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def invoke(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Run the agent graph synchronously."""
        return self.graph.invoke({"messages": messages})

    def stream(self, messages: List[BaseMessage]):
        """Stream the agent graph execution."""
        return self.graph.stream({"messages": messages})

    def get_traces(self) -> list:
        return self.mcp_client.get_traces()

    @staticmethod
    def _parse_tool_calls(content: str) -> List[ToolCall]:
        tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        matches = tool_call_pattern.findall(content)
        parsed = []
        for match in matches:
            try:
                json_str = match.strip().replace("{{", "{").replace("}}", "}")
                tool_data = json.loads(json_str)
                if "name" in tool_data and "arguments" in tool_data:
                    parsed.append(ToolCall(
                        name=tool_data["name"],
                        args=tool_data["arguments"],
                        id=f"call_{len(parsed)}"
                    ))
            except (json.JSONDecodeError, KeyError):
                continue
        return parsed

    @abstractmethod
    def get_description(self) -> str:
        ...
