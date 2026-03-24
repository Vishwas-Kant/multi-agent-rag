"""
MCP Client — Agent-facing interface for tool discovery and execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from mcp.protocol import ToolCategory, ToolDefinition, ToolRequest, ToolResponse, ToolError, ToolStatus
from mcp.server import get_mcp_server

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Lightweight client that agents use to talk to the MCP server.
    """

    def __init__(self, agent_name: str = "default", cache_size: int = 128) -> None:
        self.agent_name = agent_name
        self._server = get_mcp_server()
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = cache_size
        self._traces: List[Dict[str, Any]] = []

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """Discover available tools, optionally filtered by category."""
        return self._server.list_tools(category)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._server.get_tool(name)

    def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> str:

        arguments = arguments or {}
        cache_key = self._cache_key(tool_name, arguments)

        if use_cache and cache_key in self._cache:
            logger.debug("MCP client cache hit: %s", tool_name)
            return self._cache[cache_key]

        request = ToolRequest(
            tool_name=tool_name,
            arguments=arguments,
            caller_agent=self.agent_name,
        )

        response = self._server.execute_sync(request)
        result_str = self._response_to_str(response)

        self._traces.append({
            "tool": tool_name,
            "args": arguments,
            "status": response.status.value if hasattr(response, "status") else "error",
            "latency_ms": getattr(response, "execution_time_ms", 0),
            "result_preview": result_str[:200],
        })

        if use_cache and isinstance(response, ToolResponse):
            self._cache[cache_key] = result_str
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return result_str

    def call_tools_parallel(
        self,
        calls: List[Dict[str, Any]],
    ) -> List[str]:
        
        requests = [
            ToolRequest(
                tool_name=c["tool_name"],
                arguments=c.get("arguments", {}),
                caller_agent=self.agent_name,
            )
            for c in calls
        ]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self._server.execute_parallel(requests),
                )
                responses = future.result()
        else:
            responses = asyncio.run(self._server.execute_parallel(requests))

        results = []
        for resp in responses:
            result_str = self._response_to_str(resp)
            self._traces.append({
                "tool": resp.tool_name,
                "status": resp.status.value if hasattr(resp, "status") else "error",
                "latency_ms": getattr(resp, "execution_time_ms", 0),
                "result_preview": result_str[:200],
            })
            results.append(result_str)

        return results

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return execution traces for the current session."""
        return list(self._traces)

    def clear_traces(self) -> None:
        self._traces.clear()

    @staticmethod
    def _cache_key(tool_name: str, arguments: Dict[str, Any]) -> str:
        raw = json.dumps({"t": tool_name, "a": arguments}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _response_to_str(response: ToolResponse | ToolError) -> str:
        if isinstance(response, ToolError):
            return f"Error ({response.error_type}): {response.error_message}"
        result = response.result
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)
