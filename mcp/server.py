"""
MCP Server — Central tool registry, discovery, and async dispatch engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from mcp.protocol import (
    ToolCategory,
    ToolDefinition,
    ToolError,
    ToolParameter,
    ToolRequest,
    ToolResponse,
    ToolStatus,
)

logger = logging.getLogger(__name__)

_server_instance: Optional["MCPServer"] = None


def get_mcp_server() -> "MCPServer":
    global _server_instance
    if _server_instance is None:
        _server_instance = MCPServer()
    return _server_instance


class MCPServer:
    """
    Central registry and async dispatcher for MCP tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._metrics: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str,
        category: ToolCategory,
        parameters: Optional[List[ToolParameter]] = None,
        is_async: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        definition = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters or [],
            is_async=is_async,
            timeout_seconds=timeout_seconds,
        )
        self._tools[name] = definition
        self._handlers[name] = handler
        self._metrics[name] = {"calls": 0, "total_ms": 0.0, "errors": 0}
        logger.info("MCP: registered tool '%s' [%s]", name, category.value)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        tools = list(self._tools.values())
        if category is not None:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    async def execute(self, request: ToolRequest) -> ToolResponse | ToolError:
        """
        Execute a tool request asynchronously.
        """
        if request.tool_name not in self._handlers:
            return ToolError(
                request_id=request.request_id,
                tool_name=request.tool_name,
                error_type="ToolNotFound",
                error_message=f"Tool '{request.tool_name}' is not registered.",
            )

        definition = self._tools[request.tool_name]
        handler = self._handlers[request.tool_name]
        start = time.perf_counter()

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**request.arguments),
                    timeout=definition.timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(handler, **request.arguments),
                    timeout=definition.timeout_seconds,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics[request.tool_name]["calls"] += 1
            self._metrics[request.tool_name]["total_ms"] += elapsed_ms

            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                result=result,
                execution_time_ms=round(elapsed_ms, 2),
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics[request.tool_name]["errors"] += 1
            return ToolError(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.TIMEOUT,
                error_type="ToolTimeout",
                error_message=f"Tool '{request.tool_name}' timed out after {definition.timeout_seconds}s.",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics[request.tool_name]["errors"] += 1
            logger.exception("MCP: tool '%s' raised an exception", request.tool_name)
            return ToolError(
                request_id=request.request_id,
                tool_name=request.tool_name,
                error_message=str(exc),
            )

    def execute_sync(self, request: ToolRequest) -> ToolResponse | ToolError:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.execute(request))
                return future.result()
        else:
            return asyncio.run(self.execute(request))

    async def execute_parallel(
        self, requests: List[ToolRequest]
    ) -> List[ToolResponse | ToolError]:
        return await asyncio.gather(*(self.execute(r) for r in requests))

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, m in self._metrics.items():
            avg = m["total_ms"] / m["calls"] if m["calls"] else 0
            result[name] = {
                "calls": m["calls"],
                "avg_latency_ms": round(avg, 2),
                "errors": m["errors"],
            }
        return result
