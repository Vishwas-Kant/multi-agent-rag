import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

from mcp.protocol import (
    ToolCategory, ToolDefinition, ToolParameter,
    ToolRequest, ToolResponse, ToolError, ToolStatus,
)
from mcp.server import MCPServer
from mcp.client import MCPClient


class TestMCPProtocol(unittest.TestCase):
    """Tests for MCP protocol message types."""

    def test_tool_definition_creation(self):
        defn = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.GENERAL,
            parameters=[
                ToolParameter(name="query", type="string", description="Search query")
            ],
        )
        self.assertEqual(defn.name, "test_tool")
        self.assertEqual(defn.category, ToolCategory.GENERAL)
        self.assertEqual(len(defn.parameters), 1)

    def test_tool_request_auto_id(self):
        req = ToolRequest(tool_name="test_tool", arguments={"query": "hello"})
        self.assertIsNotNone(req.request_id)
        self.assertEqual(req.tool_name, "test_tool")

    def test_tool_response_serialization(self):
        resp = ToolResponse(
            request_id="123",
            tool_name="test_tool",
            result="success result",
            execution_time_ms=42.5,
        )
        self.assertEqual(resp.status, ToolStatus.SUCCESS)
        self.assertEqual(resp.result, "success result")

    def test_tool_error_serialization(self):
        err = ToolError(
            request_id="123",
            tool_name="test_tool",
            error_message="Something went wrong",
        )
        self.assertEqual(err.status, ToolStatus.ERROR)
        self.assertIn("Something went wrong", err.error_message)


class TestMCPServer(unittest.TestCase):
    """Tests for MCP server registration and dispatch."""

    def setUp(self):
        self.server = MCPServer()

    def test_register_and_list_tools(self):
        self.server.register_tool(
            name="greet",
            handler=lambda name: f"Hello, {name}!",
            description="Greets a person",
            category=ToolCategory.GENERAL,
        )
        tools = self.server.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "greet")

    def test_list_tools_by_category(self):
        self.server.register_tool(
            name="tool_a", handler=lambda: "a",
            description="A", category=ToolCategory.RETRIEVAL,
        )
        self.server.register_tool(
            name="tool_b", handler=lambda: "b",
            description="B", category=ToolCategory.API,
        )
        retrieval_tools = self.server.list_tools(category=ToolCategory.RETRIEVAL)
        self.assertEqual(len(retrieval_tools), 1)
        self.assertEqual(retrieval_tools[0].name, "tool_a")

    def test_execute_sync_tool(self):
        self.server.register_tool(
            name="add", handler=lambda a, b: a + b,
            description="Adds two numbers", category=ToolCategory.COMPUTATION,
        )
        request = ToolRequest(tool_name="add", arguments={"a": 3, "b": 5})
        response = self.server.execute_sync(request)
        self.assertIsInstance(response, ToolResponse)
        self.assertEqual(response.result, 8)

    def test_execute_unknown_tool(self):
        request = ToolRequest(tool_name="nonexistent", arguments={})
        response = self.server.execute_sync(request)
        self.assertIsInstance(response, ToolError)
        self.assertEqual(response.error_type, "ToolNotFound")

    def test_execute_tool_error(self):
        def bad_tool():
            raise ValueError("intentional error")

        self.server.register_tool(
            name="bad", handler=bad_tool,
            description="A broken tool", category=ToolCategory.GENERAL,
        )
        request = ToolRequest(tool_name="bad", arguments={})
        response = self.server.execute_sync(request)
        self.assertIsInstance(response, ToolError)
        self.assertIn("intentional error", response.error_message)

    def test_metrics_tracking(self):
        self.server.register_tool(
            name="counter", handler=lambda: 42,
            description="Returns 42", category=ToolCategory.GENERAL,
        )
        for _ in range(3):
            self.server.execute_sync(ToolRequest(tool_name="counter", arguments={}))

        metrics = self.server.get_metrics()
        self.assertEqual(metrics["counter"]["calls"], 3)
        self.assertGreater(metrics["counter"]["avg_latency_ms"], 0)


class TestMCPClient(unittest.TestCase):
    """Tests for MCP client caching and tool calls."""

    def setUp(self):
        from mcp.server import get_mcp_server
        server = get_mcp_server()
        server.register_tool(
            name="echo",
            handler=lambda msg: f"Echo: {msg}",
            description="Echoes input",
            category=ToolCategory.GENERAL,
        )
        self.client = MCPClient(agent_name="test_agent")

    def test_call_tool(self):
        result = self.client.call_tool("echo", {"msg": "hello"})
        self.assertIn("Echo: hello", result)

    def test_call_tool_caching(self):
        result1 = self.client.call_tool("echo", {"msg": "cached"})
        result2 = self.client.call_tool("echo", {"msg": "cached"})
        self.assertEqual(result1, result2)

    def test_call_tool_no_cache(self):
        result = self.client.call_tool("echo", {"msg": "no_cache"}, use_cache=False)
        self.assertIn("Echo: no_cache", result)

    def test_traces_recorded(self):
        self.client.clear_traces()
        self.client.call_tool("echo", {"msg": "traced"})
        traces = self.client.get_traces()
        self.assertGreater(len(traces), 0)
        self.assertEqual(traces[-1]["tool"], "echo")

    def test_call_nonexistent_tool(self):
        result = self.client.call_tool("no_such_tool", {})
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
