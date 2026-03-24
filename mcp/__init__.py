"""
MCP (Model Context Protocol) - Tool registry, dispatch, and agent-tool communication layer.
"""

from mcp.protocol import ToolDefinition, ToolRequest, ToolResponse, ToolError, ToolCategory
from mcp.server import MCPServer, get_mcp_server
from mcp.client import MCPClient

__all__ = [
    "ToolDefinition", "ToolRequest", "ToolResponse", "ToolError", "ToolCategory",
    "MCPServer", "get_mcp_server",
    "MCPClient",
]
