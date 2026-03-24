"""
MCP Protocol — Type-safe message definitions for agent <--> tool communication.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Classification used by the supervisor to route queries to specialist agents."""
    RETRIEVAL = "retrieval"
    API = "api"
    COMPUTATION = "computation"
    ANALYSIS = "analysis"
    GENERAL = "general"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Advertises a tool's capabilities so agents can discover it at runtime."""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = Field(default_factory=list)
    is_async: bool = False
    timeout_seconds: float = 30.0
    version: str = "1.0.0"


class ToolRequest(BaseModel):
    """An agent's request to execute a specific tool."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    caller_agent: str = "unknown"


class ToolResponse(BaseModel):
    """Successful result from a tool execution."""
    request_id: str
    tool_name: str
    status: ToolStatus = ToolStatus.SUCCESS
    result: Any = None
    execution_time_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class ToolError(BaseModel):
    """Structured error returned when a tool call fails."""
    request_id: str
    tool_name: str
    status: ToolStatus = ToolStatus.ERROR
    error_type: str = "ToolExecutionError"
    error_message: str = ""
    timestamp: float = Field(default_factory=time.time)
