"""Tool framework for the Research Agent."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Standardised output from any tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str = ""
    elapsed_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"
        if isinstance(self.data, dict):
            keys = ", ".join(self.data.keys())
            return f"[{self.tool_name}] OK — keys: {keys}"
        if isinstance(self.data, list):
            return f"[{self.tool_name}] OK — {len(self.data)} items"
        return f"[{self.tool_name}] OK"


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str = "base_tool"
    description: str = "No description"
    parameters: Dict[str, str] = {}  # name → human-readable description

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool and return a ToolResult."""

    def _timed_run(self, **kwargs) -> ToolResult:
        start = time.time()
        result = self.run(**kwargs)
        result.elapsed_sec = round(time.time() - start, 3)
        return result

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class ToolRegistry:
    """Central catalogue of available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())

    def run(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool_name=name, success=False, error=f"Unknown tool: {name}")
        return tool._timed_run(**kwargs)
