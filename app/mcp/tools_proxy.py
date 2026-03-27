from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from app.services.agent.tools.base import BaseTool


def _split_mcp_tool_name(name: str) -> tuple[str, str]:
    if "__" in name:
        a, b = name.split("__", 1)
        return a or "mcp", b or "unknown"
    return "mcp", name


class MCPToolProxy(BaseTool):
    """Routes tool execution to an async MCP call_tool handler."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        call_fn: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._call_fn = call_fn

    def run(self, **kwargs: Any) -> dict[str, Any]:
        srv, tname = _split_mcp_tool_name(self.name)
        from app.metrics import mcp_tool_calls_total

        mcp_tool_calls_total.labels(server=srv, tool=tname).inc()

        async def _run() -> Any:
            return await self._call_fn(self.name, kwargs)

        try:
            result = anyio.run(_run)
        except Exception as exc:
            return {"error": str(exc)}
        return _call_tool_result_to_dict(result)


def _call_tool_result_to_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    is_error = getattr(result, "isError", False)
    content_blocks = getattr(result, "content", None) or []
    texts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", "") == "text":
            texts.append(getattr(block, "text", "") or "")
    text = "\n".join(texts)
    if is_error:
        return {"error": text or "tool error"}
    if not text.strip():
        return {"result": None}
    try:
        return json.loads(text) if text.strip().startswith(("{", "[")) else {"result": text}
    except json.JSONDecodeError:
        return {"result": text}
