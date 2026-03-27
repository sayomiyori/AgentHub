from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from mcp import types
from mcp.client.session_group import ClientSessionGroup, SseServerParameters

from app.config import get_settings
from app.mcp.tools_proxy import MCPToolProxy
from app.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower() or "srv"


def _component_name_hook(tool_name: str, server_info: types.Implementation) -> str:
    return f"{_slug(server_info.name)}__{tool_name}"


def load_mcp_servers_config() -> list[dict[str, str]]:
    settings = get_settings()
    servers: list[dict[str, str]] = []
    raw = getattr(settings, "mcp_servers", None) or ""
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("url"):
                        servers.append(
                            {
                                "name": str(item.get("name", "remote")),
                                "url": str(item["url"]).strip(),
                            }
                        )
        except json.JSONDecodeError as exc:
            logger.warning("Invalid MCP_SERVERS JSON: %s", exc)

    yaml_path = getattr(settings, "mcp_servers_config", None) or ""
    if yaml_path:
        p = Path(yaml_path)
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "MCP_SERVERS" in data:
                data = data["MCP_SERVERS"]
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("url"):
                        servers.append(
                            {
                                "name": str(item.get("name", "remote")),
                                "url": str(item["url"]).strip(),
                            }
                        )
    return servers


def _tool_input_schema(tool: types.Tool) -> dict[str, Any]:
    schema = tool.inputSchema
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


class MCPClientManager:
    """Connects to external MCP servers over SSE and exposes tools to the agent."""

    def __init__(self) -> None:
        self._group: ClientSessionGroup | None = None
        self._started = False

    @property
    def group(self) -> ClientSessionGroup | None:
        return self._group

    async def startup(self) -> None:
        if self._started:
            return
        servers = load_mcp_servers_config()
        if not servers:
            logger.info("No MCP_SERVERS configured; skipping MCP client connections.")
            self._started = True
            return

        self._group = ClientSessionGroup(component_name_hook=_component_name_hook)
        await self._group.__aenter__()
        for srv in servers:
            try:
                await self._group.connect_to_server(SseServerParameters(url=srv["url"]))
                logger.info("Connected MCP server %s (%s)", srv.get("name"), srv["url"])
            except Exception as exc:
                logger.warning("Could not connect MCP server %s: %s", srv["url"], exc)
        self._started = True

    async def shutdown(self) -> None:
        if self._group and self._started:
            await self._group.__aexit__(None, None, None)
        self._group = None
        self._started = False

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        if not self._group:
            raise RuntimeError("MCP client is not initialized")
        return await self._group.call_tool(name, arguments or {})

    def iter_agent_tools(self) -> list[BaseTool]:
        if not self._group:
            return []
        proxies: list[BaseTool] = []
        async def _invoke(tool_name: str, args: dict[str, Any]) -> types.CallToolResult:
            return await self.call_tool(tool_name, args)

        for prefixed_name, spec in self._group.tools.items():
            params = _tool_input_schema(spec)
            if "type" not in params:
                params = {"type": "object", **params}

            proxies.append(
                MCPToolProxy(
                    name=prefixed_name,
                    description=spec.description or f"MCP tool {spec.name}",
                    parameters=params,
                    call_fn=_invoke,
                )
            )
        return proxies


mcp_client_manager = MCPClientManager()
