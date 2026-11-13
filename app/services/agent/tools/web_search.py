from __future__ import annotations

from typing import Any

import requests

from app.services.agent.tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for recent information."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
        "required": ["query"],
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        max_results = int(kwargs.get("max_results", 5))
        params: dict[str, str | int] = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        try:
            res = requests.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=15,
            )
            payload = res.json()
            items = []
            if payload.get("AbstractText"):
                items.append(
                    {
                        "title": payload.get("Heading") or query,
                        "snippet": payload["AbstractText"],
                        "url": payload.get("AbstractURL", ""),
                    }
                )
            for topic in payload.get("RelatedTopics", []):
                if isinstance(topic, dict) and topic.get("Text"):
                    items.append(
                        {
                            "title": topic.get("Text", "")[:80],
                            "snippet": topic["Text"],
                            "url": topic.get("FirstURL", ""),
                        }
                    )
                    if len(items) >= max_results:
                        break
            return {"query": query, "results": items[:max_results]}
        except Exception:
            return {"query": query, "results": [], "error": "Web search unavailable"}
