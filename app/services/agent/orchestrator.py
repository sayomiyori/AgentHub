from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.conversation import Message, MessageRole
from app.models.usage import UsageMetrics
from app.services.agent.prompts.system import AGENT_SYSTEM_PROMPT
from app.services.agent.tools.base import BaseTool
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.datetime_tool import DatetimeTool
from app.services.agent.tools.knowledge_base import KnowledgeBaseTool
from app.services.agent.tools.web_search import WebSearchTool
from app.services.rag_pipeline import RAGPipeline

MAX_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 20
COST_PER_TOKEN = 0.000001


def _extract_json(text: str) -> dict[str, Any]:
    """Extract first JSON object from model text output."""
    text = text.strip()
    # try to grab content between first { and matching }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"type": "final", "answer": text}


class AgentOrchestrator:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.llm_model
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.rag_pipeline = RAGPipeline()

    def _build_tools(self, db: Session) -> dict[str, BaseTool]:
        tools: list[BaseTool] = [
            KnowledgeBaseTool(db=db, rag_pipeline=self.rag_pipeline),
            WebSearchTool(),
            CalculatorTool(),
            DatetimeTool(),
        ]
        return {t.name: t for t in tools}

    def _load_history(self, db: Session, conversation_id: int) -> list[dict[str, str]]:
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(MAX_HISTORY_MESSAGES)
            .all()
        )
        return [{"role": m.role.value, "content": m.content} for m in reversed(messages)]

    def _build_prompt(self, history: list[dict[str, str]], question: str, tool_results: list[str]) -> str:
        parts: list[str] = [AGENT_SYSTEM_PROMPT, ""]
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role_label}: {msg['content']}")
        if tool_results:
            parts.append("\nTool results so far:")
            for tr in tool_results:
                parts.append(tr)
            parts.append("")
        parts.append(f"User: {question}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def run(self, db: Session, question: str, conversation_id: int, top_k: int = 5) -> dict[str, Any]:
        tools_map = self._build_tools(db)
        history = self._load_history(db, conversation_id)
        tool_results: list[str] = []
        tools_called: list[dict[str, Any]] = []
        total_tokens = 0

        for iteration in range(MAX_ITERATIONS):
            prompt = self._build_prompt(history, question, tool_results)
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                raw_text = getattr(response, "text", "") or ""
                usage = getattr(response, "usage_metadata", None)
                total_tokens += int(getattr(usage, "total_token_count", 0) or 0)
            except Exception as exc:
                return {
                    "answer": f"Agent error: {exc}",
                    "tools_called": tools_called,
                    "tokens_used": total_tokens,
                    "cost_usd": round(total_tokens * COST_PER_TOKEN, 6),
                    "sources": [],
                }

            parsed = _extract_json(raw_text)
            resp_type = parsed.get("type", "final")

            if resp_type == "tool_call":
                tool_name = parsed.get("tool_name", "")
                arguments = parsed.get("arguments", {})
                tool = tools_map.get(tool_name)
                if tool is None:
                    tool_results.append(f"[tool={tool_name}] ERROR: unknown tool")
                    continue
                try:
                    result = tool.run(**arguments)
                except Exception as exc:
                    result = {"error": str(exc)}
                tools_called.append({"tool": tool_name, "arguments": arguments, "result": result})
                tool_results.append(f"[tool={tool_name}] {json.dumps(result, ensure_ascii=False)}")

                # Extract KB sources for response
                if tool_name == "search_knowledge_base":
                    continue
            else:
                answer = parsed.get("answer", raw_text)
                kb_sources = []
                for tc in tools_called:
                    if tc["tool"] == "search_knowledge_base":
                        for item in tc["result"].get("results", []):
                            kb_sources.append({
                                "chunk_id": item.get("chunk_id"),
                                "document_title": item.get("document_title"),
                                "chunk_text_preview": item.get("chunk_text", "")[:220],
                                "score": float(item.get("score", 0.0)),
                            })
                return {
                    "answer": answer,
                    "tools_called": tools_called,
                    "tokens_used": total_tokens,
                    "cost_usd": round(total_tokens * COST_PER_TOKEN, 6),
                    "sources": kb_sources,
                }

            if iteration == MAX_ITERATIONS - 1:
                break

        return {
            "answer": "Agent reached maximum iterations without a final answer.",
            "tools_called": tools_called,
            "tokens_used": total_tokens,
            "cost_usd": round(total_tokens * COST_PER_TOKEN, 6),
            "sources": [],
        }
