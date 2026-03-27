from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.mcp.client import mcp_client_manager
from app.models.conversation import Message
from app.services.agent.prompts.system import AGENT_SYSTEM_PROMPT
from app.services.agent.tools.base import BaseTool
from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.datetime_tool import DatetimeTool
from app.services.agent.tools.knowledge_base import KnowledgeBaseTool
from app.services.agent.tools.web_search import WebSearchTool
from app.services.llm.factory import LLMFactory
from app.services.rag_pipeline import RAGPipeline
from app.services.usage_tracker import record_llm_call

MAX_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 20


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
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
        self.llm_factory = LLMFactory()
        self.rag_pipeline = RAGPipeline()

    def _build_tools(self, db: Session) -> dict[str, BaseTool]:
        tools: list[BaseTool] = [
            KnowledgeBaseTool(db=db, rag_pipeline=self.rag_pipeline),
            WebSearchTool(),
            CalculatorTool(),
            DatetimeTool(),
        ]
        tools.extend(mcp_client_manager.iter_agent_tools())
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

    def _collect_kb_sources(self, tools_called: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kb_sources: list[dict[str, Any]] = []
        for tc in tools_called:
            name = tc["tool"]
            result = tc.get("result") or {}
            if name == "search_knowledge_base":
                for item in result.get("results", []):
                    kb_sources.append({
                        "chunk_id": item.get("chunk_id"),
                        "document_title": item.get("document_title"),
                        "chunk_text_preview": str(item.get("chunk_text", ""))[:220],
                        "score": float(item.get("score", 0.0)),
                    })
            elif "search_documents" in name:
                payload = result.get("result")
                if payload is None and isinstance(result, list):
                    payload = result
                if isinstance(payload, list):
                    items = payload
                elif isinstance(payload, dict):
                    items = payload.get("results", payload.get("items", []))
                else:
                    items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    kb_sources.append({
                        "chunk_id": None,
                        "document_title": str(item.get("document_title", "")),
                        "chunk_text_preview": str(item.get("chunk_text", ""))[:220],
                        "score": float(item.get("relevance_score", item.get("score", 0.0))),
                    })
        return kb_sources

    def run(
        self,
        db: Session,
        question: str,
        conversation_id: int,
        top_k: int = 5,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        tools_map = self._build_tools(db)
        history = self._load_history(db, conversation_id)
        tool_results: list[str] = []
        tools_called: list[dict[str, Any]] = []
        total_tokens = 0
        total_cost = 0.0

        tool_hint = ", ".join(sorted(tools_map.keys()))

        for iteration in range(MAX_ITERATIONS):
            prompt = self._build_prompt(history, question, tool_results)
            prompt = f"{prompt}\n\nAvailable tool names: {tool_hint}"

            try:
                resp = self.llm_factory.generate(
                    messages=[{"role": "user", "content": prompt}],
                    provider=provider,
                    model=model,
                    temperature=0.2,
                )
                raw_text = resp.content or ""
                total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
                total_cost += resp.usage.cost_usd
                if request_id:
                    record_llm_call(
                        db,
                        conversation_id=conversation_id,
                        message_id=None,
                        request_id=request_id,
                        provider=resp.provider or "gemini",
                        model=resp.model or "",
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        cost_usd=float(resp.usage.cost_usd),
                    )
            except Exception as exc:
                return {
                    "answer": f"Agent error: {exc}",
                    "tools_called": tools_called,
                    "tokens_used": total_tokens,
                    "cost_usd": round(total_cost, 8),
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
            else:
                answer = parsed.get("answer", raw_text)
                kb_sources = self._collect_kb_sources(tools_called)
                return {
                    "answer": answer,
                    "tools_called": tools_called,
                    "tokens_used": total_tokens,
                    "cost_usd": round(total_cost, 8),
                    "sources": kb_sources,
                }

            if iteration == MAX_ITERATIONS - 1:
                break

        return {
            "answer": "Agent reached maximum iterations without a final answer.",
            "tools_called": tools_called,
            "tokens_used": total_tokens,
            "cost_usd": round(total_cost, 8),
            "sources": [],
        }
