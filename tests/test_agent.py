from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.agent.tools.calculator import CalculatorTool
from app.services.agent.tools.datetime_tool import DatetimeTool
from app.services.llm.base import LLMResponse, LLMUsage


def _llm_resp(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage=LLMUsage(input_tokens=100, output_tokens=100, cost_usd=0.0002),
    )


# ---------------------------------------------------------------------------
# Unit tests: individual tools
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def test_basic_arithmetic(self) -> None:
        tool = CalculatorTool()
        result = tool.run(expression="(2 + 3) * 4")
        assert result["result"] == 20.0

    def test_division(self) -> None:
        tool = CalculatorTool()
        result = tool.run(expression="10 / 4")
        assert result["result"] == 2.5

    def test_power(self) -> None:
        tool = CalculatorTool()
        result = tool.run(expression="2 ** 10")
        assert result["result"] == 1024.0

    def test_invalid_expression(self) -> None:
        tool = CalculatorTool()
        result = tool.run(expression="import os")
        assert "error" in result

    def test_unsafe_expression(self) -> None:
        tool = CalculatorTool()
        result = tool.run(expression="__import__('os')")
        assert "error" in result


class TestDatetimeTool:
    def test_returns_utc_by_default(self) -> None:
        tool = DatetimeTool()
        result = tool.run()
        assert result["timezone"] == "UTC"
        assert "datetime" in result
        assert "date" in result
        assert "weekday" in result

    def test_valid_timezone(self) -> None:
        tool = DatetimeTool()
        result = tool.run(timezone="Europe/Moscow")
        assert result["timezone"] == "Europe/Moscow"

    def test_invalid_timezone_fallback(self) -> None:
        tool = DatetimeTool()
        result = tool.run(timezone="Invalid/Zone")
        assert result["timezone"] == "UTC"


# ---------------------------------------------------------------------------
# Integration tests: orchestrator with mocked LLM
# ---------------------------------------------------------------------------

class TestAgentUsesKnowledgeBase:
    """Verify that orchestrator calls search_knowledge_base tool when needed."""

    def test_agent_uses_knowledge_base(self, mock_db: MagicMock) -> None:
        tool_call_resp = _llm_resp(
            json.dumps({
                "type": "tool_call",
                "tool_name": "search_knowledge_base",
                "arguments": {"query": "FastAPI", "top_k": 3},
            })
        )
        final_resp = _llm_resp(
            json.dumps({
                "type": "final",
                "answer": "FastAPI is built on Starlette and Pydantic [chunk_id=2].",
            })
        )

        kb_tool_result = {
            "query": "FastAPI",
            "results": [
                {
                    "chunk_id": 2,
                    "document_title": "test.txt",
                    "chunk_text": "FastAPI is built on Starlette for web handling and Pydantic for data validation.",
                    "score": 0.95,
                }
            ],
        }

        import importlib

        orch_mod = importlib.import_module("app.services.agent.orchestrator")
        with patch.object(orch_mod, "LLMFactory") as MockLF:
            mock_factory = MagicMock()
            MockLF.return_value = mock_factory
            mock_factory.generate.side_effect = [tool_call_resp, final_resp]

            with patch("app.services.agent.tools.knowledge_base.KnowledgeBaseTool.run", return_value=kb_tool_result):
                from app.services.agent.orchestrator import AgentOrchestrator

                orchestrator = AgentOrchestrator()
                result = orchestrator.run(
                    db=mock_db,
                    question="What is FastAPI?",
                    conversation_id=1,
                )

        assert "FastAPI" in result["answer"]
        tool_names = [tc["tool"] for tc in result["tools_called"]]
        assert "search_knowledge_base" in tool_names
        assert result["tokens_used"] == 400


class TestAgentMultiTool:
    """Verify chaining of multiple tools in one session."""

    def test_multi_tool_chain(self, mock_db: MagicMock) -> None:
        calc_call = _llm_resp(
            json.dumps({
                "type": "tool_call",
                "tool_name": "calculator",
                "arguments": {"expression": "3 * 7"},
            })
        )
        dt_call = _llm_resp(
            json.dumps({
                "type": "tool_call",
                "tool_name": "get_datetime",
                "arguments": {"timezone": "UTC"},
            })
        )
        final = _llm_resp(
            json.dumps({
                "type": "final",
                "answer": "3 * 7 = 21 and today is some date.",
            })
        )

        import importlib

        orch_mod = importlib.import_module("app.services.agent.orchestrator")
        with patch.object(orch_mod, "LLMFactory") as MockLF:
            mock_factory = MagicMock()
            MockLF.return_value = mock_factory
            mock_factory.generate.side_effect = [calc_call, dt_call, final]

            from app.services.agent.orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator()
            result = orchestrator.run(
                db=mock_db,
                question="What is 3*7 and what day is today?",
                conversation_id=2,
            )

        tool_names = [tc["tool"] for tc in result["tools_called"]]
        assert "calculator" in tool_names
        assert "get_datetime" in tool_names
        assert result["tokens_used"] == 600


class TestConversationHistory:
    """Verify that existing messages from conversation are loaded into prompt."""

    def test_history_included_in_prompt(self) -> None:
        history_msg1 = MagicMock()
        history_msg1.role = MagicMock()
        history_msg1.role.value = "user"
        history_msg1.content = "Tell me about Python."
        history_msg1.id = 1

        history_msg2 = MagicMock()
        history_msg2.role = MagicMock()
        history_msg2.role.value = "assistant"
        history_msg2.content = "Python is a programming language."
        history_msg2.id = 2

        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            history_msg2,
            history_msg1,
        ]

        final_resp = _llm_resp(
            json.dumps({
                "type": "final",
                "answer": "Python was created by Guido van Rossum.",
            })
        )

        captured_prompts: list[str] = []

        def capture_generate(
            messages: list,
            *,
            provider: str | None = None,
            model: str | None = None,
            temperature: float = 0.2,
            tools: list | None = None,
        ) -> LLMResponse:
            captured_prompts.append(messages[0]["content"])
            return final_resp

        import importlib

        orch_mod = importlib.import_module("app.services.agent.orchestrator")
        with patch.object(orch_mod, "LLMFactory") as MockLF:
            mock_factory = MagicMock()
            MockLF.return_value = mock_factory
            mock_factory.generate.side_effect = capture_generate

            from app.services.agent.orchestrator import AgentOrchestrator

            orchestrator = AgentOrchestrator()
            result = orchestrator.run(
                db=db,
                question="Who created it?",
                conversation_id=3,
            )

        assert result["answer"] == "Python was created by Guido van Rossum."
        assert len(captured_prompts) == 1
        assert "Tell me about Python" in captured_prompts[0] or "Python is a programming language" in captured_prompts[0]
