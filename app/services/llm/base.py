from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLMProvider(ABC):
    """Unified LLM interface for chat completions with optional tool definitions."""

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """messages: OpenAI-style roles system|user|assistant with content string."""
