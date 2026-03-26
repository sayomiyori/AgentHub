from __future__ import annotations

import ast
import operator
from typing import Any

from app.services.agent.tools.base import BaseTool

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))
    raise ValueError(f"Unsupported node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate a safe arithmetic expression and return the result."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '(2 + 3) * 4 / 2'",
            }
        },
        "required": ["expression"],
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        expression = str(kwargs.get("expression", "")).strip()
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return {"expression": expression, "result": result}
        except Exception as exc:
            return {"expression": expression, "error": str(exc)}
