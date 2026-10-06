from __future__ import annotations

# USD per 1M tokens (input, output)
MODEL_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    "openai": {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": (3.0, 15.0),
        "claude-3-5-haiku-20241022": (1.0, 5.0),
        "claude-sonnet-4-20250514": (3.0, 15.0),
    },
    "gemini": {
        "models/gemini-2.5-flash": (0.075, 0.30),
        "models/gemini-2.0-flash": (0.10, 0.40),
        "models/gemini-1.5-flash": (0.075, 0.30),
    },
    "groq": {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "llama-3.2-1b-preview": (0.04, 0.04),
        "llama-3.2-3b-preview": (0.06, 0.06),
        "llama-3.2-11b-text-preview": (0.18, 0.18),
        "llama-3.2-90b-text-preview": (0.90, 0.90),
        # Current catalog estimates, not actual free-account billing.
        "openai/gpt-oss-20b": (0.075, 0.30),
        "qwen/qwen3.8-27b": (0.80, 4.00),
    },
}


def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    prov = MODEL_PRICING.get(provider.lower(), {})
    # exact match
    if model in prov:
        inp_p, out_p = prov[model]
    else:
        # prefix / partial match
        inp_p, out_p = (1.0, 3.0)
        for key, prices in prov.items():
            if key in model or model in key:
                inp_p, out_p = prices
                break
    return round((input_tokens * inp_p + output_tokens * out_p) / 1_000_000, 8)
