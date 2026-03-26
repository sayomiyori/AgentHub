from dataclasses import dataclass


@dataclass
class UsageMetrics:
    tokens_used: int
    cost_usd: float
