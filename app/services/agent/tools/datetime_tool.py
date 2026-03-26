from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.services.agent.tools.base import BaseTool


class DatetimeTool(BaseTool):
    name = "get_datetime"
    description = "Return the current date and time, optionally in a given timezone."
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Europe/Moscow'. Defaults to UTC.",
                "default": "UTC",
            }
        },
        "required": [],
    }

    def run(self, **kwargs: Any) -> dict[str, Any]:
        tz_name: str = str(kwargs.get("timezone", "UTC")).strip() or "UTC"
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            tz = timezone.utc
            tz_name = "UTC"
        now = datetime.now(tz)
        return {
            "timezone": tz_name,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": now.strftime("%A"),
        }
