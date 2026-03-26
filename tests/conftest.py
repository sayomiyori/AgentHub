from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return db


@pytest.fixture()
def mock_gemini_response():
    """Factory that creates a mock Gemini response with given text."""
    def _make(text: str) -> MagicMock:
        resp = MagicMock()
        resp.text = text
        resp.usage_metadata = MagicMock()
        resp.usage_metadata.total_token_count = 100
        return resp
    return _make
