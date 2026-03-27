from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return db


