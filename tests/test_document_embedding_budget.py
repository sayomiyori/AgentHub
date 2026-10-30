import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_document_chunk_limit_is_explicit():
    assert Settings(_env_file=None).max_document_chunks == 1000


@pytest.mark.parametrize("value", [0, -1, 10001, "invalid"])
def test_invalid_document_chunk_limit_fails_config(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, MAX_DOCUMENT_CHUNKS=value)


@pytest.mark.parametrize("value", [1, 10000])
def test_document_chunk_limit_accepts_supported_boundaries(value):
    assert Settings(_env_file=None, MAX_DOCUMENT_CHUNKS=value).max_document_chunks == value
