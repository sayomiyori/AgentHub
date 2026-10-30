from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.workers.embed_worker import _parse_document_bytes


@pytest.mark.parametrize("kind", ["txt", "md"])
def test_text_snapshot_preserves_previous_newline_decoding(kind):
    assert _parse_document_bytes(b"first\r\nsecond\rthird\nfourth", kind) == "first\nsecond\nthird\nfourth"


def test_pdf_snapshot_uses_immutable_source_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = BytesIO()
    writer.write(stream)
    assert _parse_document_bytes(stream.getvalue(), "pdf") == ""


def test_invalid_snapshot_is_not_silently_decoded():
    with pytest.raises(UnicodeDecodeError):
        _parse_document_bytes(b"\xff", "txt")
    with pytest.raises(ValueError):
        _parse_document_bytes(b"text", "other")
