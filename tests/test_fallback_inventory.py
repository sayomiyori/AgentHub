import struct

from app.services.rag.embedder import GeminiEmbedder
from scripts.audit_embedding_fallback import classify_vector


def test_exact_fallback_is_recognized_and_other_vectors_are_unknown():
    embedder = GeminiEmbedder.__new__(GeminiEmbedder)
    values = embedder._fallback_embedding("synthetic text")
    stored = [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]
    assert classify_vector("synthetic text", stored) is True
    stored[0] = 0.0
    assert classify_vector("synthetic text", stored) is False
    assert classify_vector("different text", values) is False
    assert classify_vector("synthetic text", values[:-1]) is False


def test_connection_failure_does_not_print_private_details(monkeypatch, capsys):
    from scripts import audit_embedding_fallback

    def fail():
        raise RuntimeError("private sentinel")

    monkeypatch.setattr(audit_embedding_fallback, "get_settings", fail)
    assert audit_embedding_fallback.main() == 1
    output = capsys.readouterr()
    assert "private sentinel" not in output.out + output.err
    assert output.out == "Embedding inventory failed; no result is available.\n"


def test_fallback_inventory_on_isolated_postgres():
    import os
    import subprocess
    import sys

    import pytest
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from app.models.chunk import Chunk
    from app.models.document import ContentType, Document
    from scripts.audit_embedding_fallback import inventory

    url = os.environ.get("AGENTHUB_PLATFORM_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Isolated PostgreSQL required")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            name = connection.scalar(text("SELECT current_database()"))
            assert name.startswith("nexus_agent_ai_") and name.endswith("_test")
        assert inventory(engine) == {"total": 0, "fallback_matches": 0, "unknown": 0,
                                     "documents_with_fallback": 0, "documents_with_unknown": 0}
        with Session(engine) as db:
            document = Document(title="synthetic", content_type=ContentType.txt, file_path="unused")
            second = Document(title="synthetic second", content_type=ContentType.txt, file_path="unused")
            third = Document(title="synthetic third", content_type=ContentType.txt, file_path="unused")
            db.add_all([document, second, third])
            db.flush()
            vector = GeminiEmbedder.__new__(GeminiEmbedder)._fallback_embedding("synthetic text")
            db.add_all([
                Chunk(document_id=document.id, text="synthetic text", chunk_index=0, embedding=vector, meta={}),
                Chunk(document_id=document.id, text="other", chunk_index=1, embedding=[0.5] * 1536, meta={}),
                Chunk(document_id=document.id, text="synthetic text", chunk_index=2, embedding=vector, meta={}),
                Chunk(document_id=second.id, text="other", chunk_index=0, embedding=[0.5] * 1536, meta={}),
                Chunk(document_id=third.id, text="synthetic text", chunk_index=0, embedding=vector, meta={}),
            ])
            db.commit()
        before = inventory(engine)
        after = inventory(engine)
        assert before == after == {"total": 5, "fallback_matches": 3, "unknown": 2,
                                   "documents_with_fallback": 2, "documents_with_unknown": 2}
        environment = {**os.environ, "DATABASE_URL": url}
        result = subprocess.run([sys.executable, "-m", "scripts.audit_embedding_fallback"],
                                env=environment, capture_output=True, text=True, check=False)
        assert result.returncode == 0
        import json
        assert json.loads(result.stdout) == before
        assert result.stderr == ""
    finally:
        engine.dispose()


def test_dispose_failure_is_reported_without_details(monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts import audit_embedding_fallback

    engine = Mock()
    engine.dispose.side_effect = RuntimeError("private sentinel")
    monkeypatch.setattr(audit_embedding_fallback, "get_settings", lambda: SimpleNamespace(database_url="unused"))
    monkeypatch.setattr(audit_embedding_fallback, "create_engine", lambda _: engine)
    monkeypatch.setattr(audit_embedding_fallback, "inventory", lambda _: {"total": 0})
    assert audit_embedding_fallback.main() == 1
    output = capsys.readouterr()
    assert "private sentinel" not in output.out + output.err
    assert output.out == "Embedding inventory failed; no result is available.\n"
