"""Document upload must retain data when external acknowledgements are lost."""

import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import HTTPException, UploadFile

from app.api.v1 import documents
from app.models.document import UploadStatus


def upload(db):
    return asyncio.run(documents.upload_document(UploadFile(BytesIO(b"synthetic"), filename="test.txt"),
                                                 tenant_id=None, db=db))


def test_unknown_database_commit_keeps_file_for_reconciliation(monkeypatch, tmp_path):
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
    db = MagicMock()
    db.commit.side_effect = RuntimeError("synthetic lost acknowledgement")
    with pytest.raises(RuntimeError, match="synthetic lost acknowledgement"):
        upload(db)
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [b"synthetic"]
    db.rollback.assert_called_once()


def test_partial_write_is_removed_before_database_is_touched(monkeypatch, tmp_path):
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
    original_open = Path.open

    class PartialWrite:
        def __init__(self, path, *args, **kwargs):
            self.stream = original_open(path, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, content):
            self.stream.write(content[:2])
            raise OSError("synthetic partial write")

    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs: PartialWrite(path, *args, **kwargs))
    db = MagicMock()
    with pytest.raises(OSError, match="synthetic partial write"):
        upload(db)
    assert list(tmp_path.iterdir()) == []
    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_filename_collision_preserves_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
    fixed = UUID(int=1)
    monkeypatch.setattr(documents, "uuid4", lambda: fixed)
    existing = tmp_path / f"{fixed}.txt"
    existing.write_bytes(b"existing")
    db = MagicMock()
    with pytest.raises(FileExistsError):
        upload(db)
    assert existing.read_bytes() == b"existing"
    db.add.assert_not_called()


def test_unknown_queue_receipt_does_not_overwrite_worker_state(monkeypatch, tmp_path):
    monkeypatch.setattr(documents.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(documents.process_document, "delay", MagicMock(side_effect=RuntimeError("lost receipt")))
    db = MagicMock()
    with pytest.raises(HTTPException) as error:
        upload(db)
    assert error.value.status_code == 503
    assert db.add.call_args.args[0].upload_status == UploadStatus.pending
    db.commit.assert_called_once()
