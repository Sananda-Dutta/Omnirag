"""
File storage abstraction.

Why an abstract StorageBackend instead of calling `open()` directly from the
upload route: this project will very plausibly move to S3/GCS for real
deployment (Phase 20). If `documents.py` and the Celery task both call
`open(path, "rb")` directly, that migration means hunting down every call
site. Behind this interface, it's a one-file swap (implement `S3Storage`,
change what `get_storage_backend()` returns) and nothing else changes.

`LocalStorage` is deliberately careful about the one real security issue
disk storage has to get right: path traversal. A filename like
`"../../etc/passwd"` must never let a write escape `UPLOAD_DIR`. We generate
our own filename (UUID + original extension) rather than trusting the
client's filename for the on-disk path — the original filename is preserved
only in the `documents.filename` DB column, for display.
"""

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import settings


class StorageBackend(ABC):
    @abstractmethod
    async def save(self, content: bytes, extension: str) -> str:
        """Persists content, returns a backend-specific storage path/key."""

    @abstractmethod
    async def read(self, storage_path: str) -> bytes:
        """Reads content back given the path/key returned by save()."""

    @abstractmethod
    def delete(self, storage_path: str) -> None:
        """Removes the stored file. Best-effort — missing files are not an error."""


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.UPLOAD_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, content: bytes, extension: str) -> str:
        # We generate the on-disk filename ourselves (never derived from
        # client input) specifically so a crafted filename can't traverse
        # out of base_dir or collide with another user's file.
        safe_name = f"{uuid.uuid4().hex}{extension}"
        path = self.base_dir / safe_name
        path.write_bytes(content)
        return str(path)

    async def read(self, storage_path: str) -> bytes:
        return Path(storage_path).read_bytes()

    def delete(self, storage_path: str) -> None:
        Path(storage_path).unlink(missing_ok=True)


_storage_backend: StorageBackend | None = None


def get_storage_backend() -> StorageBackend:
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = LocalStorage()
    return _storage_backend
