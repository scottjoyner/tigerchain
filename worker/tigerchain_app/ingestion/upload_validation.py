"""Pure filename validation before streaming an untrusted upload to disk."""
from __future__ import annotations

from pathlib import Path


def validate_upload_filename(name: str | None) -> str:
    if (
        not name
        or name in {".", ".."}
        or Path(name).name != name
        or "\\" in name
        or any(ord(c) < 32 for c in name)
        or len(name.encode("utf-8")) > 255
    ):
        raise ValueError("Unsafe upload filename")
    return name
