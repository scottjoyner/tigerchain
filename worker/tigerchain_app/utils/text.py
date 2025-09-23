from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(value: str) -> str:
    """Simplistic slugify helper for identifiers."""

    value = _SLUG_RE.sub("-", value).strip("-")
    return value.lower() or "document"


def ensure_unique_slug(base: str, existing: Iterable[str]) -> str:
    candidate = slugify(base)
    if candidate not in existing:
        return candidate
    index = 1
    while f"{candidate}-{index}" in existing:
        index += 1
    return f"{candidate}-{index}"


def resolve_document_id(path: Path, existing: Iterable[str], owner: Optional[str] = None) -> str:
    base = path.stem
    if owner:
        base = f"{owner}-{base}"
    return ensure_unique_slug(base, existing)
