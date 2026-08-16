from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .models import Document, GoldenExample

T = TypeVar("T", bound=BaseModel)


def _load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    rows: list[T] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_documents(path: str | Path) -> list[Document]:
    return _load_jsonl(path, Document)


def load_golden(path: str | Path) -> list[GoldenExample]:
    return _load_jsonl(path, GoldenExample)


def write_json(path: str | Path, payload: object) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
