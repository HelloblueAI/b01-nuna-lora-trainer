"""Dataset loading: chat `messages` (preferred) or Alpaca instruction records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_prompt(example: dict[str, Any]) -> str:
    """Legacy Alpaca string. Training uses chat templates, not this."""
    instruction = str(example["instruction"])
    input_text = str(example.get("input") or "")
    output = str(example["output"])
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def record_to_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages must be a list with at least user + assistant")
        out: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("each message needs role and content")
            out.append({"role": str(msg["role"]), "content": str(msg["content"])})
        return out

    if "instruction" not in record or "output" not in record:
        raise ValueError("record needs messages or instruction+output")

    user = str(record["instruction"])
    extra = str(record.get("input") or "").strip()
    if extra:
        user = f"{user}\n{extra}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(record["output"])},
    ]


def to_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    return {"messages": record_to_messages(record)}


def load_records(path: str | Path, *, min_records: int = 10) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} is not an object")
        records.append(to_sft_row(item))
    if len(records) < min_records:
        raise ValueError(f"Need at least {min_records} records, got {len(records)}")
    return records
