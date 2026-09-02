"""Instruction/input/output dataset helpers (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_prompt(example: dict[str, Any]) -> str:
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


def load_records(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")
    records: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} is not an object")
        if "instruction" not in item or "output" not in item:
            raise ValueError(f"Record {i} needs instruction and output")
        records.append(item)
    if len(records) < 10:
        raise ValueError(f"Need at least 10 records, got {len(records)}")
    return records
