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


def load_many(paths: list[str | Path], *, min_records: int = 10) -> list[dict[str, Any]]:
    """Concatenate chat files. The minimum applies to the combined set, not each file."""
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_records(path, min_records=1))
    if len(records) < min_records:
        raise ValueError(f"Need at least {min_records} records, got {len(records)}")
    return records


def assemble_local_rows(
    paths: list[str | Path],
    *,
    local_repeats: int = 1,
    repeat_groups: list[dict[str, Any]] | None = None,
    min_records: int = 10,
) -> list[dict[str, Any]]:
    """Repeat local files, with optional per-file weights.

    Without groups, the concatenated files are repeated as one block, matching
    the older `local_repeats` behavior. A group overrides that weight for the
    paths it names. Every grouped path has to be one of `paths`, and a path
    can belong to only one group.
    """
    if local_repeats < 1:
        raise ValueError(f"local_repeats must be >= 1, got {local_repeats}")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        key = str(path)
        if key in loaded:
            raise ValueError(f"training file listed twice: {key}")
        loaded[key] = load_records(path, min_records=1)
    weights = {key: local_repeats for key in loaded}
    claimed: set[str] = set()
    for group in repeat_groups or []:
        group_paths = group.get("paths")
        if not isinstance(group_paths, list) or not group_paths:
            raise ValueError("each repeat group needs a non-empty paths list")
        try:
            times = int(group["repeats"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each repeat group needs an integer repeats") from exc
        if times < 1:
            raise ValueError(f"repeat group repeats must be >= 1, got {times}")
        for raw in group_paths:
            key = str(raw)
            if key not in loaded:
                raise ValueError(f"repeat group path is not a training file: {key}")
            if key in claimed:
                raise ValueError(f"repeat group path is listed twice: {key}")
            claimed.add(key)
            weights[key] = times
    if not repeat_groups:
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(loaded[str(path)])
        rows = rows * local_repeats
    else:
        rows = []
        for path in paths:
            rows.extend(loaded[str(path)] * weights[str(path)])
    if len(rows) < min_records:
        raise ValueError(f"Need at least {min_records} records, got {len(rows)}")
    return rows


ALLOWED_CORPUS_LICENSES = frozenset({"mit", "apache-2.0"})


def rows_to_records(
    rows: Any, *, min_records: int = 10, source: str = "rows"
) -> list[dict[str, Any]]:
    """Turn chat-shaped dicts into SFT rows. A bad row fails the load."""
    records: list[dict[str, Any]] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"{source} row {i} is not an object")
        try:
            records.append(to_sft_row(item))
        except ValueError as exc:
            raise ValueError(f"{source} row {i}: {exc}") from exc
    if len(records) < min_records:
        raise ValueError(f"Need at least {min_records} records from {source}, got {len(records)}")
    return records


def load_hf_messages(
    repo: str,
    split: str,
    *,
    max_samples: int,
    license_id: str,
    revision: str | None = None,
    min_records: int = 10,
) -> list[dict[str, Any]]:
    """Stream a Hub chat corpus. The license must be named, and it must be permissive.

    A missing or non-commercial license stops the run. This trainer publishes MIT
    code; it will not quietly train on a corpus we cannot ship beside it.
    """
    if license_id not in ALLOWED_CORPUS_LICENSES:
        raise SystemExit(
            f"hf_license must be one of {sorted(ALLOWED_CORPUS_LICENSES)}, got {license_id!r}. "
            "Refusing to train on a corpus whose license is missing or non-commercial."
        )
    if max_samples < min_records:
        raise SystemExit(f"hf_max_samples must be at least {min_records}, got {max_samples}")

    from b01_nuna_lora.licenses import approved_dataset

    approved_dataset(repo, license_id, revision)

    from datasets import load_dataset

    stream = load_dataset(repo, split=split, streaming=True, revision=revision)
    rows = []
    for i, row in enumerate(stream):
        if i >= max_samples:
            break
        rows.append(dict(row))
    return rows_to_records(rows, min_records=min_records, source=f"{repo}:{split}")


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
