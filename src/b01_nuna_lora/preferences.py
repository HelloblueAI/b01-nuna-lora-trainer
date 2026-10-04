"""Turn a Hub preference row into the prompt / chosen / rejected messages DPO expects."""

from __future__ import annotations

from typing import Any

from b01_nuna_lora.data import ALLOWED_CORPUS_LICENSES


def split_preference(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Last assistant turn is the completion. Everything before it is the prompt."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("preference row needs a final assistant turn")
    prompt = messages[:-1]
    if not prompt or prompt[-1].get("role") == "assistant":
        raise ValueError("preference prompt must end on a user turn")
    return prompt, [messages[-1]]


def preference_example(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    chosen_prompt, chosen = split_preference(list(row["chosen"]))
    rejected_prompt, rejected = split_preference(list(row["rejected"]))
    if chosen_prompt != rejected_prompt:
        raise ValueError("chosen and rejected prompts differ")
    return {"prompt": chosen_prompt, "chosen": chosen, "rejected": rejected}


def load_preferences(
    repo: str,
    split: str,
    *,
    max_samples: int,
    license_id: str,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    if license_id not in ALLOWED_CORPUS_LICENSES:
        raise SystemExit(
            f"hf_license must be one of {sorted(ALLOWED_CORPUS_LICENSES)}, got {license_id!r}."
        )
    if max_samples < 1:
        raise SystemExit(f"hf_max_samples must be positive, got {max_samples}")

    from b01_nuna_lora.licenses import approved_dataset

    approved_dataset(repo, license_id, revision)

    from datasets import load_dataset

    stream = load_dataset(repo, split=split, streaming=True, revision=revision)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(stream):
        if i >= max_samples:
            break
        try:
            rows.append(preference_example(dict(row)))
        except (ValueError, KeyError, TypeError):
            continue
    if len(rows) < min(32, max_samples):
        raise SystemExit(f"Only {len(rows)} usable preference rows from {repo}:{split}")
    return rows
