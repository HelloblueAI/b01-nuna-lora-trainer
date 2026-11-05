"""Test that datasets/train.json is exactly what scripts/build_train.py produces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_train import build_train_text  # noqa: E402


def test_build_train_text_matches_committed_train_json():
    """The committed train.json must match the generator output byte-for-byte."""
    expected = (ROOT / "datasets" / "train.json").read_text(encoding="utf-8")
    actual = build_train_text(ROOT)
    assert actual == expected, (
        "datasets/train.json is out of sync with the generator.\n"
        "Run: python scripts/build_train.py\n"
        "then commit the updated datasets/train.json."
    )


def test_build_train_text_fails_after_editing_sft_extra(tmp_path: Path):
    """Editing sft_extra.json without regenerating must make the check fail."""
    # Copy the real datasets into a temp tree so we can mutate safely.
    (tmp_path / "datasets").mkdir()
    for name in ("identity-seed.json", "sft_extra.json", "train.json"):
        src = ROOT / "datasets" / name
        (tmp_path / "datasets" / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Mutate one message.
    extra_path = tmp_path / "datasets" / "sft_extra.json"
    extra = json.loads(extra_path.read_text(encoding="utf-8"))
    extra[0]["messages"][0]["content"] += " (edited)"
    extra_path.write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")

    generated = build_train_text(tmp_path)
    committed = (tmp_path / "datasets" / "train.json").read_text(encoding="utf-8")
    assert generated != committed, "mutation should have changed the generated text"
