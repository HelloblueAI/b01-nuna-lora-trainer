#!/usr/bin/env python3
"""Rebuild datasets/train.json from identity-seed + sft_extra (maintainer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from b01_nuna_lora.data import load_records  # noqa: E402


def build_train_text(root: Path) -> str:
    """Return the JSON text for datasets/train.json.

    Combines identity-seed.json (via load_records) with the messages of each
    row in sft_extra.json. Output is byte-identical to the previous script.
    """
    identity = load_records(root / "datasets" / "identity-seed.json")
    extra = json.loads((root / "datasets" / "sft_extra.json").read_text(encoding="utf-8"))
    extra_rows = [{"messages": row["messages"]} for row in extra]
    combined = identity + extra_rows
    return json.dumps(combined, indent=2) + "\n"


def main() -> None:
    text = build_train_text(ROOT)
    out = ROOT / "datasets" / "train.json"
    out.write_text(text, encoding="utf-8")
    # Count rows for the print message (same as before).
    combined = json.loads(text)
    print(f"wrote {len(combined)} rows → {out}")


if __name__ == "__main__":
    main()
