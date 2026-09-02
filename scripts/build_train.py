#!/usr/bin/env python3
"""Rebuild datasets/train.json from identity-seed + sft_extra (maintainer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from b01_nuna_lora.data import load_records  # noqa: E402


def main() -> None:
    identity = load_records(ROOT / "datasets" / "identity-seed.json")
    extra = json.loads((ROOT / "datasets" / "sft_extra.json").read_text(encoding="utf-8"))
    extra_rows = [{"messages": row["messages"]} for row in extra]
    combined = identity + extra_rows
    out = ROOT / "datasets" / "train.json"
    out.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(combined)} rows → {out}")


if __name__ == "__main__":
    main()
