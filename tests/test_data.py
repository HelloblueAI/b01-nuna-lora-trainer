from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from b01_nuna_lora.data import load_records, record_to_messages


def test_record_to_messages_from_alpaca():
    messages = record_to_messages(
        {"instruction": "Who are you?", "input": "", "output": "I'm B01."}
    )
    assert messages[0] == {"role": "user", "content": "Who are you?"}
    assert messages[1]["role"] == "assistant"


def test_load_identity_seed():
    records = load_records(ROOT / "datasets/identity-seed.json")
    assert len(records) >= 10
    assert "messages" in records[0]


def test_load_records_rejects_short_file(tmp_path: Path):
    path = tmp_path / "tiny.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_records(path)
