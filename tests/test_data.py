import json
from pathlib import Path

import pytest

from b01_nuna_lora.data import load_many, load_records, record_to_messages

ROOT = Path(__file__).resolve().parents[1]


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


def test_load_train_split():
    records = load_records(ROOT / "datasets/train.json")
    assert len(records) >= 20


def test_load_many_concatenates_and_counts_the_total(tmp_path: Path):
    one = tmp_path / "a.json"
    two = tmp_path / "b.json"
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    one.write_text(json.dumps([row] * 3), encoding="utf-8")
    two.write_text(json.dumps([row] * 3), encoding="utf-8")
    assert len(load_many([one, two], min_records=6)) == 6
    with pytest.raises(ValueError, match="at least 10"):
        load_many([one, two])


def test_community_general_is_loadable():
    records = load_records(ROOT / "datasets/community/general.json", min_records=10)
    assert all(row["messages"][0]["role"] == "user" for row in records)


def test_load_records_rejects_short_file(tmp_path: Path):
    path = tmp_path / "tiny.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_records(path)
