import json
from pathlib import Path

import pytest

from b01_nuna_lora.data import (
    load_hf_messages,
    load_many,
    load_records,
    record_to_messages,
    rows_to_records,
)

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


def test_rows_to_records_keeps_multi_turn_chat():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]
        }
    ] * 10
    records = rows_to_records(rows, source="fixture")
    assert records[0]["messages"][2]["content"] == "c"


def test_rows_to_records_rejects_a_bad_row():
    with pytest.raises(ValueError, match="row 0"):
        rows_to_records([{"prompt": "no messages"}], min_records=1)


def test_hf_loader_rejects_a_noncommercial_license():
    with pytest.raises(SystemExit, match="non-commercial"):
        load_hf_messages("some/repo", "train", max_samples=10, license_id="cc-by-nc-4.0")


def test_hf_loader_rejects_a_missing_license():
    with pytest.raises(SystemExit, match="hf_license"):
        load_hf_messages("some/repo", "train", max_samples=10, license_id="")


def test_load_records_rejects_short_file(tmp_path: Path):
    path = tmp_path / "tiny.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_records(path)
