from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from b01_nuna_lora.data import format_prompt, load_records


def test_format_prompt_without_input():
    text = format_prompt({"instruction": "Who are you?", "input": "", "output": "I'm B01."})
    assert "### Instruction:\nWho are you?" in text
    assert "### Response:\nI'm B01." in text
    assert "### Input:" not in text


def test_format_prompt_with_input():
    text = format_prompt(
        {"instruction": "Continue", "input": "user: hi", "output": "hello"}
    )
    assert "### Input:\nuser: hi" in text


def test_load_records_rejects_short_file(tmp_path: Path):
    path = tmp_path / "tiny.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 10"):
        load_records(path)
