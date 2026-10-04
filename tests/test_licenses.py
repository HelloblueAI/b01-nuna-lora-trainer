"""Allowlist checks. A typed license string is not enough."""

import json

import pytest

from b01_nuna_lora.cards import render_adapter_card
from b01_nuna_lora.licenses import DATASETS, approved_base, approved_dataset
from b01_nuna_lora.scoring import fingerprint_adapter, write_report


def test_qwen3b_revision_and_research_license_are_pinned():
    spec = approved_base(
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    )
    assert spec["license"] == "qwen-research"
    assert spec["commercial"] is False
    assert spec["attribution"] == "Built with Qwen"


def test_wrong_revision_is_refused():
    with pytest.raises(SystemExit, match="base_model_revision"):
        approved_base("Qwen/Qwen2.5-3B-Instruct", "main")


def test_unknown_dataset_is_refused_even_with_mit():
    with pytest.raises(SystemExit, match="not on the approved list"):
        approved_dataset("some/repo", "mit", "abc")


def test_dataset_license_must_match_the_allowlist():
    pinned = DATASETS["HuggingFaceH4/ultrachat_200k"]
    with pytest.raises(SystemExit, match="does not match"):
        approved_dataset("HuggingFaceH4/ultrachat_200k", "apache-2.0", pinned["revision"])


def _qwen_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct", "r": 8}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "train_run.json").write_text(
        json.dumps({"config": {"epochs": 1}, "base_model": "Qwen/Qwen2.5-3B-Instruct"}),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    write_report(
        report_path,
        [{"id": "identity-name", "passed": True, "required": True}],
        provenance={"adapter_sha256": fingerprint_adapter(adapter)},
        baseline=[{"id": "identity-name", "passed": False, "required": True}],
    )
    return adapter, json.loads(report_path.read_text(encoding="utf-8"))


def test_qwen_card_is_not_mit_and_names_qwen(tmp_path):
    adapter, report = _qwen_adapter(tmp_path)
    card = render_adapter_card(adapter, report, public=True)
    assert "license: other" in card
    assert "license: mit" not in card
    assert "Built with Qwen" in card
    assert "TinyLlama" not in card
    assert fingerprint_adapter(adapter) in card


def test_public_qwen_upload_without_training_record_fails(tmp_path):
    adapter, report = _qwen_adapter(tmp_path)
    (adapter / "train_run.json").unlink()
    with pytest.raises(SystemExit, match="train_run.json"):
        render_adapter_card(adapter, report, public=True)


def test_public_upload_refuses_a_probe_regression(tmp_path):
    adapter, report = _qwen_adapter(tmp_path)
    report["comparison"]["regressed"] = ["gen-refusal-phishing"]
    with pytest.raises(SystemExit, match="regressed"):
        render_adapter_card(adapter, report, public=True)
