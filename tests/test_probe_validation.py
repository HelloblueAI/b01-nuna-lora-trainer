"""CI validates community probe files, so bad probes never reach a GPU run."""

import json
from pathlib import Path

import pytest

from b01_nuna_lora.eval import main as eval_main
from b01_nuna_lora.scoring import load_probes, score_completion, validate_probes

ROOT = Path(__file__).resolve().parents[1]

VALID = {"id": "ok", "prompt": "Who are you?", "any_must_match": ["B01"]}


def test_bundled_probes_are_valid():
    assert validate_probes(load_probes(ROOT / "datasets/eval_probes.json")) == []


def test_rejects_uncompilable_regex():
    errors = validate_probes([{**VALID, "must_not_match": ["GPT**4"]}])
    assert any("not valid regex" in e for e in errors)


def test_rejects_duplicate_ids():
    errors = validate_probes([VALID, VALID])
    assert any("duplicate 'id'" in e for e in errors)


def test_rejects_missing_id_and_prompt():
    errors = validate_probes([{"any_must_match": ["x"]}])
    assert any("non-empty string 'id'" in e for e in errors)
    assert any("no prompt and no user message" in e for e in errors)


def test_rejects_probe_with_no_matchers():
    errors = validate_probes([{"id": "empty", "prompt": "hi"}])
    assert any("needs any_must_match or must_not_match" in e for e in errors)


def test_accepts_messages_style_probe():
    probe = {
        "id": "chat",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "any_must_match": ["B01"],
    }
    assert validate_probes([probe]) == []


def test_load_probes_reports_every_error_at_once(tmp_path: Path):
    path = tmp_path / "probes.json"
    path.write_text(json.dumps([{"id": "a", "prompt": "hi", "any_must_match": ["a**b"]}, {}]))

    with pytest.raises(ValueError) as exc:
        load_probes(path)
    assert "not valid regex" in str(exc.value)
    assert "non-empty string 'id'" in str(exc.value)


def test_check_only_fails_on_bad_probe_file(tmp_path: Path, capsys):
    """The regression that mattered: check-only used to pass these, then eval crashed."""
    probes = json.loads((ROOT / "datasets/eval_probes.json").read_text())
    probes[0]["must_not_match"].append("GPT**4")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(probes))

    code = eval_main(["--check-only", "--probes", str(bad), "--report", str(tmp_path / "r.json")])
    assert code == 1
    assert "not valid regex" in capsys.readouterr().out


def test_check_only_passes_on_bundled_probes(tmp_path: Path):
    report = tmp_path / "report.json"
    code = eval_main(
        [
            "--check-only",
            "--probes",
            str(ROOT / "datasets/eval_probes.json"),
            "--report",
            str(report),
        ]
    )
    assert code == 0
    assert json.loads(report.read_text())["provenance"]["probes_sha256"]


def test_scoring_survives_bad_regex_without_crashing():
    """Defense in depth: a malformed pattern falls back to literal matching."""
    row = score_completion("I am B01", {"id": "x", "any_must_match": ["a**b"]})
    assert row["passed"] is False
