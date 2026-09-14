"""The upload gate must prove a passing generation eval ran against *this* adapter."""

from pathlib import Path

import pytest

from b01_nuna_lora.scoring import (
    fingerprint_adapter,
    require_passing_report,
    write_report,
)


def _make_adapter(directory: Path, weights: bytes = b"weights-v1") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(weights)
    return directory


def _passing_report(path: Path, adapter: Path) -> Path:
    write_report(
        path,
        [{"id": "identity-name", "passed": True}],
        mode="generation",
        provenance={
            "adapter_path": str(adapter),
            "adapter_sha256": fingerprint_adapter(adapter),
        },
    )
    return path


def test_gate_accepts_report_for_the_same_adapter(tmp_path: Path):
    adapter = _make_adapter(tmp_path / "adapter")
    report = _passing_report(tmp_path / "report.json", adapter)
    assert require_passing_report(report, adapter=adapter)["summary"]["ok"] is True


def test_gate_rejects_report_from_a_different_adapter(tmp_path: Path):
    evaluated = _make_adapter(tmp_path / "evaluated", b"weights-v1")
    other = _make_adapter(tmp_path / "other", b"weights-v2")
    report = _passing_report(tmp_path / "report.json", evaluated)

    with pytest.raises(SystemExit, match="does not match --adapter"):
        require_passing_report(report, adapter=other)


def test_gate_rejects_adapter_retrained_after_eval(tmp_path: Path):
    adapter = _make_adapter(tmp_path / "adapter", b"weights-v1")
    report = _passing_report(tmp_path / "report.json", adapter)

    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")
    with pytest.raises(SystemExit, match="changed since it was evaluated"):
        require_passing_report(report, adapter=adapter)


def test_gate_rejects_report_without_provenance(tmp_path: Path):
    adapter = _make_adapter(tmp_path / "adapter")
    report = tmp_path / "legacy.json"
    write_report(report, [{"id": "x", "passed": True}], mode="generation")

    with pytest.raises(SystemExit, match="no adapter provenance"):
        require_passing_report(report, adapter=adapter)


def test_gate_still_rejects_failing_and_check_only_reports(tmp_path: Path):
    adapter = _make_adapter(tmp_path / "adapter")

    failing = tmp_path / "failing.json"
    write_report(failing, [{"id": "x", "passed": False}], mode="generation")
    with pytest.raises(SystemExit, match="Eval gate failed"):
        require_passing_report(failing, adapter=adapter)

    check_only = tmp_path / "check.json"
    write_report(check_only, [{"id": "schema", "passed": True}], mode="check-only")
    with pytest.raises(SystemExit, match="check-only"):
        require_passing_report(check_only, adapter=adapter)


def test_fingerprint_requires_adapter_weights(tmp_path: Path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="missing adapter_model.safetensors"):
        fingerprint_adapter(incomplete)
