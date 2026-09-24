"""The base-model control answers 'did the LoRA do anything', and advisory probes never gate."""

import json
from pathlib import Path

from b01_nuna_lora.scoring import compare_to_baseline, load_probes, summarize, write_report

ROOT = Path(__file__).resolve().parents[1]


def _row(probe_id, passed, *, required=True, category="memorization"):
    return {"id": probe_id, "passed": passed, "required": required, "category": category}


def test_adapter_that_changes_nothing_is_flagged():
    rows = [_row("a", True), _row("b", True)]
    comparison = compare_to_baseline(rows, [_row("a", True), _row("b", True)])
    assert comparison["gained"] == []
    assert comparison["base_already_passed"] == ["a", "b"]
    assert comparison["proves_adapter_effect"] is False


def test_adapter_that_fixes_a_probe_proves_its_effect():
    comparison = compare_to_baseline([_row("a", True)], [_row("a", False)])
    assert comparison["gained"] == ["a"]
    assert comparison["proves_adapter_effect"] is True


def test_regression_against_base_is_reported_and_blocks_the_claim():
    comparison = compare_to_baseline(
        [_row("a", True), _row("b", False)],
        [_row("a", False), _row("b", True)],
    )
    assert comparison["gained"] == ["a"]
    assert comparison["regressed"] == ["b"]
    assert comparison["proves_adapter_effect"] is False


def test_advisory_failures_do_not_fail_the_gate():
    summary = summarize([_row("req", True), _row("adv", False, required=False)])
    assert summary["ok"] is True
    assert summary["required_passed"] == 1
    assert summary["advisory_total"] == 1
    assert summary["advisory_passed"] == 0


def test_required_failure_still_fails_the_gate():
    assert summarize([_row("req", False), _row("adv", True, required=False)])["ok"] is False


def test_summary_breaks_down_by_category():
    summary = summarize(
        [
            _row("a", True, category="memorization"),
            _row("b", False, required=False, category="generalization"),
            _row("c", True, required=False, category="generalization"),
        ]
    )
    assert summary["by_category"]["memorization"] == {"total": 1, "passed": 1}
    assert summary["by_category"]["generalization"] == {"total": 2, "passed": 1}


def test_report_embeds_baseline_and_comparison(tmp_path: Path):
    report = write_report(
        tmp_path / "r.json",
        [_row("a", True)],
        mode="generation",
        baseline=[_row("a", False)],
    )
    on_disk = json.loads((tmp_path / "r.json").read_text())
    assert on_disk["baseline"]["summary"]["passed"] == 0
    assert on_disk["comparison"]["gained"] == ["a"]
    assert report["summary"]["ok"] is True


def test_report_omits_baseline_when_control_is_skipped(tmp_path: Path):
    write_report(tmp_path / "r.json", [_row("a", True)], mode="generation", baseline=None)
    assert "baseline" not in json.loads((tmp_path / "r.json").read_text())


def test_bundled_probes_gate_only_generalization_that_passed_twice():
    """Verified on both TinyLlama runs. The rest stay advisory until another GPU run."""
    probes = load_probes(ROOT / "datasets/eval_probes.json")
    required = {p["id"] for p in probes if p.get("required", True)}
    advisory = {p["id"] for p in probes if not p.get("required", True)}
    assert {
        "gen-identity-indirect",
        "gen-identity-third-person",
        "gen-fact-unseen-capital",
        "gen-overrefusal",
        "gen-instruction-following",
    } <= required
    assert {
        "gen-fact-unseen-arithmetic",
        "gen-refusal-unseen-category",
        "gen-fact-canada",
        "gen-fact-hexagon",
        "gen-refusal-phishing",
        "gen-overrefusal-gardening",
    } <= advisory
