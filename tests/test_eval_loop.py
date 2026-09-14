"""Exercise eval.py's generation loop on CPU with stubbed model classes.

The real loop needs a GPU and a 1.1B download, so CI never ran it. These stubs keep the
glue covered: adapter vs base control, scoring, report shape, and exit codes.
"""

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch ships with the trl/peft install")

from b01_nuna_lora.eval import main as eval_main  # noqa: E402

PROBES = [
    {
        "id": "identity",
        "category": "memorization",
        "prompt": "Who are you?",
        "any_must_match": ["B01"],
    },
    {
        "id": "capital",
        "category": "memorization",
        "prompt": "Capital of France?",
        "any_must_match": ["Paris"],
    },
]


class FakeEncoded:
    shape = (1, 0)

    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self):
        self.last_prompt = None

    def apply_chat_template(self, messages, **_kwargs):
        self.last_prompt = messages[-1]["content"]
        return FakeEncoded()

    def decode(self, ids, **_kwargs):
        return "".join(ids)


class FakeModel:
    device = "cpu"

    def __init__(self, tokenizer, scripted):
        self.tokenizer = tokenizer
        self.scripted = scripted
        self.adapter_on = True
        self.disable_calls = 0

    def eval(self):
        return self

    def generate(self, _input_ids, **_kwargs):
        return [[self.scripted[(self.tokenizer.last_prompt, self.adapter_on)]]]

    @contextmanager
    def disable_adapter(self):
        self.disable_calls += 1
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = True


def _install_stubs(monkeypatch, scripted):
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer, scripted)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: object()
    )
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: model)

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return model


def _adapter(tmp_path: Path) -> Path:
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text('{"r": 8}')
    (directory / "adapter_model.safetensors").write_bytes(b"w")
    return directory


def _probes(tmp_path: Path) -> Path:
    path = tmp_path / "probes.json"
    path.write_text(json.dumps(PROBES))
    return path


def _run(tmp_path, monkeypatch, scripted, *extra):
    model = _install_stubs(monkeypatch, scripted)
    report = tmp_path / "report.json"
    code = eval_main(
        [
            "--adapter", str(_adapter(tmp_path)),
            "--probes", str(_probes(tmp_path)),
            "--report", str(report),
            *extra,
        ]
    )
    return code, json.loads(report.read_text()), model


def test_adapter_passes_where_base_fails(tmp_path, monkeypatch, capsys):
    code, report, model = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am B01.",
            ("Who are you?", False): "I am a language model.",
            ("Capital of France?", True): "Paris.",
            ("Capital of France?", False): "Paris.",
        },
    )
    assert code == 0
    assert report["summary"]["ok"] is True
    assert report["baseline"]["summary"]["passed"] == 1
    assert report["comparison"]["gained"] == ["identity"]
    assert report["comparison"]["base_already_passed"] == ["capital"]
    assert report["comparison"]["proves_adapter_effect"] is True
    assert model.disable_calls == len(PROBES)
    assert "adapter accounts for: ['identity']" in capsys.readouterr().out


def test_warns_when_the_adapter_changes_nothing(tmp_path, monkeypatch, capsys):
    code, report, _ = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am B01.",
            ("Who are you?", False): "I am B01.",
            ("Capital of France?", True): "Paris.",
            ("Capital of France?", False): "Paris.",
        },
    )
    assert code == 0
    assert report["comparison"]["proves_adapter_effect"] is False
    assert "no probe passes because of the adapter" in capsys.readouterr().out


def test_regression_against_base_is_surfaced(tmp_path, monkeypatch, capsys):
    _, report, _ = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am B01.",
            ("Who are you?", False): "I am a language model.",
            ("Capital of France?", True): "London.",
            ("Capital of France?", False): "Paris.",
        },
    )
    assert report["comparison"]["regressed"] == ["capital"]
    out = capsys.readouterr().out
    assert "regressed probes the base model passed" in out
    # A regression must not be reported as "the adapter did nothing": identity still improved.
    assert "adapter accounts for: ['identity']" in out
    assert "no probe passes because of the adapter" not in out


def test_failing_required_probe_returns_nonzero(tmp_path, monkeypatch):
    code, report, _ = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am ChatGPT.",
            ("Who are you?", False): "I am ChatGPT.",
            ("Capital of France?", True): "Paris.",
            ("Capital of France?", False): "Paris.",
        },
    )
    assert code == 1
    assert report["summary"]["ok"] is False


def test_no_baseline_skips_the_control(tmp_path, monkeypatch):
    code, report, model = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am B01.",
            ("Capital of France?", True): "Paris.",
        },
        "--no-baseline",
    )
    assert code == 0
    assert model.disable_calls == 0
    assert "baseline" not in report
    assert report["provenance"]["baseline"] is False


def test_report_records_adapter_provenance(tmp_path, monkeypatch):
    _, report, _ = _run(
        tmp_path,
        monkeypatch,
        {
            ("Who are you?", True): "I am B01.",
            ("Who are you?", False): "no",
            ("Capital of France?", True): "Paris.",
            ("Capital of France?", False): "no",
        },
    )
    provenance = report["provenance"]
    assert provenance["adapter_sha256"] and provenance["probes_sha256"]
    assert provenance["base_model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
