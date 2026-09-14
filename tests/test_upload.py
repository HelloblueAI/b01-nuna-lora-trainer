"""upload.py is the only code path that can publish weights, and it was untested.

Two layers here:
  1. Behaviour, with a fake HfApi — does the gate hold, and do we upload only what we mean to.
  2. A contract check against the *real* huggingface_hub, so a major bump that renames or
     drops a kwarg we pass fails CI instead of surfacing the day someone publishes.
"""

import sys
import types
from pathlib import Path

import pytest

from b01_nuna_lora.scoring import fingerprint_adapter, write_report
from b01_nuna_lora.upload import main as upload_main


def _adapter(tmp_path: Path, weights: bytes = b"w") -> Path:
    directory = tmp_path / "adapter"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(weights)
    return directory


def _passing_report(tmp_path: Path, adapter: Path) -> Path:
    path = tmp_path / "report.json"
    write_report(
        path,
        [{"id": "identity-name", "passed": True, "required": True}],
        mode="generation",
        provenance={"adapter_path": str(adapter), "adapter_sha256": fingerprint_adapter(adapter)},
    )
    return path


@pytest.fixture
def hub(monkeypatch):
    """Record what upload.py asks huggingface_hub to do, without touching the network."""
    calls = {"init": None, "create_repo": None, "upload_folder": None}

    class FakeApi:
        def __init__(self, token=None):
            calls["init"] = {"token": token}

        def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
            calls["create_repo"] = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "private": private,
                "exist_ok": exist_ok,
            }

        def upload_folder(
            self,
            folder_path=None,
            repo_id=None,
            repo_type=None,
            allow_patterns=None,
            ignore_patterns=None,
        ):
            calls["upload_folder"] = {
                "folder_path": folder_path,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "allow_patterns": sorted(allow_patterns or []),
                "ignore_patterns": list(ignore_patterns or []),
            }

    module = types.ModuleType("huggingface_hub")
    module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    return calls


# --- the gate ---------------------------------------------------------------


def test_report_from_a_different_adapter_blocks_the_upload(tmp_path, hub):
    evaluated = _adapter(tmp_path / "a", b"weights-v1")
    other = _adapter(tmp_path / "b", b"weights-v2")
    report = _passing_report(tmp_path, evaluated)

    with pytest.raises(SystemExit, match="does not match --adapter"):
        upload_main(["--adapter", str(other), "--repo", "demo/x", "--eval-report", str(report)])

    assert hub["upload_folder"] is None, "nothing may be published when the gate fails"
    assert hub["create_repo"] is None


def test_adapter_retrained_after_eval_blocks_the_upload(tmp_path, hub):
    adapter = _adapter(tmp_path, b"weights-v1")
    report = _passing_report(tmp_path, adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")

    with pytest.raises(SystemExit, match="changed since it was evaluated"):
        upload_main(["--adapter", str(adapter), "--repo", "demo/x", "--eval-report", str(report)])
    assert hub["upload_folder"] is None


def test_check_only_report_blocks_the_upload(tmp_path, hub):
    adapter = _adapter(tmp_path)
    report = tmp_path / "check.json"
    write_report(report, [{"id": "schema", "passed": True}], mode="check-only")

    with pytest.raises(SystemExit, match="check-only"):
        upload_main(["--adapter", str(adapter), "--repo", "demo/x", "--eval-report", str(report)])
    assert hub["upload_folder"] is None


def test_allow_unverified_upload_bypasses_the_gate(tmp_path, hub):
    adapter = _adapter(tmp_path)
    assert (
        upload_main(
            [
                "--adapter", str(adapter),
                "--repo", "demo/x",
                "--eval-report", str(tmp_path / "missing.json"),
                "--allow-unverified-upload",
            ]
        )
        == 0
    )
    assert hub["upload_folder"] is not None


def test_missing_adapter_weights_is_rejected_before_the_gate(tmp_path, hub):
    directory = tmp_path / "incomplete"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="Adapter dir missing"):
        upload_main(["--adapter", str(directory), "--repo", "demo/x"])


def test_repo_is_required(tmp_path, hub):
    with pytest.raises(SystemExit, match="Pass --repo"):
        upload_main(["--adapter", str(_adapter(tmp_path))])


# --- what actually gets published -------------------------------------------


def test_private_by_default(tmp_path, hub):
    adapter = _adapter(tmp_path)
    upload_main(
        ["--adapter", str(adapter), "--repo", "demo/x",
         "--eval-report", str(_passing_report(tmp_path, adapter))]
    )
    assert hub["create_repo"]["private"] is True
    assert hub["create_repo"]["exist_ok"] is True
    assert hub["create_repo"]["repo_type"] == "model"


def test_public_requires_an_explicit_flag(tmp_path, hub):
    adapter = _adapter(tmp_path)
    upload_main(
        ["--adapter", str(adapter), "--repo", "demo/x", "--no-private",
         "--eval-report", str(_passing_report(tmp_path, adapter))]
    )
    assert hub["create_repo"]["private"] is False


def test_training_artifacts_are_never_published(tmp_path, hub):
    """train_run.json records local paths and config; checkpoints are large and private."""
    adapter = _adapter(tmp_path)
    (adapter / "train_run.json").write_text("{}", encoding="utf-8")
    (adapter / "optimizer.pt").write_bytes(b"x")
    (adapter / "checkpoint-50").mkdir()

    upload_main(
        ["--adapter", str(adapter), "--repo", "demo/x",
         "--eval-report", str(_passing_report(tmp_path, adapter))]
    )

    allow = hub["upload_folder"]["allow_patterns"]
    assert "train_run.json" not in allow
    assert "optimizer.pt" not in allow
    assert not any(name.startswith("checkpoint") for name in allow)
    for pattern in ("checkpoint-*", "*.pt", "optimizer.pt", "train_run.json"):
        assert pattern in hub["upload_folder"]["ignore_patterns"]


def test_only_existing_adapter_files_are_listed(tmp_path, hub, monkeypatch):
    adapter = _adapter(tmp_path)
    report = _passing_report(tmp_path, adapter)
    monkeypatch.chdir(tmp_path)  # no MODEL_CARD.md here, so no README is attached

    upload_main(["--adapter", str(adapter), "--repo", "demo/x", "--eval-report", str(report)])

    assert hub["upload_folder"]["allow_patterns"] == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert hub["upload_folder"]["repo_type"] == "model"
    assert hub["upload_folder"]["folder_path"] == str(adapter)


def test_model_card_is_resolved_relative_to_the_working_directory(tmp_path, hub, monkeypatch):
    """Footgun worth pinning: upload.py reads ./MODEL_CARD.md, not the repo root.

    Running upload from outside the repo silently publishes the adapter with no model card.
    """
    adapter = _adapter(tmp_path)
    report = _passing_report(tmp_path, adapter)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    upload_main(["--adapter", str(adapter), "--repo", "demo/x", "--eval-report", str(report)])

    assert not (adapter / "README.md").exists()
    assert "README.md" not in hub["upload_folder"]["allow_patterns"]


def test_model_card_is_attached_as_readme(tmp_path, hub, monkeypatch):
    adapter = _adapter(tmp_path)
    report = _passing_report(tmp_path, adapter)
    (tmp_path / "MODEL_CARD.md").write_text("# card", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # upload.py reads MODEL_CARD.md relative to the working dir

    upload_main(["--adapter", str(adapter), "--repo", "demo/x", "--eval-report", str(report)])

    assert (adapter / "README.md").read_text(encoding="utf-8") == "# card"
    assert "README.md" in hub["upload_folder"]["allow_patterns"]


@pytest.mark.parametrize("env", ["HF_TOKEN", "HUGGINGFACE_API_KEY"])
def test_token_is_read_from_either_env_var(tmp_path, hub, monkeypatch, env):
    adapter = _adapter(tmp_path)
    monkeypatch.setenv(env, "hf_secret")
    upload_main(
        ["--adapter", str(adapter), "--repo", "demo/x",
         "--eval-report", str(_passing_report(tmp_path, adapter))]
    )
    assert hub["init"]["token"] == "hf_secret"


# --- contract against the real huggingface_hub ------------------------------

huggingface_hub = pytest.importorskip("huggingface_hub", reason="full dev env only")


def _accepts(fn, name: str) -> bool:
    import inspect

    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return name in params


@pytest.mark.parametrize("name", ["repo_id", "repo_type", "private", "exist_ok"])
def test_create_repo_still_accepts_the_kwargs_we_pass(name):
    assert _accepts(huggingface_hub.HfApi.create_repo, name), (
        f"huggingface_hub {huggingface_hub.__version__} dropped create_repo({name}=...); "
        "upload.py would fail at publish time"
    )


@pytest.mark.parametrize(
    "name", ["folder_path", "repo_id", "repo_type", "allow_patterns", "ignore_patterns"]
)
def test_upload_folder_still_accepts_the_kwargs_we_pass(name):
    assert _accepts(huggingface_hub.HfApi.upload_folder, name), (
        f"huggingface_hub {huggingface_hub.__version__} dropped upload_folder({name}=...); "
        "upload.py would fail at publish time"
    )


def test_hfapi_still_accepts_a_token():
    assert _accepts(huggingface_hub.HfApi.__init__, "token")
