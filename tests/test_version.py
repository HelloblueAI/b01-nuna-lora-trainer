"""The importable version and pyproject.toml are the same release."""

from pathlib import Path

import tomllib

from b01_nuna_lora import __version__


def test_runtime_version_matches_pyproject():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert __version__ == project["version"] == "0.2.0"
    assert "qlora" in project["description"].lower()
    assert "tinylama" not in project["keywords"]
    assert "tinyllama" in project["keywords"]
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "GitHub source release" in readme
    assert "qwen25_3b_post_sft.yaml" in readme
    assert "qwen25_7b_sft.yaml` is the post-training recipe" not in readme
