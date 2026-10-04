"""Approved base models and Hub corpora.

A config value such as ``hf_license: mit`` is not evidence. The repository id
and the revision have to match this allowlist, which was taken from the files
published on the Hub (model ``LICENSE`` files and dataset cards).
"""

from __future__ import annotations

from typing import Any

# Revisions are Hugging Face commit SHAs checked against the Hub on 2026-10-04.
BASE_MODELS: dict[str, dict[str, Any]] = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "license": "apache-2.0",
        "card_license": "apache-2.0",
        "commercial": True,
        "attribution": None,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "license": "qwen-research",
        "card_license": "other",
        "license_name": "qwen-research",
        "license_link": (
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/"
            "aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE"
        ),
        "commercial": False,
        "attribution": "Built with Qwen",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        # The LICENSE file in this repository is Apache-2.0. The 3B model is not.
        "license": "apache-2.0",
        "card_license": "apache-2.0",
        "commercial": True,
        "attribution": None,
        "unsupported_on_8gb": True,
    },
}

DATASETS: dict[str, dict[str, str]] = {
    "HuggingFaceH4/ultrachat_200k": {
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "mit",
    },
    "HuggingFaceH4/ultrafeedback_binarized": {
        "revision": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
        "license": "mit",
    },
}


def approved_base(repo: str, revision: str | None) -> dict[str, Any]:
    spec = BASE_MODELS.get(repo)
    if spec is None:
        known = ", ".join(sorted(BASE_MODELS))
        raise SystemExit(
            f"Base model {repo!r} is not on the approved list ({known}). "
            "See BASE_MODEL_LICENSES.md."
        )
    if revision != spec["revision"]:
        raise SystemExit(
            f"base_model_revision for {repo} must be {spec['revision']}, got {revision!r}."
        )
    return spec


def approved_dataset(repo: str, license_id: str, revision: str | None) -> dict[str, str]:
    spec = DATASETS.get(repo)
    if spec is None:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(
            f"Dataset {repo!r} is not on the approved list ({known}). "
            "A hand-written hf_license is not accepted for an unknown repository."
        )
    if license_id != spec["license"]:
        raise SystemExit(
            f"hf_license {license_id!r} does not match the approved license "
            f"{spec['license']!r} for {repo}."
        )
    if revision != spec["revision"]:
        raise SystemExit(
            f"hf_revision for {repo} must be {spec['revision']}, got {revision!r}."
        )
    return spec
