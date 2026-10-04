"""Model card for one adapter, from the base model recorded in that adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from b01_nuna_lora.licenses import BASE_MODELS
from b01_nuna_lora.scoring import fingerprint_adapter

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def base_model_name(adapter: Path) -> str:
    config_path = adapter / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {config_path}: {exc}") from exc
    name = str(config.get("base_model_name_or_path") or "")
    if not name:
        raise SystemExit(
            "Adapter config has no base_model_name_or_path. "
            "Refusing to upload without a recorded base model."
        )
    return name


def _training_record(adapter: Path) -> dict[str, Any] | None:
    for name in ("dpo_run.json", "train_run.json"):
        path = adapter / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def render_adapter_card(adapter: Path, report: dict[str, Any] | None, *, public: bool) -> str:
    """Build the Hub README. Public uploads fail closed when provenance is incomplete."""
    base = base_model_name(adapter)
    spec = BASE_MODELS.get(base)
    if spec is None:
        known = ", ".join(sorted(BASE_MODELS))
        raise SystemExit(
            f"No model card for base model {base!r}. Approved bases: {known}. "
            "Refusing to attach a different card."
        )
    if public and base.startswith("Qwen/") and spec["card_license"] == "mit":
        raise SystemExit("Refusing to publish a Qwen adapter with an MIT card.")

    training = _training_record(adapter)
    if public and training is None:
        raise SystemExit(
            "Public upload requires train_run.json or dpo_run.json in the adapter directory. "
            "Refusing to publish without the training configuration."
        )
    if public and not report:
        raise SystemExit("Public upload requires an evaluation report.")
    if public:
        comparison = (report or {}).get("comparison") or {}
        if "regressed" not in comparison:
            raise SystemExit(
                "Public upload requires an adapter-versus-base comparison in the eval report."
            )
        if comparison.get("regressed"):
            raise SystemExit(
                "Public upload refused because the adapter regressed probes the base model "
                f"passed: {comparison['regressed']}"
            )

    sha = fingerprint_adapter(adapter)
    summary = (report or {}).get("summary") or {}
    provenance = (report or {}).get("provenance") or {}
    comparison = (report or {}).get("comparison") or {}
    license_line = f"license: {spec['card_license']}"
    extra_license = ""
    if spec["card_license"] == "other":
        extra_license = (
            f"license_name: {spec['license_name']}\n"
            f"license_link: {spec['license_link']}\n"
        )
    attribution = spec.get("attribution") or ""
    attribution_line = f"{attribution}\n\n" if attribution else ""
    commercial = (
        "This adapter is research and non-commercial only. A separate commercial "
        "license from Alibaba is required before commercial use. It is not MIT."
        if not spec["commercial"]
        else "Permitted use follows the base-model license named above. Trainer code is MIT."
    )
    training_block = "Training configuration was not recorded in this directory."
    if training is not None:
        shown = {
            key: training.get(key)
            for key in (
                "status",
                "base_model",
                "base_model_revision",
                "config",
                "hf_dataset",
                "hf_revision",
                "hf_license",
                "n_examples",
                "n_preferences",
                "final_train_loss",
                "global_step",
                "duration_seconds",
                "peak_vram_mib",
                "peak_vram_reserved_mib",
                "adapter_sha256",
            )
            if key in training
        }
        training_block = "```json\n" + json.dumps(shown, indent=2, default=str) + "\n```"
    eval_block = "No evaluation report was attached."
    if report:
        eval_block = "```json\n" + json.dumps(
            {
                "adapter_sha256": sha,
                "report_adapter_sha256": provenance.get("adapter_sha256"),
                "summary": summary,
                "gained": comparison.get("gained"),
                "regressed": comparison.get("regressed"),
            },
            indent=2,
        ) + "\n```"

    if base.startswith("Qwen/") and "license: mit" in license_line:
        raise SystemExit("Refusing to write a Qwen card with license: mit.")

    return f"""---
language:
  - en
{license_line}
{extra_license}library_name: peft
base_model: {base}
tags:
  - lora
  - peft
  - helloblue
  - b01-nuna
pipeline_tag: text-generation
---

# Adapter for {base}

{attribution_line}{commercial}

This file was generated for this adapter from its recorded base model.

- Base model: `{base}`
- Base-model revision: `{spec["revision"]}`
- Base-model license: `{spec["license"]}`
- Adapter SHA-256: `{sha}`

## Evaluation

{eval_block}

## Training

{training_block}

## How to load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base}")
model = PeftModel.from_pretrained(base, "YOUR_ADAPTER_REPO")
```
"""


def write_adapter_card(adapter: Path, report: dict[str, Any] | None, *, public: bool) -> Path:
    text = render_adapter_card(adapter, report, public=public)
    if TINYLLAMA in text and base_model_name(adapter).startswith("Qwen/"):
        raise SystemExit("Refusing to publish a Qwen adapter with the TinyLlama card.")
    if base_model_name(adapter).startswith("Qwen/") and "license: mit" in text:
        raise SystemExit("Refusing to publish a Qwen adapter with an MIT card.")
    path = adapter / "README.md"
    path.write_text(text, encoding="utf-8")
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        raise SystemExit("Model card was not written. Public upload must fail closed.")
    return path
