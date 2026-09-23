"""TinyLlama LoRA via TRL SFTTrainer + tokenizer chat template."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from b01_nuna_lora.data import load_records
from b01_nuna_lora.scoring import fingerprint_adapter, fingerprint_file

TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")

# Dropping these silently changes what training does, so say so out loud.
CRITICAL_SFT_KWARGS = ("assistant_only_loss", "seed", "learning_rate", "num_train_epochs")

# TRL renamed this; exactly one survives _filter_kwargs, but losing both is fatal.
EITHER_OR_SFT_KWARGS = (("max_length", "max_seq_length"),)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only kwargs the installed version accepts, and report what was dropped."""
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    kept = {key: value for key, value in kwargs.items() if key in params}
    return kept, sorted(set(kwargs) - set(kept))


def _check_dropped_sft_kwargs(dropped: list[str]) -> list[str]:
    """Turn silent API drift into visible warnings before a run burns GPU hours."""
    warnings: list[str] = []
    for name in CRITICAL_SFT_KWARGS:
        if name not in dropped:
            continue
        if name == "assistant_only_loss":
            warnings.append(
                "installed TRL does not accept 'assistant_only_loss': loss will include prompt "
                "tokens, not just assistant turns. Upgrade TRL to train as documented."
            )
        else:
            warnings.append(f"installed TRL does not accept '{name}': the config value is ignored.")
    for pair in EITHER_OR_SFT_KWARGS:
        if all(name in dropped for name in pair):
            raise SystemExit(
                f"Installed TRL accepts none of {pair}; sequence length would be a silent default. "
                "Upgrade or pin TRL."
            )
    return warnings


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def count_supervised_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """How many tokens assistant_only_loss would actually train on for this example."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    return sum(encoded.get("assistant_masks") or [])


def check_assistant_masks(tokenizer: Any, records: list[dict[str, Any]]) -> int:
    """assistant_only_loss needs {% generation %} markers or it supervises nothing.

    Stock TinyLlama has no such markers, so this fails fast with a fix instead of either
    a RuntimeError deep inside TRL or a run that quietly trains on the whole sequence.
    """
    supervised = count_supervised_tokens(tokenizer, records[0]["messages"])
    if supervised:
        return supervised
    raise SystemExit(
        "assistant_only_loss is enabled but the chat template marks no assistant tokens, "
        "so training would have nothing to learn from.\n"
        "The template needs {% generation %} / {% endgeneration %} around assistant content.\n"
        "For TinyLlama, set in your config:\n"
        "    chat_template: configs/tinyllama_chat_template.jinja\n"
        "Or set 'assistant_only_loss: false' to train on the full sequence instead."
    )


def _try_fingerprint(adapter: Path) -> str | None:
    """Never lose a finished run's log because the adapter dir looks unexpected."""
    try:
        return fingerprint_adapter(adapter)
    except SystemExit:
        return None


def quantization_kwargs(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """BitsAndBytesConfig kwargs for QLoRA, or None when 4-bit is off.

    Separated from model loading so the config surface is covered without a GPU.
    nf4 + double quant is the standard QLoRA setup; compute dtype stays fp16 to
    match the rest of the recipe on consumer cards without bf16.
    """
    if not cfg.get("load_in_4bit"):
        return None

    quant_type = str(cfg.get("bnb_4bit_quant_type", "nf4"))
    if quant_type not in ("nf4", "fp4"):
        raise SystemExit(f"bnb_4bit_quant_type must be 'nf4' or 'fp4', got {quant_type!r}")

    compute_dtype = str(cfg.get("bnb_4bit_compute_dtype", "float16"))
    if compute_dtype not in ("float16", "bfloat16", "float32"):
        raise SystemExit(
            "bnb_4bit_compute_dtype must be float16, bfloat16 or float32, "
            f"got {compute_dtype!r}"
        )

    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": quant_type,
        "bnb_4bit_use_double_quant": bool(cfg.get("bnb_4bit_use_double_quant", True)),
        "bnb_4bit_compute_dtype": compute_dtype,
    }


def apply_4bit_training_precision(
    sft_kwargs: dict[str, Any], quant: dict[str, Any] | None, *, bf16_supported: bool
) -> str:
    """Match the trainer dtype to how TRL stores QLoRA adapter weights.

    Installed TRL casts trainable QLoRA params to bfloat16. Leaving fp16 on would
    train those weights under the wrong autocast. Returns the precision name
    recorded in the run log.
    """
    if quant is None:
        return "bf16" if sft_kwargs.get("bf16") else "fp16"
    if not bf16_supported:
        raise SystemExit(
            "load_in_4bit needs a GPU with bfloat16. This TRL casts QLoRA adapter "
            "weights to bf16, so fp16 training would not match them."
        )
    sft_kwargs["fp16"] = False
    sft_kwargs["bf16"] = True
    quant["bnb_4bit_compute_dtype"] = "bfloat16"
    return "bf16"


def require_trainer_accepts_quantization(dropped: list[str], *, enabled: bool) -> None:
    """4-bit that the installed TRL ignores would silently train in full precision."""
    if enabled and "quantization_config" in dropped:
        raise SystemExit(
            "load_in_4bit is set, but the installed TRL does not accept "
            "quantization_config on SFTTrainer. Training would silently stay in "
            "full precision. Upgrade TRL, or set load_in_4bit: false."
        )


def require_bitsandbytes() -> None:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "load_in_4bit needs bitsandbytes. Install the extra: pip install -e '.[qlora]'"
        ) from exc


def bitsandbytes_config_from_kwargs(quant: dict[str, Any]) -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
    )


def recorded_quantization(adapter: str | Path) -> dict[str, Any] | None:
    """Quantization settings from the adapter's train_run.json, if it was 4-bit."""
    path = Path(adapter) / "train_run.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    quant = payload.get("quantization")
    if isinstance(quant, dict) and quant.get("load_in_4bit"):
        return quant
    if payload.get("load_in_4bit"):
        return quantization_kwargs({"load_in_4bit": True})
    return None


def _vram_stats(torch: Any) -> dict[str, float | None]:
    """Peak VRAM for the run, in MiB.

    Reported because it is the number that decides whether a recipe fits a given
    card. Reserved rather than allocated is the one that has to fit: it is what
    the caching allocator held from the driver.
    """
    try:
        return {
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "total_vram_mib": round(
                torch.cuda.get_device_properties(0).total_memory / 2**20, 1
            ),
        }
    except Exception:
        # A finished run's log must survive an allocator API that moved.
        return {
            "peak_vram_mib": None,
            "peak_vram_reserved_mib": None,
            "total_vram_mib": None,
        }


def _write_run_log(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-GPU TinyLlama PEFT LoRA (TRL SFT)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data", type=Path, default=Path("datasets/train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and write train_run.json without CUDA",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    # Reject a bad 4-bit config before the dry-run return, so CI catches it.
    quantization_kwargs(cfg)
    records = load_records(args.data)
    seed = int(cfg.get("seed", 42))
    command = ["python", "-m", "b01_nuna_lora.train", *sys.argv[1:]]
    run_log = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(args.config),
        "config": cfg,
        "data_path": str(args.data),
        "data_sha256": fingerprint_file(args.data),
        "n_examples": len(records),
        "seed": seed,
        "base_model": cfg.get("base_model"),
        "dry_run": bool(args.dry_run),
        "packages": _package_versions(),
        "note": "Smoke/SFT scale. Promote Hub tags only after generation eval.",
    }
    _write_run_log(args.output, run_log)
    print(
        json.dumps(
            {k: run_log[k] for k in ("command", "n_examples", "seed", "base_model")},
            indent=2,
        )
    )

    if args.dry_run:
        run_log["status"] = "dry-run"
        _write_run_log(args.output, run_log)
        print(f"dry-run ok → {args.output / 'train_run.json'}")
        return 0

    import torch
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training. Use --dry-run without a GPU.")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_path = cfg.get("chat_template")
    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            raise SystemExit(f"chat_template not found: {template_file}")
        tokenizer.chat_template = template_file.read_text(encoding="utf-8")
        run_log["chat_template_path"] = str(template_file)
        run_log["chat_template_sha256"] = fingerprint_file(template_file)

    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("Tokenizer has no chat_template; pick a chat base model.")

    dataset = Dataset.from_list(records)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        bias="none",
    )

    sft_kwargs = {
        "output_dir": str(args.output),
        "num_train_epochs": int(cfg["epochs"]),
        "per_device_train_batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "logging_steps": 10,
        "save_steps": int(cfg["save_steps"]),
        "fp16": bool(cfg.get("fp16", True)),
        "seed": seed,
        "data_seed": seed,
        "report_to": "none",
        "save_total_limit": 3,
        "max_length": int(cfg["max_length"]),
        "max_seq_length": int(cfg["max_length"]),
        "dataset_text_field": None,
        "assistant_only_loss": bool(cfg.get("assistant_only_loss", True)),
        "packing": False,
        "bf16": False,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "eval_strategy": "no",
        "load_best_model_at_end": False,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
    }

    quant = quantization_kwargs(cfg)
    precision = apply_4bit_training_precision(
        sft_kwargs, quant, bf16_supported=torch.cuda.is_bf16_supported()
    )
    run_log["load_in_4bit"] = quant is not None
    run_log["quantization"] = quant
    run_log["train_precision"] = precision
    if quant is not None:
        require_bitsandbytes()
        print(
            "4-bit QLoRA: "
            f"{quant['bnb_4bit_quant_type']}, "
            f"double_quant={quant['bnb_4bit_use_double_quant']}, "
            f"compute={quant['bnb_4bit_compute_dtype']}"
        )

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: A002
            if not logs:
                return
            print(
                "PROGRESS:"
                + json.dumps(
                    {
                        "epoch": state.epoch,
                        "step": state.global_step,
                        "loss": logs.get("loss", 0),
                        "learning_rate": logs.get("learning_rate", 0),
                    }
                )
            )

    sft_config_kwargs, dropped_sft = _filter_kwargs(SFTConfig, sft_kwargs)
    for warning in _check_dropped_sft_kwargs(dropped_sft):
        print(f"WARNING: {warning}")

    if sft_config_kwargs.get("assistant_only_loss"):
        supervised = check_assistant_masks(tokenizer, records)
        run_log["supervised_tokens_first_example"] = supervised
        print(f"assistant-only loss active ({supervised} supervised tokens in example 0)")

    trainer_kwargs = {
        "model": str(cfg["base_model"]),
        "args": SFTConfig(**sft_config_kwargs),
        "train_dataset": dataset,
        "peft_config": lora,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": [ProgressCallback()],
    }
    if quant is not None:
        trainer_kwargs["quantization_config"] = bitsandbytes_config_from_kwargs(quant)
    trainer_init_kwargs, dropped_trainer = _filter_kwargs(SFTTrainer, trainer_kwargs)
    require_trainer_accepts_quantization(dropped_trainer, enabled=quant is not None)
    trainer = SFTTrainer(**trainer_init_kwargs)

    run_log["dropped_sft_config_kwargs"] = dropped_sft
    run_log["dropped_trainer_kwargs"] = dropped_trainer
    run_log["gpu"] = torch.cuda.get_device_name(0)
    _write_run_log(args.output, run_log)

    resume = None if cfg.get("fresh_start", True) else args.resume
    # Peak VRAM is the number that decides whether this recipe fits a given card,
    # so measure it rather than leaving contributors to guess from the model size.
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if resume:
        result = trainer.train(resume_from_checkpoint=str(resume))
    else:
        result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    duration = round(time.monotonic() - started, 2)
    metrics = getattr(result, "metrics", None) or {}
    run_log.update(
        {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "final_train_loss": getattr(result, "training_loss", None),
            "global_step": getattr(result, "global_step", None),
            "adapter_sha256": _try_fingerprint(args.output),
            "train_samples_per_second": metrics.get("train_samples_per_second"),
            **_vram_stats(torch),
        }
    )
    _write_run_log(args.output, run_log)
    print(
        f"Saved adapter to {args.output} (loss {run_log['final_train_loss']}, "
        f"peak VRAM {run_log['peak_vram_reserved_mib']} MiB / "
        f"{run_log['total_vram_mib']} MiB, {duration}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
