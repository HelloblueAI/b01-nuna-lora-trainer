"""TinyLlama LoRA via TRL SFTTrainer + tokenizer chat template."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from b01_nuna_lora.data import load_records


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _write_run_log(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-GPU TinyLlama LoRA (TRL SFT). Not helloblue.ai production chat."
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
    records = load_records(args.data)
    seed = int(cfg.get("seed", 42))
    command = ["python", "-m", "b01_nuna_lora.train", *sys.argv[1:]]
    run_log = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(args.config),
        "data_path": str(args.data),
        "n_examples": len(records),
        "seed": seed,
        "base_model": cfg.get("base_model"),
        "dry_run": bool(args.dry_run),
        "note": (
            "Smoke/SFT scale only. Official helloblue.ai chat is Groq orchestration, not this adapter."
        ),
    }
    _write_run_log(args.output, run_log)
    print(json.dumps({k: run_log[k] for k in ("command", "n_examples", "seed", "base_model")}, indent=2))

    if args.dry_run:
        print(f"dry-run ok → {args.output / 'train_run.json'}")
        return 0

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training. Use --dry-run without a GPU.")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        "assistant_only_loss": True,
        "packing": False,
        "bf16": False,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "eval_strategy": "no",
        "load_best_model_at_end": False,
    }

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

    trainer_kwargs = {
        "model": str(cfg["base_model"]),
        "args": SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs)),
        "train_dataset": dataset,
        "peft_config": lora,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": [ProgressCallback()],
    }
    trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, trainer_kwargs))

    resume = None if cfg.get("fresh_start", True) else args.resume
    if resume:
        trainer.train(resume_from_checkpoint=str(resume))
    else:
        trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    print(f"Saved adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
