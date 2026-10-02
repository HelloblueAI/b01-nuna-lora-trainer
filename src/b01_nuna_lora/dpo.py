"""DPO preference stage on an existing QLoRA adapter. Does not upload anything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from b01_nuna_lora.preferences import load_preferences
from b01_nuna_lora.train import (
    _filter_kwargs,
    apply_4bit_training_precision,
    bitsandbytes_config_from_kwargs,
    quantization_kwargs,
    require_bitsandbytes,
    require_trainer_accepts_quantization,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DPO on an existing LoRA adapter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="SFT adapter to keep training.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    quant = quantization_kwargs(cfg)
    if quant is None:
        raise SystemExit("DPO config must set load_in_4bit: true on an 8GB card.")

    rows = load_preferences(
        str(cfg["hf_dataset"]),
        str(cfg.get("hf_split") or "train"),
        max_samples=int(cfg["hf_max_samples"]),
        license_id=str(cfg["hf_license"]),
    )
    print(f"preference rows: {len(rows)} ({cfg['hf_license']})")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from datasets import Dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DPO.")
    require_bitsandbytes()

    tokenizer = AutoTokenizer.from_pretrained(str(cfg["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template = cfg.get("chat_template")
    if template:
        tokenizer.chat_template = Path(template).read_text(encoding="utf-8")

    sft_kwargs = {
        "output_dir": str(args.output),
        "num_train_epochs": int(cfg.get("epochs", 1)),
        "per_device_train_batch_size": int(cfg.get("batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_steps": int(cfg.get("warmup_steps", 10)),
        "logging_steps": 10,
        "bf16": False,
        "fp16": True,
        "beta": float(cfg.get("beta", 0.1)),
        "max_length": int(cfg.get("max_length", 512)),
        "max_prompt_length": int(cfg.get("max_prompt_length", 256)),
        "seed": int(cfg.get("seed", 42)),
        "report_to": "none",
        "gradient_checkpointing": True,
    }
    apply_4bit_training_precision(sft_kwargs, quant, bf16_supported=torch.cuda.is_bf16_supported())
    dpo_kwargs, dropped = _filter_kwargs(DPOConfig, sft_kwargs)
    if "beta" in dropped:
        raise SystemExit("Installed TRL does not accept DPOConfig.beta.")

    base = AutoModelForCausalLM.from_pretrained(
        str(cfg["base_model"]),
        quantization_config=bitsandbytes_config_from_kwargs(quant),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": DPOConfig(**dpo_kwargs),
        "train_dataset": Dataset.from_list(rows),
        "processing_class": tokenizer,
    }
    init_kwargs, dropped_trainer = _filter_kwargs(DPOTrainer, trainer_kwargs)
    require_trainer_accepts_quantization(dropped_trainer, enabled=False)
    trainer = DPOTrainer(**init_kwargs)
    result = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    summary = {
        "status": "completed",
        "base_model": cfg["base_model"],
        "sft_adapter": str(args.adapter),
        "n_preferences": len(rows),
        "hf_dataset": cfg["hf_dataset"],
        "hf_license": cfg["hf_license"],
        "final_train_loss": getattr(result, "training_loss", None),
        "global_step": getattr(result, "global_step", None),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2) + "\n"
    (args.output / "dpo_run.json").write_text(payload, encoding="utf-8")
    print(f"Saved DPO adapter to {args.output} (loss {summary['final_train_loss']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
