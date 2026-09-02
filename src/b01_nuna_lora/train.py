"""Train a PEFT LoRA adapter on a single CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from b01_nuna_lora.data import format_prompt, load_records


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-GPU TinyLlama LoRA trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint dir (needs torch>=2.6)")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    records = load_records(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. CPU training is not supported in this CLI.")

    device = torch.device("cuda:0")
    base_model = str(cfg["base_model"])
    print(f"Base model: {base_model}")
    print(f"Examples: {len(records)}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("LoRA produced no trainable parameters")
    print(f"Trainable parameters: {trainable:,}")

    dataset = Dataset.from_list(records).map(lambda ex: {"text": format_prompt(ex)})
    max_length = int(cfg["max_length"])

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=int(cfg["epochs"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_steps=int(cfg["warmup_steps"]),
        logging_steps=10,
        save_steps=int(cfg["save_steps"]),
        eval_strategy="no",
        fp16=bool(cfg.get("fp16", True)),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none",
        save_total_limit=3,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if not logs:
                return
            payload = {
                "epoch": state.epoch,
                "step": state.global_step,
                "loss": logs.get("loss", 0),
                "learning_rate": logs.get("learning_rate", 0),
            }
            print(f"PROGRESS:{json.dumps(payload)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback()],
    )

    resume = args.resume
    if cfg.get("fresh_start", True):
        resume = None
        print("fresh_start=true: not resuming checkpoints")
    elif resume is None:
        print("Starting without --resume")

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
