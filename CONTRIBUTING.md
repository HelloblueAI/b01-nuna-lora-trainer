# Contributing

This trainer is a small CUDA + PEFT CLI. Keep it that way.

## Scope

In:

- Single-GPU LoRA on a documented Hugging Face causal LM
- JSON instruction/input/output datasets
- Optional Hub upload (private by default)

Out:

- Next.js, chat collection, auto-train from user traffic
- Dual-GPU, ROCm, Colossal AI
- Merging this LoRA into Ollama `llama3.2:*` (architecture mismatch with TinyLlama)

## Setup

Follow `README.md`. Run `python -m pytest tests/` (no GPU required for `tests/test_data.py`).

## Pull requests

- Do not commit `.env`, `outputs/`, `datasets/from-b01.json`, or adapter weights
- Do not add live conversation logs
- Keep claims accurate: this is R&D, not production NUna chat
