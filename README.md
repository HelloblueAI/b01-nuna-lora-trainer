# B01-NUna LoRA trainer

Public **workshop** for growing **B01-NUna weights** (TinyLlama PEFT LoRA).

**helloblue.ai production chat is not this repo.** The product uses closed Groq orchestration. Do not send product code, keys, or user logs here.

This is **not** Llama 4 / DeepSeek / Kimi. Those labs ship large open weights trained at cluster scale. This repo is a **small, honest SFT LoRA** plus a contribution path so the community can improve **the adapter**, while Helloblue Inc promotes official Hub tags after eval.

## Community

- Data and evals: `CONTRIBUTING.md`, `datasets/community/`, `GOVERNANCE.md`
- License: MIT (`LICENSE`)
- Conduct: `CODE_OF_CONDUCT.md`
- Model card (Hub copy): `MODEL_CARD.md`

## Requirements

- Linux, NVIDIA GPU + CUDA (~8GB) for **train** / **generation eval**
- Python 3.10+
- `--dry-run` and `--check-only` work without a GPU (CI)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

CPU CI / laptops:

```bash
pip install -e ".[dev]"
python -m b01_nuna_lora.train --dry-run
python -m b01_nuna_lora.eval --check-only
```

## Train (GPU)

```bash
python -m b01_nuna_lora.train \
  --config configs/default.yaml \
  --data datasets/train.json \
  --output outputs/adapter
```

Uses TRL `SFTTrainer` and the base tokenizer **chat template**. Writes `outputs/adapter/train_run.json` (command, seed, data path).

## Eval gate (GPU) then upload (opt-in)

```bash
python -m b01_nuna_lora.eval --adapter outputs/adapter --report outputs/eval_report.json
export HF_TOKEN=hf_...
python -m b01_nuna_lora.upload --adapter outputs/adapter --repo helloblueai/B01-NUna --private
```

`--private` is default. Public Hub tags are maintainer-only after generation eval. `--allow-unverified-upload` skips the gate and must not be used for public tags.

## Dataset

Bundled JSON is **Helloblue-authored MIT smoke SFT** (`datasets/README.md`). It is enough to exercise the CLI, not a pretraining mix.

## What stays closed

`HelloblueAI/B01.beta`, Groq routing, cloud GPU job orchestration, and live chats.

## License

MIT — see `LICENSE`.
