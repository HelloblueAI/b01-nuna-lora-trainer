# b01-nuna-lora-trainer

Single-GPU PEFT LoRA SFT for [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (chat templates, eval gate).

## Community

- Data and evals: [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`datasets/community/`](./datasets/community/), [`GOVERNANCE.md`](./GOVERNANCE.md)
- License: MIT ([`LICENSE`](./LICENSE))
- Conduct: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)
- Model card (Hub copy): [`MODEL_CARD.md`](./MODEL_CARD.md)

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
ruff check .
pytest
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

Bundled JSON is Helloblue-authored MIT smoke SFT ([`datasets/README.md`](./datasets/README.md)). Enough to exercise the CLI, not a pretraining mix. New examples go in [`datasets/community/`](./datasets/community/).

## License

This project is licensed under the MIT License. The full text is in [`LICENSE`](./LICENSE) at the repository root.
