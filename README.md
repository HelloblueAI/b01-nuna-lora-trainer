# b01-nuna-lora-trainer

Single-GPU PEFT LoRA SFT for [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (chat templates, eval gate).

Measured results from real runs on an RTX 4060 — including a case where the gate
caught the fine-tune destroying a capability the base model had — are in
[`docs/EVIDENCE.md`](./docs/EVIDENCE.md).

**Looking to contribute?** See our [Good First Issues](https://github.com/HelloblueAI/b01-nuna-lora-trainer/labels/good%20first%20issue) and [Help Wanted](https://github.com/HelloblueAI/b01-nuna-lora-trainer/labels/help%20wanted) tasks. Adding eval probes and licensed SFT data needs no GPU.

## Community

- Data and evals: [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`datasets/community/`](./datasets/community/), [`GOVERNANCE.md`](./GOVERNANCE.md)
- License: MIT ([`LICENSE`](./LICENSE))
- Conduct: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)
- Model card (Hub copy): [`MODEL_CARD.md`](./MODEL_CARD.md)

## Requirements

- Linux, NVIDIA GPU + CUDA (~8GB) for **train** / **generation eval**. A 3B model in fp16 does not fit; use `configs/qwen25_3b_qlora.yaml`.
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

`--check-only` validates every probe without a GPU: unique non-empty `id`, a usable prompt, at least one matcher, and that each `any_must_match` / `must_not_match` pattern compiles as a regex. Bad probes fail in CI instead of crashing a GPU run mid-generation.

## Troubleshooting

From a real run on an RTX 4060 (8GB), loading `Qwen/Qwen2.5-3B-Instruct` in fp16 reserved 5,958 MiB and the first training forward then died with:

```
CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity of 7.60 GiB of which 62.69 MiB is free.
```

The same model with `configs/qwen25_3b_qlora.yaml` finished at 3,712 MiB reserved. That config is the fix: 4-bit base weights, batch size 1, gradient checkpointing.

## Train (GPU)

```bash
python -m b01_nuna_lora.train \
  --config configs/default.yaml \
  --data datasets/train.json \
  --output outputs/adapter
```

4-bit QLoRA (same recipe, base weights in NF4) is `configs/qlora.yaml`. It needs the extra: `pip install -e ".[qlora]"`. On this 1.1B model the memory saving is small; the path is what lets a 3B–8B base fit an 8GB card. Eval follows the adapter's `train_run.json` and loads the base in 4-bit when that is how it was trained (`--no-load-in-4bit` overrides).

Uses TRL `SFTTrainer` with **assistant-only loss**, so gradients come from assistant turns only.
That requires a chat template containing `{% generation %}` / `{% endgeneration %}`, which stock
TinyLlama does not have — with the stock template TRL raises `at least one example has no
assistant tokens`. `configs/tinyllama_chat_template.jinja` is the stock template plus those
markers; it renders byte-identically, so the prompt format is unchanged. `configs/default.yaml`
points at it, and training preflights the mask before loading the model:

```
assistant-only loss active (44 supervised tokens in example 0)
```

Set `assistant_only_loss: false` to train on the full sequence instead.

Writes
`outputs/adapter/train_run.json` with a `status` of `started` → `completed`, so a crashed run is
never mistaken for a finished one. A completed run also records duration, final train loss,
resolved package versions, GPU name, data and adapter SHA-256, and any config kwargs the installed
TRL did not accept.

That last field matters: `pyproject` allows a wide TRL range, and kwargs TRL does not recognise are
dropped. Losing `assistant_only_loss` means loss is computed over prompt tokens too, so the run now
warns instead of silently training differently.

## Eval gate (GPU) then upload (opt-in)

```bash
python -m b01_nuna_lora.eval --adapter outputs/adapter --report outputs/eval_report.json
export HF_TOKEN=hf_...
python -m b01_nuna_lora.upload --adapter outputs/adapter --repo helloblueai/B01-NUna --private
```

Eval scores every probe twice by default: once with the adapter, once with the same weights and
the LoRA switched off (`--no-baseline` to skip). The report's `comparison` block names the probes
that pass **because of** the adapter, the ones the base model already passed, and any the adapter
made worse. If no probe is in the first group, the run warns that these probes cannot tell the
LoRA apart from stock TinyLlama.

Probes are `required` (gate the upload) or advisory. The `generalization` probes ship advisory,
so they report signal without blocking a maintainer's tag until they are confirmed on a real run.

`--private` is default. Public Hub tags are maintainer-only after generation eval. `--allow-unverified-upload` skips the gate and must not be used for public tags.

The gate is bound to the adapter it evaluated. `eval` records a SHA-256 of `adapter_config.json` + `adapter_model.safetensors` in the report's `provenance`, and `upload` re-hashes `--adapter` and refuses to proceed unless they match. A stale report, or a report from a different or retrained adapter, fails the upload.

## Dataset

Bundled JSON is Helloblue-authored MIT smoke SFT ([`datasets/README.md`](./datasets/README.md)). Enough to exercise the CLI, not a pretraining mix. New examples go in [`datasets/community/`](./datasets/community/).

## License

This project is licensed under the MIT License. The full text is in [`LICENSE`](./LICENSE) at the repository root.
