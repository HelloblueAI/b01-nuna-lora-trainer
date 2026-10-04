---
language:
  - en
license: mit
library_name: peft
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - lora
  - peft
  - tinyllama
  - helloblue
  - b01-nuna
pipeline_tag: text-generation
---

# B01-NUna LoRA (workshop artifact)

**This research artifact is not the production NUna system.** Production NUna has its own model card at [helloblue.ai/model-card](https://helloblue.ai/model-card). This card describes the **TinyLlama PEFT LoRA workshop**. It is not the card for a Qwen adapter. Qwen2.5-3B-Instruct is under the Qwen Research License (`license: other`), not MIT, and is not commercially usable without a separate license. See `BASE_MODEL_LICENSES.md`. Upload generates a card from the adapter's recorded base model and will not attach this file to a Qwen adapter.

## Intended use

- Learn / reproduce a **small** chat LoRA on a single NVIDIA GPU
- Community PRs of **licensed** SFT data and eval probes
- Optional private Hub upload after the eval gate

## Out of scope

- Matching Llama 4, DeepSeek, Kimi, or other frontier-class quality
- Serving as a production assistant
- Training on product user logs
- Merging this adapter onto Ollama `llama3.2:*` (architecture mismatch)

## How to load (after a gated upload)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "helloblueai/B01-NUna"  # when published

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER)
```

## Training

See the GitHub workshop: [HelloblueAI/b01-nuna-lora-trainer](https://github.com/HelloblueAI/b01-nuna-lora-trainer).

- Method: TRL `SFTTrainer` + patched `chat_template` with `{% generation %}` markers (assistant-only loss)
- Default: TinyLlama-1.1B, 30 epochs, LoRA r=16, alpha=32, dropout=0.05, targets `q/k/v/o_proj` and `gate/up/down_proj`
- Optional: Qwen2.5-3B 4-bit QLoRA, including a 2,000-row MIT UltraChat mix (`configs/qwen25_3b_open_sft.yaml`)
- Data: Helloblue-authored smoke SFT (MIT). The UltraChat mix is separate and MIT-licensed. Not a web scrape of product logs.

## Evaluation

`datasets/eval_probes.json` has 17 probes: 11 required and 6 advisory. Upload to Hub requires a **generation** eval report (`mode=generation`), not `--check-only`, and the report is bound to the adapter by SHA-256. A report from a different or retrained adapter is rejected. `--allow-unverified-upload` can create only a private repo.

Eval scores every probe against the base model with the LoRA disabled. `--fail-on-regression` fails the run if the adapter loses a probe the base passed. GSM8K, IFEval, and MMLU are a separate command (`python -m b01_nuna_lora.benchmark`) and are not a substitute for the probe gate.

The 2026-09-30 UltraChat adapter (loss 1.074) failed that gate: 7/11 required, and it regressed `gen-refusal-phishing`. It is not approved for a public Hub release. See `docs/EVIDENCE.md`.

## Maintainers

Helloblue Inc. Official tags are maintainer-promoted only (`GOVERNANCE.md`).
