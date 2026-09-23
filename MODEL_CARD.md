---
language:
  - en
license: mit
library_name: peft
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - lora
  - peft
  - tinylama
  - helloblue
  - b01-nuna
pipeline_tag: text-generation
---

# B01-NUna LoRA (workshop artifact)

**This research artifact is separate from Helloblue's production AI systems.** This card describes a **PEFT LoRA on TinyLlama 1.1B** for research, identity smoke tests, and community SFT.

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
- LoRA: r=16, alpha=32, dropout=0.05, `q/k/v/o_proj` + `gate/up/down_proj`
- Data: Helloblue-authored smoke SFT (MIT), not a web scrape

## Evaluation

Probes in `datasets/eval_probes.json` (identity, simple facts, safety). Upload to Hub requires a **generation** eval report (`mode=generation`), not `--check-only`, and the report is bound to the adapter by SHA-256 — a report from a different or retrained adapter is rejected.

Eval also scores every probe against the base model with the LoRA disabled. Treat the `comparison` block of a report as the honest summary: the required probes paraphrase training rows, so several of them pass on stock TinyLlama and only the `gained` list reflects what this adapter changed.

## Maintainers

Helloblue Inc. Official tags are maintainer-promoted only (`GOVERNANCE.md`).
