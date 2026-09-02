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

**This is not helloblue.ai production chat.** The product at [helloblue.ai](https://helloblue.ai) uses **Groq** and closed orchestration. This card describes a **PEFT LoRA on TinyLlama 1.1B** for research, identity smoke tests, and community SFT.

## Intended use

- Learn / reproduce a **small** chat LoRA on a single NVIDIA GPU
- Community PRs of **licensed** SFT data and eval probes
- Optional private Hub upload after the eval gate

## Out of scope

- Matching Llama 4, DeepSeek, Kimi, or Groq 70B quality
- Serving as the default NUna for helloblue.ai
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

- Method: TRL `SFTTrainer` + tokenizer `chat_template`
- LoRA: r=8, alpha=16, dropout=0.1, `q/k/v/o_proj`
- Data: Helloblue-authored smoke SFT (MIT), not a web scrape

## Evaluation

Held-out probes in `datasets/eval_probes.json` (identity, simple facts, safety). Upload to Hub requires a **generation** eval report (`mode=generation`), not `--check-only`.

## Maintainers

Helloblue Inc. Official tags are maintainer-promoted only (`GOVERNANCE.md`).
