# b01-nuna-lora-trainer

Single-GPU **PEFT LoRA** trainer for [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).

This is **research tooling**, not [helloblue.ai](https://helloblue.ai) production chat. Production NUna uses Groq orchestration. A TinyLlama adapter will not match a 70B API model.

**Do not train on live user chats** unless you have a lawful basis and the data is not in git. The bundled dataset is a curated identity seed only.

## Requirements

- Linux, NVIDIA GPU + CUDA (tested around 8GB VRAM)
- Python 3.10+
- A Hugging Face-compatible base model (default TinyLlama)

AMD / dual-GPU / Colossal / Ollama import are **not** in this repo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Train

```bash
python -m b01_nuna_lora.train \
  --config configs/default.yaml \
  --data datasets/identity-seed.json \
  --output outputs/adapter
```

Expect `adapter_model.safetensors` and `adapter_config.json` under `--output`.

## Upload (opt-in, off by default)

```bash
export HF_TOKEN=hf_...
python -m b01_nuna_lora.upload \
  --adapter outputs/adapter \
  --repo your-username/tinyllama-lora-rd \
  --private
```

`--private` is the default. Do not publish weights until you have evalled them.

## Dataset format

JSON array:

```json
[
  {
    "instruction": "Who are you?",
    "input": "",
    "output": "I'm B01, an AI assistant developed by Helloblue."
  }
]
```

From the private B01-NUna app (sibling `B01.beta` checkout):

```bash
pnpm run training:export-lora-dataset -- --out ../b01-nuna-lora-trainer/datasets/identity-seed.json
# live chats (do not commit):
pnpm run training:export-lora-dataset -- --include-live --out data/lora-trainer-export.json
```

## License

MIT — see `LICENSE`.
