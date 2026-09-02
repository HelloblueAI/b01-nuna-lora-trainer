# Contributing

This repo is how the community helps **grow B01-NUna weights**. It is **not** the helloblue.ai app.

Please read `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, and `datasets/README.md`.

## You can contribute

- Licensed SFT JSON under `datasets/community/` (your original work or a stated license)
- Eval probes with `any_must_match` / `must_not_match`
- Trainer, docs, and CI fixes
- Repro of eval failures with **synthetic** prompts

## You cannot contribute here

- Anything from **B01.beta**, Groq, or production logs
- Personal data or live user conversations
- Requests to “make this the default helloblue.ai model”
- Full pretraining dumps or unlicensed scrapes

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
python -m b01_nuna_lora.train --dry-run
python -m b01_nuna_lora.eval --check-only
```

GPU training is optional and local. Official Hub tags require a **generation** eval report.

## Pull requests

- One concern per PR
- MIT unless maintainers agree otherwise
- Do not commit `outputs/`, `.env`, or adapter weights
