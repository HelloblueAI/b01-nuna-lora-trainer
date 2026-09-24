# Contributing

This repo is how the community helps the **open-source workshop trainer**. It is **not** the production NUna system. That system is documented at [helloblue.ai/model-card](https://helloblue.ai/model-card), and contributions here do not change it.

Please read `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, and `datasets/README.md`.

## You can contribute

- Licensed SFT JSON under `datasets/community/` (your original work or a stated license)
- Eval probes with `any_must_match` / `must_not_match` (patterns are regex first, falling back to
  normalized substring; run `python -m b01_nuna_lora.eval --check-only --probes <file>` before you
  open the PR, since CI rejects patterns that do not compile). See the schema in
  `datasets/README.md`. **`generalization` probes are the most useful contribution** — probes that
  only paraphrase a training row mostly prove the model memorized it. Open new probes with
  `"required": false` so maintainers can verify them on a GPU before they gate uploads.
- Trainer, docs, and CI fixes
- Repro of eval failures with **synthetic** prompts

## You cannot contribute here

- Anything from production systems, private models, or production logs
- Personal data or live user conversations
- Requests to “make this a production model”
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
