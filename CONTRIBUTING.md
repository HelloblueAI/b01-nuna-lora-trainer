# Contributing

This repo is how the community helps the **open-source workshop trainer**. It is **not** the production NUna system. That system is documented at [helloblue.ai/model-card](https://helloblue.ai/model-card), and contributions here do not change it.

Please read `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, and `datasets/README.md`.

A merged pull request is credited in the Contributors section of `README.md`: your GitHub handle and a short line on what you added.

## GPU and data licensing

University of Washington GPU access is unconfirmed. Do not start a task marked **needs GPU** until a maintainer says a machine is available. `--dry-run` and `--check-only` run on a laptop and in CI.

| Mark | Meaning |
| --- | --- |
| **needs GPU** | Training, DPO, generation eval, or GSM8K / IFEval / MMLU. The contribution is unfinished without a CUDA run. |
| **no GPU** | Docs, tests, probe files, and community JSON. CI can review the PR. |
| **data license** | New or changed training rows, a Hub corpus, or a base model. Name the license in the PR. See `BASE_MODEL_LICENSES.md`. |

Qwen2.5-3B-Instruct is the Qwen Research License, not MIT. A new corpus is accepted only when its repository and revision are on the allowlist in `src/b01_nuna_lora/licenses.py`, and the config's `hf_license` matches that entry. Original Helloblue JSON under `datasets/` is MIT.

Issues use the labels `needs-gpu` and `data-license`. A good first issue is **no GPU** unless the title says otherwise.

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

GPU training is optional for a docs or data PR and required before a maintainer promotes a probe to `"required": true` or tags a Hub revision. That verification is **needs GPU**, and it waits while UW GPU access is unconfirmed. Official Hub tags require a **generation** eval report.

## Pull requests

- One concern per PR
- MIT unless maintainers agree otherwise
- Do not commit `outputs/`, `.env`, or adapter weights
