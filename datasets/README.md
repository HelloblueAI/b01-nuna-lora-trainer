---
license: mit
pretty_name: B01-NUna SFT smoke set
language:
  - en
tags:
  - sft
  - chat
  - tinylama
  - helloblue
task_categories:
  - text-generation
size_categories:
  - n<1K
---

# Dataset card (this repo)

**License:** MIT. All bundled rows are authored by Helloblue Inc for this trainer.

**This is not a pretraining corpus.** `identity-seed.json` is an **install/smoke** set (~20 identity turns). `sft_extra.json` adds a few original fact and refusal turns. Together they are enough to **run the CLI**, not to match Llama/DeepSeek.

## Files

| File | Role |
| --- | --- |
| `identity-seed.json` | Smoke identity (Alpaca or converted to chat) |
| `sft_extra.json` | Extra Helloblue-authored SFT |
| `train.json` | Combined chat `messages` used by default train |
| `eval_probes.json` | Held-out prompts + matchers (not training targets) |
| `community/` | Third-party PRs (see that README) |

## Probe schema

| Field | Meaning |
| --- | --- |
| `id` | Unique, non-empty |
| `prompt` or `messages` | What gets sent to the model |
| `any_must_match` | Passes if **any** pattern matches |
| `must_not_match` | Fails if **any** pattern matches |
| `category` | `memorization` or `generalization` (free-form; summarised per category) |
| `required` | `true` gates the upload, `false` is advisory only (default `true`) |
| `note` | Why the probe exists |

Patterns are tried as regex first, then as a normalized substring. `--check-only` rejects patterns
that do not compile, so write `\\b4\\b` rather than `4` when you mean the whole number.

`memorization` probes paraphrase a training row, so a fine-tuned model is expected to pass them.
They confirm the run did not break, but they do **not** show the adapter generalizes — several
pass on stock TinyLlama, which is why eval runs a base-model control. `generalization` probes
deliberately sit outside the training distribution (an unseen capital, an unseen harm category,
over-refusal, format adherence) and ship advisory until verified on a real GPU run.

## Format

Preferred:

```json
{ "messages": [
  { "role": "user", "content": "Who are you?" },
  { "role": "assistant", "content": "I'm B01..." }
] }
```

Alpaca `instruction` / `input` / `output` is still accepted and converted.

## Do not contribute

- Live product chats, emails, or any personal data
- Scraped web dumps without a clear license
- Data that only exists to jailbreak or to attack others

Production NUna ([helloblue.ai/model-card](https://helloblue.ai/model-card)) does not train from this folder.
