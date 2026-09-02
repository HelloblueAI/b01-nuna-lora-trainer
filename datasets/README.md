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

Official **helloblue.ai** chat does not train from this folder.
