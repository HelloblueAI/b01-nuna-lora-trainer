# Community data

Helloblue Inc maintains official `datasets/train.json` and Hub tags.

To propose examples:

1. Add a new JSON file here, MIT-or-compatible licensed, **your original work** or a dataset with an explicit license in the PR.
2. Use chat `messages` (user + assistant).
3. No personal data and no production logs.
4. Open a PR. CI validates JSON. A maintainer may merge a subset after review and eval.

This folder is **data license**, not **needs GPU**. State the license in the PR. A laptop can prepare the file. Scoring it with generation eval waits until a GPU is confirmed.

`general.json`, `instruction_constraints.json`, `identity_frames.json`, and `identity_jailbreaks.json` in this folder are original Helloblue-authored MIT data. They are not part of `train.json`. `configs/qwen25_3b_qlora.yaml` appends `general.json` via `extra_data`. `configs/qwen25_3b_repair_sft.yaml` also appends the constraint, identity-frame, and identity-refusal files, and repeats the identity files more often than the others.

Do not edit `train.json` in the same PR unless a maintainer asks.
