# Community data

Helloblue Inc maintains official `datasets/train.json` and Hub tags.

To propose examples:

1. Add a new JSON file here, MIT-or-compatible licensed, **your original work** or a dataset with an explicit license in the PR.
2. Use chat `messages` (user + assistant).
3. No personal data and no production logs.
4. Open a PR. CI validates JSON. A maintainer may merge a subset after review and eval.

`general.json` in this folder is original Helloblue-authored MIT data. It is not part of `train.json`. `configs/qwen25_3b_qlora.yaml` appends it via `extra_data`.

Do not edit `train.json` in the same PR unless a maintainer asks.
