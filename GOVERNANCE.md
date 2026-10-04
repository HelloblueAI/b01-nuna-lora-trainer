# Governance

**Production systems and private models** are **not** governed here and are not open for contribution.

**This repository** is the public workshop trainer (TinyLlama LoRA / future licensed bases). It is not the production NUna system at [helloblue.ai/model-card](https://helloblue.ai/model-card).

| Role | Who |
| --- | --- |
| Official Hub tags and `datasets/train.json` | Helloblue Inc maintainers |
| PRs: data, evals, trainer bugs, docs | Anyone under MIT + CONTRIBUTING |
| Production chat routing | Closed; not in this repo |

A merged PR does **not** change the production NUna system at [helloblue.ai/model-card](https://helloblue.ai/model-card). Maintainers may train, eval, and tag Hub revisions of this workshop when probes pass.

License of contributions: MIT, same as this repo's code, unless a file states otherwise and maintainers accept it.

Adapter weights follow the base model. A Qwen2.5-3B adapter is under the Qwen Research License. It is not MIT and it is not commercially usable unless Helloblue holds a separate commercial license from Alibaba. Do not tag or publicly upload that adapter with an MIT model card. See `BASE_MODEL_LICENSES.md`.
