# Base-model and dataset licenses

Helloblue-authored code in this repository, and the bundled files under `datasets/`, are MIT. See `LICENSE`.

An adapter is a derivative of the base model it was trained on. Its license and permitted use follow that base model. Do not describe a Qwen2.5-3B adapter as MIT, and do not treat it as commercially usable.

| Piece | License | Commercial use |
| --- | --- | --- |
| This repository's code and bundled JSON datasets | MIT | Yes, under MIT |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache 2.0 | Yes, under Apache 2.0 |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen Research License | No, unless Helloblue has a separate commercial license from Alibaba |
| `Qwen/Qwen2.5-7B-Instruct` | Apache 2.0 (the `LICENSE` file in that Hub repository) | The published 7B license is Apache 2.0. These configs do not fit an 8GB card and are unsupported here |
| `HuggingFaceH4/ultrachat_200k` | MIT | Yes, under MIT |
| `HuggingFaceH4/ultrafeedback_binarized` | MIT | Yes, under MIT |

`Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` ships a file titled **Qwen RESEARCH LICENSE AGREEMENT** (release date September 19, 2024). The Hub card marks that model as `license: other`. It is research and non-commercial unless a separate commercial license is obtained. Distributed derivatives of that model need the Qwen license, attribution, and the wording **Built with Qwen** or **Improved using Qwen**.

`Qwen/Qwen2.5-7B-Instruct` is a different license. Its Hub `LICENSE` file is Apache 2.0. Do not copy that conclusion onto the 3B model.

Approved repository ids and commit SHAs live in `src/b01_nuna_lora/licenses.py`. Training refuses a dataset that is not on that list, even if the config says `hf_license: mit`.

The root `MODEL_CARD.md` describes the TinyLlama workshop. Upload does not copy that file. A Qwen adapter gets a card generated from its recorded base model, with `license: other` and the Qwen attribution.
