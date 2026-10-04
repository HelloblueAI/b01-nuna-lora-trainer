# Unsupported on an 8GB GPU

These recipes were tried on an RTX 4060 (8GB). Qwen2.5-7B in 4-bit did not fit: bitsandbytes refused a CPU offload. They are kept so the attempt is reproducible. They are not the recommended path.

Use `configs/qwen25_3b_post_sft.yaml` and `configs/qwen25_3b_post_dpo.yaml` instead.
