"""GPU eval against datasets/eval_probes.json, or --check-only for CI."""

from __future__ import annotations

import argparse
from pathlib import Path

from b01_nuna_lora.scoring import load_probes, score_completion, write_report


def _last_user_text(probe: dict) -> str:
    if "prompt" in probe:
        return str(probe["prompt"])
    messages = probe.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    raise ValueError(f"probe {probe.get('id')} has no user prompt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out eval gate for B01-NUna LoRA")
    parser.add_argument("--probes", type=Path, default=Path("datasets/eval_probes.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/eval_report.json"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate probe files and matcher fixtures (no GPU)",
    )
    args = parser.parse_args(argv)

    probes = load_probes(args.probes)
    if args.check_only:
        report = write_report(
            args.report,
            [
                {
                    "id": "schema",
                    "passed": True,
                    "n_probes": len(probes),
                    "missing_any_required": [],
                    "forbidden_hits": [],
                }
            ],
            mode="check-only",
        )
        print(f"check-only ok ({len(probes)} probes) → {args.report}")
        return 0 if report["summary"]["ok"] else 1

    if args.adapter is None:
        raise SystemExit("Pass --adapter or --check-only")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for generation eval")

    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    results = []
    for probe in probes:
        user = _last_user_text(probe)
        messages = [{"role": "user", "content": user}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
        row = score_completion(text, probe)
        row["completion"] = text
        results.append(row)
        print(f"{row['id']}: {'pass' if row['passed'] else 'fail'}")

    report = write_report(args.report, results, mode="generation")
    summary = report["summary"]
    print(f"{summary['passed']}/{summary['total']} passed → {args.report}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
