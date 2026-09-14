"""GPU eval against datasets/eval_probes.json, or --check-only for CI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from b01_nuna_lora.scoring import (
    fingerprint_adapter,
    fingerprint_file,
    last_user_text,
    load_probes,
    score_completion,
    write_report,
)


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
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also score the base model with the adapter disabled, as a control",
    )
    args = parser.parse_args(argv)

    try:
        probes = load_probes(args.probes)
    except ValueError as exc:
        print(str(exc))
        return 1

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
            provenance={
                "probes_path": str(args.probes),
                "probes_sha256": fingerprint_file(args.probes),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"check-only ok ({len(probes)} probes validated) → {args.report}")
        return 0 if report["summary"]["ok"] else 1

    if args.adapter is None:
        raise SystemExit("Pass --adapter or --check-only")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for generation eval")

    # Fingerprint before generating so a malformed adapter dir fails fast.
    provenance = {
        "adapter_path": str(args.adapter),
        "adapter_sha256": fingerprint_adapter(args.adapter),
        "base_model": args.base_model,
        "probes_path": str(args.probes),
        "probes_sha256": fingerprint_file(args.probes),
        "max_new_tokens": args.max_new_tokens,
        "baseline": bool(args.baseline),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

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

    def generate(prompt: str) -> str:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
        return tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)

    results: list[dict] = []
    baseline_results: list[dict] | None = [] if args.baseline else None
    for probe in probes:
        user = last_user_text(probe)

        text = generate(user)
        row = score_completion(text, probe)
        row["completion"] = text
        results.append(row)

        line = f"{row['id']}: {'pass' if row['passed'] else 'fail'}"
        if args.baseline:
            # Same weights, LoRA switched off: isolates what the adapter changed.
            with model.disable_adapter():
                base_text = generate(user)
            base_row = score_completion(base_text, probe)
            base_row["completion"] = base_text
            baseline_results.append(base_row)
            line += f" (base: {'pass' if base_row['passed'] else 'fail'})"
        if not row["required"]:
            line += " [advisory]"
        print(line)

    report = write_report(
        args.report,
        results,
        mode="generation",
        provenance=provenance,
        baseline=baseline_results,
    )
    summary = report["summary"]
    print(
        f"{summary['required_passed']}/{summary['required_total']} required probes passed"
        f" ({summary['advisory_passed']}/{summary['advisory_total']} advisory) → {args.report}"
    )

    comparison = report.get("comparison")
    if comparison:
        base_summary = report["baseline"]["summary"]
        print(f"base model alone: {base_summary['passed']}/{base_summary['total']} probes")
        if comparison["regressed"]:
            print(
                "WARNING: adapter regressed probes the base model passed: "
                f"{comparison['regressed']}"
            )
        if comparison["gained"]:
            print(f"adapter accounts for: {comparison['gained']}")
        else:
            print(
                "WARNING: no probe passes because of the adapter. "
                "These probes cannot distinguish the LoRA from the base model."
            )

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
