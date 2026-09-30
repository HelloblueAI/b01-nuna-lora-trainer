"""Standard benchmarks, so a run can be compared with other open models.

The probe file answers 'did this adapter break'. GSM8K, IFEval, and MMLU answer
'where does this model sit'. Both numbers belong in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Small enough to cite, standard enough that other model cards use the same names.
DEFAULT_TASKS = ("gsm8k", "ifeval", "mmlu")


def model_args(*, base_model: str, adapter: str | None, load_in_4bit: bool) -> str:
    """lm-eval Hugging Face backend arguments for the base, or the base plus a LoRA."""
    parts = [f"pretrained={base_model}", "dtype=bfloat16"]
    if load_in_4bit:
        parts.append("load_in_4bit=True")
    if adapter:
        parts.append(f"peft={adapter}")
    return ",".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GSM8K, IFEval, and MMLU via lm-eval.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per task. Omit for a full run.",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/benchmark.json"))
    parser.add_argument(
        "--batch-size",
        default="1",
        help="lm-eval batch size. Keep at 1 on an 8GB card.",
    )
    args = parser.parse_args(argv)

    try:
        from lm_eval import simple_evaluate
    except ImportError as exc:
        raise SystemExit(
            "Benchmarks need lm-eval. Install the extra: pip install -e '.[benchmarks]'"
        ) from exc

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = simple_evaluate(
        model="hf",
        model_args=model_args(
            base_model=args.base_model,
            adapter=str(args.adapter) if args.adapter else None,
            load_in_4bit=bool(args.load_in_4bit),
        ),
        tasks=tasks,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    payload = {
        "base_model": args.base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "load_in_4bit": bool(args.load_in_4bit),
        "limit": args.limit,
        "results": results.get("results") if isinstance(results, dict) else results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["results"], indent=2, default=str))
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
