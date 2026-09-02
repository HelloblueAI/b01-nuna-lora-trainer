"""Opt-in Hub upload of adapter files. Requires a passing eval report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from b01_nuna_lora.scoring import require_passing_report

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "MODEL_CARD.md",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload LoRA adapter files to Hugging Face Hub")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO_ID", ""))
    parser.add_argument("--eval-report", type=Path, default=Path("outputs/eval_report.json"))
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-unverified-upload",
        action="store_true",
        help="Skip eval gate (discouraged; never use for a public Hub tag)",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        raise SystemExit("Pass --repo OWNER/NAME or set HF_REPO_ID")

    if not args.allow_unverified_upload:
        require_passing_report(args.eval_report)

    missing = [
        name
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if not (args.adapter / name).exists()
    ]
    if missing:
        raise SystemExit(f"Adapter dir missing {missing}")

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    card = Path("MODEL_CARD.md")
    if card.exists():
        (args.adapter / "README.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")

    allow = {name for name in ADAPTER_FILES if (args.adapter / name).exists()}
    if (args.adapter / "README.md").exists():
        allow.add("README.md")
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=list(allow),
        ignore_patterns=["checkpoint-*", "*.pt", "optimizer.pt", "train_run.json"],
    )
    visibility = "private" if args.private else "public"
    print(f"Uploaded {sorted(allow)} to {args.repo} ({visibility})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
