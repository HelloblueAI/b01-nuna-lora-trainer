"""Opt-in Hub upload of adapter files. Requires a passing eval report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from b01_nuna_lora.cards import write_adapter_card
from b01_nuna_lora.scoring import require_passing_report

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
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
        help="Skip the eval gate. Allowed only for a private repo; cannot publish publicly.",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        raise SystemExit("Pass --repo OWNER/NAME or set HF_REPO_ID")

    if args.allow_unverified_upload and not args.private:
        raise SystemExit(
            "Refusing --allow-unverified-upload with a public repo. "
            "A bypass can upload only with the default private visibility. "
            "Drop --no-private, or pass a passing --eval-report and omit the bypass."
        )

    missing = [
        name
        for name in ("adapter_config.json", "adapter_model.safetensors")
        if not (args.adapter / name).exists()
    ]
    if missing:
        raise SystemExit(f"Adapter dir missing {missing}")

    report = None
    if not args.allow_unverified_upload:
        report = require_passing_report(args.eval_report, adapter=args.adapter)

    # Generate the card before create_repo. A missing or wrong card must not publish.
    # MODEL_CARD.md in the working directory is ignored: it is the TinyLlama workshop card.
    write_adapter_card(args.adapter, report, public=not args.private)

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    allow = {name for name in ADAPTER_FILES if (args.adapter / name).exists()}
    allow.add("README.md")
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=list(allow),
        ignore_patterns=[
            "checkpoint-*",
            "*.pt",
            "optimizer.pt",
            "train_run.json",
            "dpo_run.json",
            "MODEL_CARD.md",
        ],
    )
    visibility = "private" if args.private else "public"
    print(f"Uploaded {sorted(allow)} to {args.repo} ({visibility})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
