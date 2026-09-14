# Changelog

## Unreleased

### Fixed

- **Training was broken with the documented config.** `assistant_only_loss: true` requires a chat
  template with `{% generation %}` / `{% endgeneration %}` markers, and stock TinyLlama has none.
  On TRL >= 0.17 the documented `train` command failed with `at least one example has no assistant
  tokens`; on older TRL the kwarg was silently dropped and the model trained on prompt tokens too.
  Added `configs/tinyllama_chat_template.jinja` (stock template plus markers, byte-identical
  rendering, verified in tests) and wired it in via a new `chat_template` config key.
- Training preflights the assistant mask and fails with a fix-it message instead of a RuntimeError
  from inside TRL dataset prep.
- Raised the TRL floor to `>=0.17.0`, the release that introduced `assistant_only_loss`.

- Upload gate is now bound to the adapter: eval reports record a `provenance` block
  (adapter SHA-256, base model, probe file SHA-256, timestamp) and `upload` verifies it against
  `--adapter`. Stale reports, reports from another adapter, and adapters retrained after eval are
  rejected. Reports written before this change have no provenance and are refused.
- `eval --check-only` validates probes instead of only parsing JSON: unique non-empty ids, a usable
  prompt, at least one matcher, and every pattern compiles as a regex. Previously a probe file with
  an invalid pattern passed CI and then crashed the GPU eval with `re.error`.
- Matchers fall back to literal matching on an uncompilable pattern, so a generation run cannot die
  mid-eval.
- `eval` runs a base-model control by default (`--no-baseline` to skip), scoring each probe with the
  LoRA enabled and disabled. The report gains `baseline` and `comparison` blocks, and the run warns
  when no probe passes because of the adapter.
- Probes carry `category` and `required`. The gate counts required probes only, so advisory probes
  can report signal without blocking. Summaries break down by category.
- Added `generalization` probes (unseen capital, unseen arithmetic, unseen harm category,
  over-refusal, instruction following, indirect identity), all advisory pending a GPU run. The
  existing probes are labelled `memorization`, which is what they actually test.
- `fact-arithmetic` now matches `\b4\b` / `\bfour\b` instead of the bare substring `4`, which also
  matched "2024".
- `train_run.json` distinguishes `started` / `dry-run` / `completed` and records duration, final
  train loss, global step, package versions, GPU name, config, data SHA-256, and adapter SHA-256.
  A crashed run no longer leaves a log that looks successful.
- `_filter_kwargs` reports dropped kwargs. Training warns when the installed TRL ignores a critical
  one (notably `assistant_only_loss`, which changes loss masking) and aborts if no sequence-length
  kwarg survives. A contract test fails CI on that drift.
- Covered eval's generation loop with stubbed model classes, so the adapter-vs-base path, report
  shape, and exit codes run in CI without a GPU.
- Added a real CPU smoke train (tiny random Llama, bundled dataset, seconds) so the TRL/PEFT
  integration is exercised in CI. Previously only `--dry-run` ran, which returns before importing
  torch.
- `train_run.json` records the chat template path and SHA-256, and how many tokens the first
  example actually supervises.

## 0.1.0

- Public workshop layout: TRL SFT + chat template, smoke SFT, eval probes, Hub upload eval gate
- Community data path under `datasets/community/`
- CI: ruff, pytest, dry-run train, eval `--check-only`
- README: one-line purpose; MIT link at repo root
