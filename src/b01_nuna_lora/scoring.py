"""Eval matchers (no GPU). Same idea as product goldens: any_must_match / must_not_match."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Files that determine adapter behavior; hashed to bind an eval report to an adapter.
ADAPTER_FINGERPRINT_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _normalize(text: str) -> str:
    folded = text.replace("\u00a0", " ").replace("\u202f", " ")
    return " ".join(folded.lower().split())


def _matches(pattern: str, haystack: str) -> bool:
    try:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return True
    except re.error:
        # validate_probes rejects these up front; treat as literal so a GPU run
        # never dies mid-generation on a contributor's regex typo.
        pass
    return _normalize(pattern) in _normalize(haystack)


def score_completion(text: str, probe: dict[str, Any]) -> dict[str, Any]:
    any_must = probe.get("any_must_match") or []
    must_not = probe.get("must_not_match") or []
    missing = [p for p in any_must if not _matches(str(p), text)]
    leaked = [p for p in must_not if _matches(str(p), text)]
    hit_required = True if not any_must else any(_matches(str(p), text) for p in any_must)
    passed = hit_required and not leaked
    return {
        "id": probe.get("id"),
        "category": probe.get("category", "uncategorized"),
        "required": bool(probe.get("required", True)),
        "passed": passed,
        "missing_any_required": missing if not hit_required else [],
        "forbidden_hits": leaked,
    }


def last_user_text(probe: dict[str, Any]) -> str:
    """The prompt sent to the model: `prompt`, or the last user turn in `messages`."""
    if "prompt" in probe:
        return str(probe["prompt"])
    messages = probe.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    raise ValueError("has no prompt and no user message")


def validate_probes(probes: list[Any]) -> list[str]:
    """Return every schema/regex problem in a probe list, so CI can fail with all of them."""
    errors: list[str] = []
    seen: set[str] = set()
    for i, probe in enumerate(probes):
        if not isinstance(probe, dict):
            errors.append(f"probe[{i}] is not an object")
            continue

        probe_id = probe.get("id")
        where = f"probe[{i}] id={probe_id!r}"
        if not isinstance(probe_id, str) or not probe_id.strip():
            errors.append(f"{where}: needs a non-empty string 'id'")
        elif probe_id in seen:
            errors.append(f"{where}: duplicate 'id'")
        else:
            seen.add(probe_id)

        try:
            if not last_user_text(probe).strip():
                errors.append(f"{where}: prompt is empty")
        except ValueError as exc:
            errors.append(f"{where}: {exc}")

        if "required" in probe and not isinstance(probe["required"], bool):
            errors.append(f"{where}: 'required' must be true or false")
        if "category" in probe and not (
            isinstance(probe["category"], str) and probe["category"].strip()
        ):
            errors.append(f"{where}: 'category' must be a non-empty string")

        for field in ("any_must_match", "must_not_match"):
            patterns = probe.get(field)
            if patterns is None:
                continue
            if not isinstance(patterns, list):
                errors.append(f"{where}: '{field}' must be a list")
                continue
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern:
                    errors.append(f"{where}: '{field}' has an empty or non-string pattern")
                    continue
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(
                        f"{where}: '{field}' pattern {pattern!r} is not valid regex ({exc})"
                    )

        if not probe.get("any_must_match") and not probe.get("must_not_match"):
            errors.append(f"{where}: needs any_must_match or must_not_match")
    return errors


def load_probes(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    errors = validate_probes(raw)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"Invalid probes in {path}:\n  - {joined}")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> str:
    return _sha256_file(Path(path))


def fingerprint_adapter(adapter: str | Path) -> str:
    """Stable hash over the adapter files that change generation behavior."""
    directory = Path(adapter)
    digest = hashlib.sha256()
    for name in ADAPTER_FINGERPRINT_FILES:
        target = directory / name
        if not target.exists():
            raise SystemExit(f"Adapter dir is missing {name}: {directory}")
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_file(target).encode("utf-8"))
    return digest.hexdigest()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate on `required` probes only; advisory probes are reported but never block."""
    passed = sum(1 for row in results if row.get("passed"))
    required = [row for row in results if row.get("required", True)]
    advisory = [row for row in results if not row.get("required", True)]
    required_passed = sum(1 for row in required if row.get("passed"))

    by_category: dict[str, dict[str, int]] = {}
    for row in results:
        bucket = by_category.setdefault(
            str(row.get("category", "uncategorized")), {"total": 0, "passed": 0}
        )
        bucket["total"] += 1
        bucket["passed"] += 1 if row.get("passed") else 0

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "required_total": len(required),
        "required_passed": required_passed,
        "advisory_total": len(advisory),
        "advisory_passed": sum(1 for row in advisory if row.get("passed")),
        "by_category": by_category,
        "ok": bool(required) and required_passed == len(required),
    }


def compare_to_baseline(
    results: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> dict[str, Any]:
    """Which probes the adapter actually changed, versus the base model alone."""
    base_by_id = {row.get("id"): bool(row.get("passed")) for row in baseline}
    gained, lost, already = [], [], []
    for row in results:
        probe_id = row.get("id")
        if probe_id not in base_by_id:
            continue
        if row.get("passed") and not base_by_id[probe_id]:
            gained.append(probe_id)
        elif not row.get("passed") and base_by_id[probe_id]:
            lost.append(probe_id)
        elif row.get("passed"):
            already.append(probe_id)
    return {
        "gained": gained,
        "regressed": lost,
        "base_already_passed": already,
        "proves_adapter_effect": bool(gained) and not lost,
    }


def write_report(
    path: Path,
    results: list[dict[str, Any]],
    *,
    mode: str = "generation",
    provenance: dict[str, Any] | None = None,
    baseline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": mode,
        "provenance": provenance or {},
        "summary": summarize(results),
        "results": results,
    }
    if baseline is not None:
        report["baseline"] = {"summary": summarize(baseline), "results": baseline}
        report["comparison"] = compare_to_baseline(results, baseline)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def require_passing_report(path: Path, *, adapter: str | Path | None = None) -> dict[str, Any]:
    """Gate an upload: the report must be a passing generation run for *this* adapter."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("mode") == "check-only":
        raise SystemExit("Eval report is check-only (no generation). Run GPU eval before upload.")
    summary = report.get("summary") or {}
    if not summary.get("ok"):
        required_total = summary.get("required_total", summary.get("total"))
        required_passed = summary.get("required_passed", summary.get("passed"))
        raise SystemExit(
            f"Eval gate failed ({required_passed}/{required_total} required probes): {path}"
        )

    if adapter is not None:
        recorded = (report.get("provenance") or {}).get("adapter_sha256")
        if not recorded:
            raise SystemExit(
                f"Eval report {path} has no adapter provenance (written by an older version). "
                "Re-run the generation eval against this adapter before uploading."
            )
        actual = fingerprint_adapter(adapter)
        if recorded != actual:
            raise SystemExit(
                f"Eval report {path} does not match --adapter {adapter}.\n"
                f"  report adapter_sha256: {recorded}\n"
                f"  actual adapter_sha256: {actual}\n"
                "The adapter changed since it was evaluated. Re-run the generation eval."
            )
    return report
