"""Eval matchers (no GPU). Same idea as product goldens: any_must_match / must_not_match."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _normalize(text: str) -> str:
    folded = text.replace("\u00a0", " ").replace("\u202f", " ")
    return " ".join(folded.lower().split())


def _matches(pattern: str, haystack: str) -> bool:
    if re.search(pattern, haystack, flags=re.IGNORECASE):
        return True
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
        "passed": passed,
        "missing_any_required": missing if not hit_required else [],
        "forbidden_hits": leaked,
    }


def load_probes(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    return raw


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in results if row.get("passed"))
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "ok": bool(results) and passed == len(results),
    }


def write_report(
    path: Path, results: list[dict[str, Any]], *, mode: str = "generation"
) -> dict[str, Any]:
    report = {"mode": mode, "summary": summarize(results), "results": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def require_passing_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") == "check-only":
        raise SystemExit("Eval report is check-only (no generation). Run GPU eval before upload.")
    summary = report.get("summary") or {}
    if not summary.get("ok"):
        raise SystemExit(
            f"Eval gate failed ({summary.get('passed')}/{summary.get('total')}): {path}"
        )
    return report
