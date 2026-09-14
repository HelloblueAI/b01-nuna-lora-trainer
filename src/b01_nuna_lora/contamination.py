"""Train/eval overlap detection (no GPU).

A probe that restates a training row measures memorization, not capability. This
module scores every probe against the training corpus so a report can say which
of its own probes are compromised, instead of leaving the reader to guess.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .scoring import last_user_text, load_probes

# Token-Jaccard / char-Dice above these counts as overlap. Calibrated against the
# bundled corpus: "What is 2 + 2?" vs "What is two plus two?" must land in NEAR.
NEAR_DUPLICATE = 0.60
RELATED = 0.35

_WORD = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    folded = text.replace("\u00a0", " ").replace("\u202f", " ").lower()
    # Digits and their words are the same claim for contamination purposes.
    for digit, word in (
        ("0", "zero"), ("1", "one"), ("2", "two"), ("3", "three"), ("4", "four"),
        ("5", "five"), ("6", "six"), ("7", "seven"), ("8", "eight"), ("9", "nine"),
    ):
        folded = re.sub(rf"\b{digit}\b", word, folded)
    folded = folded.replace("+", " plus ").replace("=", " equals ")
    return " ".join(folded.split())


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(_normalize(text)))


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    squashed = _normalize(text).replace(" ", "")
    if len(squashed) < n:
        return {squashed} if squashed else set()
    return {squashed[i : i + n] for i in range(len(squashed) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def similarity(left: str, right: str) -> float:
    """0..1 overlap. Word-set and character-shape agreement, whichever is stronger.

    Two measures because each misses a different kind of restatement: token
    Jaccard catches reordering ("capital of France" / "France's capital"), char
    n-grams catch inflection and punctuation drift.
    """
    return max(
        _jaccard(_tokens(left), _tokens(right)),
        _jaccard(_char_ngrams(left), _char_ngrams(right)),
    )


def classify(score: float, *, exact: bool, answer_leaked: bool) -> str:
    """Contamination needs both halves: a reused question *and* the same answer.

    Reusing a training question's shape with an unseen answer ("capital of
    Japan" against a France row) is what a generalization probe is supposed to
    do, so it gets its own verdict rather than being called a leak.
    """
    reused = exact or score >= NEAR_DUPLICATE
    if reused and answer_leaked:
        return "recall"
    if reused:
        return "form-reuse"
    if score >= RELATED and answer_leaked:
        return "related-recall"
    if score >= RELATED:
        return "related"
    return "clean"


def training_pairs(path: str | Path) -> list[dict[str, str]]:
    """Flatten a chat-format SFT file into (prompt, answer) pairs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}")

    pairs: list[dict[str, str]] = []
    for row in raw:
        messages = (row or {}).get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list):
            continue
        prompt = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role, content = message.get("role"), str(message.get("content") or "")
            if role == "user":
                prompt = content
            elif role == "assistant" and prompt:
                pairs.append({"prompt": prompt, "answer": content, "source": str(path)})
                prompt = ""
    return pairs


def load_corpus(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    for path in paths:
        corpus.extend(training_pairs(path))
    return corpus


def _literals(patterns: Iterable[Any]) -> list[str]:
    """Regex patterns reduced to the plain text a reader would recognize."""
    out = []
    for pattern in patterns or []:
        literal = re.sub(r"\\b|\\s|[\\^$.*+?()\[\]{}|]", "", str(pattern)).strip()
        if literal:
            out.append(literal)
    return out


def analyze_probe(probe: dict[str, Any], corpus: list[dict[str, str]]) -> dict[str, Any]:
    prompt = last_user_text(probe)
    normalized = _normalize(prompt)

    best: dict[str, str] = {}
    best_score = 0.0
    exact = False
    for pair in corpus:
        score = similarity(prompt, pair["prompt"])
        if _normalize(pair["prompt"]) == normalized:
            exact = True
        if score > best_score:
            best_score, best = score, pair

    # Leakage is only meaningful against training rows the probe actually
    # resembles. Scanning the whole corpus flags any probe expecting a common
    # word ("blue", "can't"), which says nothing about this probe.
    expected = _literals(probe.get("any_must_match"))
    neighbours = [p for p in corpus if similarity(prompt, p["prompt"]) >= RELATED]
    leaked = sorted(
        {
            literal
            for literal in expected
            if any(literal.lower() in pair["answer"].lower() for pair in neighbours)
        }
    )

    return {
        "id": probe.get("id"),
        "category": probe.get("category", "uncategorized"),
        "required": bool(probe.get("required", True)),
        "prompt": prompt,
        "verdict": classify(best_score, exact=exact, answer_leaked=bool(leaked)),
        "prompt_similarity": round(best_score, 3),
        "nearest_training_prompt": best.get("prompt", ""),
        "nearest_training_answer": best.get("answer", ""),
        "answer_leaked_from_neighbour": leaked,
    }


# Verdicts where a pass is explained by having seen the answer during training.
CONTAMINATED = ("recall", "related-recall")


def analyze(probes: list[dict[str, Any]], corpus: list[dict[str, str]]) -> dict[str, Any]:
    rows = [analyze_probe(probe, corpus) for probe in probes]
    contaminated = [r for r in rows if r["verdict"] in CONTAMINATED]

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    required = [r for r in rows if r["required"]]
    return {
        "training_examples": len(corpus),
        "probes": len(rows),
        "by_verdict": counts,
        "contaminated": [r["id"] for r in contaminated],
        # The honest headline: how much of the gate is not explained by recall.
        "required_uncontaminated": sum(1 for r in required if r["verdict"] not in CONTAMINATED),
        "required_total": len(required),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect train/eval contamination.")
    parser.add_argument("--probes", default="datasets/eval_probes.json")
    parser.add_argument(
        "--train", nargs="+", default=["datasets/train.json"], help="SFT files to compare against"
    )
    parser.add_argument("--report", default=None, help="Write JSON here")
    parser.add_argument(
        "--fail-on-contaminated-required",
        action="store_true",
        help="Exit non-zero if any required probe restates a training row",
    )
    args = parser.parse_args()

    probes = load_probes(args.probes)
    corpus = load_corpus(args.train)
    result = analyze(probes, corpus)

    print(f"{result['probes']} probes vs {result['training_examples']} training examples\n")
    for label in ("recall", "related-recall", "form-reuse", "related", "clean"):
        if label in result["by_verdict"]:
            print(f"  {label:<15} {result['by_verdict'][label]}")
    print(
        f"\nrequired probes not explained by recall: "
        f"{result['required_uncontaminated']}/{result['required_total']}"
    )

    for row in result["results"]:
        if row["verdict"] in CONTAMINATED:
            print(f"\n  [{row['verdict']}] {row['id']} (prompt sim={row['prompt_similarity']})")
            print(f"    probe:    {row['prompt']}")
            print(f"    training: {row['nearest_training_prompt']}")
            leaked = ", ".join(row["answer_leaked_from_neighbour"])
            print(f"    answer already in training: {leaked}")

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {path}")

    if args.fail_on_contaminated_required:
        blocking = [
            r["id"] for r in result["results"] if r["required"] and r["verdict"] in CONTAMINATED
        ]
        if blocking:
            raise SystemExit(
                "Required probes restate training data: " + ", ".join(map(str, blocking))
            )


if __name__ == "__main__":
    main()
