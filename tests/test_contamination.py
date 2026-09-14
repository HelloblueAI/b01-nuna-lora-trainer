"""Train/eval overlap detection.

The distinction under test: reusing a training question's *shape* with an unseen
answer is a good generalization probe, while reusing the question *and* its
answer means a pass proves only recall.
"""

from __future__ import annotations

import json

import pytest

from b01_nuna_lora.contamination import (
    analyze,
    analyze_probe,
    load_corpus,
    similarity,
    training_pairs,
)


def _pair(prompt: str, answer: str) -> dict[str, str]:
    return {"prompt": prompt, "answer": answer, "source": "test"}


def _probe(pid: str, prompt: str, must_match: list[str], **extra):
    return {"id": pid, "prompt": prompt, "any_must_match": must_match, **extra}


class TestSimilarity:
    def test_identical_text_scores_one(self):
        assert similarity("What is the capital of France?", "What is the capital of France?") == 1.0

    def test_unrelated_text_scores_low(self):
        assert similarity("What is the capital of France?", "Write me a poem about rain") < 0.35

    def test_digits_and_number_words_are_the_same_question(self):
        # "What is 2 + 2?" and "What is two plus two?" are the same probe.
        assert similarity("What is 2 + 2?", "What is two plus two?") >= 0.6

    def test_reordering_still_counts_as_overlap(self):
        assert similarity("the capital of France", "France's capital") >= 0.35


class TestVerdicts:
    def test_reused_question_with_leaked_answer_is_recall(self):
        corpus = [_pair("What is the capital of France?", "The capital of France is Paris.")]
        row = analyze_probe(_probe("p", "What is the capital of France?", ["Paris"]), corpus)
        assert row["verdict"] == "recall"
        assert row["answer_leaked_from_neighbour"] == ["Paris"]

    def test_reused_question_with_unseen_answer_is_form_reuse(self):
        """The point of the module: same template, different answer, is legitimate."""
        corpus = [_pair("What is the capital of France?", "The capital of France is Paris.")]
        row = analyze_probe(_probe("p", "What is the capital of Japan?", ["Tokyo"]), corpus)
        assert row["verdict"] == "form-reuse"
        assert row["answer_leaked_from_neighbour"] == []

    def test_paraphrased_question_with_leaked_answer_is_flagged(self):
        corpus = [_pair("What is the capital of France?", "The capital of France is Paris.")]
        row = analyze_probe(_probe("p", "Name the capital city of France.", ["Paris"]), corpus)
        assert row["verdict"] == "related-recall"

    def test_unrelated_probe_is_clean(self):
        corpus = [_pair("What is the capital of France?", "The capital of France is Paris.")]
        row = analyze_probe(_probe("p", "Write a haiku about autumn", ["leaves"]), corpus)
        assert row["verdict"] == "clean"

    def test_common_words_elsewhere_in_corpus_do_not_count_as_leakage(self):
        """A probe expecting "blue" is not contaminated by an unrelated row saying "blue"."""
        corpus = [_pair("Name one primary color.", "Red is a primary color, with blue and yellow.")]
        row = analyze_probe(_probe("p", "What colour is a clear sky?", ["blue"]), corpus)
        assert row["answer_leaked_from_neighbour"] == []
        assert row["verdict"] == "clean"

    def test_regex_patterns_are_reduced_to_literals(self):
        corpus = [_pair("What is 2 + 2?", "2 + 2 equals 4.")]
        row = analyze_probe(_probe("p", "What is 2 + 2?", [r"\b4\b", r"\bfour\b"]), corpus)
        assert "4" in row["answer_leaked_from_neighbour"]


class TestCorpusLoading:
    def test_chat_rows_become_prompt_answer_pairs(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(
            json.dumps(
                [{"messages": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}]}]
            )
        )
        assert training_pairs(path) == [
            {"prompt": "hi", "answer": "hello", "source": str(path)}
        ]

    def test_rows_without_messages_are_skipped(self, tmp_path):
        """identity-seed.json is Alpaca-format; it must not crash the loader."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps([{"instruction": "Who are you?", "output": "B01"}]))
        assert training_pairs(path) == []

    def test_multi_turn_rows_yield_every_exchange(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(
            json.dumps(
                [{"messages": [
                    {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
                ]}]
            )
        )
        assert [p["prompt"] for p in training_pairs(path)] == ["a", "c"]

    def test_non_array_file_is_rejected(self, tmp_path):
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"messages": []}))
        with pytest.raises(ValueError):
            training_pairs(path)


class TestAnalyze:
    def test_summary_counts_required_probes_not_explained_by_recall(self):
        corpus = [_pair("What is 2 + 2?", "2 + 2 equals 4.")]
        probes = [
            _probe("leaky", "What is two plus two?", [r"\b4\b"], required=True),
            _probe("clean", "Write a haiku", ["autumn"], required=True),
        ]
        result = analyze(probes, corpus)
        assert result["contaminated"] == ["leaky"]
        assert result["required_uncontaminated"] == 1
        assert result["required_total"] == 2


class TestBundledData:
    """Guards the repo's own numbers, so a new probe cannot quietly be a copy."""

    def test_bundled_probes_are_analyzable(self):
        from b01_nuna_lora.scoring import load_probes

        probes = load_probes("datasets/eval_probes.json")
        result = analyze(probes, load_corpus(["datasets/train.json"]))
        assert result["probes"] == len(probes)

    def test_known_contaminated_probes_are_detected(self):
        from b01_nuna_lora.scoring import load_probes

        probes = load_probes("datasets/eval_probes.json")
        result = analyze(probes, load_corpus(["datasets/train.json"]))
        # Measured against the bundled corpus; these restate training rows.
        assert "fact-arithmetic" in result["contaminated"]
        assert "fact-paris" in result["contaminated"]

    def test_unseen_capital_probe_is_not_called_contaminated(self):
        """It reuses the France question's shape but Tokyo is never trained."""
        from b01_nuna_lora.scoring import load_probes

        probes = load_probes("datasets/eval_probes.json")
        result = analyze(probes, load_corpus(["datasets/train.json"]))
        assert "gen-fact-unseen-capital" not in result["contaminated"]
