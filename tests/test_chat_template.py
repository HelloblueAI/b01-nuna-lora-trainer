"""assistant_only_loss needs {% generation %} markers that stock TinyLlama does not have.

Without them TRL either raises deep in dataset prep or trains on the whole sequence, so the
repo ships a patched template. It must mark assistant spans *and* render exactly like stock,
because eval and any downstream user rely on the prompt format being unchanged.
"""

from pathlib import Path

import pytest

from b01_nuna_lora.train import check_assistant_masks, count_supervised_tokens

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/tinyllama_chat_template.jinja"

CASES = {
    "single turn": [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I'm B01."},
    ],
    "with system": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
    ],
    "multi turn": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ],
}


def test_template_file_has_generation_markers():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "{% generation %}" in text and "{% endgeneration %}" in text


def test_default_config_points_at_the_patched_template():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    assert cfg["assistant_only_loss"] is True
    assert (ROOT / cfg["chat_template"]).resolve() == TEMPLATE.resolve()


class FakeTokenizer:
    """Mimics apply_chat_template's mask contract without a tokenizer download."""

    def __init__(self, masks):
        self._masks = masks

    def apply_chat_template(self, messages, **_kwargs):
        return {"input_ids": [1] * len(self._masks), "assistant_masks": self._masks}


def test_preflight_accepts_a_template_that_marks_assistant_tokens():
    records = [{"messages": CASES["single turn"]}]
    assert check_assistant_masks(FakeTokenizer([0, 0, 1, 1]), records) == 2


def test_preflight_rejects_a_template_that_marks_nothing():
    records = [{"messages": CASES["single turn"]}]
    with pytest.raises(SystemExit) as exc:
        check_assistant_masks(FakeTokenizer([0, 0, 0, 0]), records)
    message = str(exc.value)
    assert "nothing to learn from" in message
    assert "configs/tinyllama_chat_template.jinja" in message
    assert "assistant_only_loss: false" in message


def test_counts_zero_when_masks_are_absent_entirely():
    assert count_supervised_tokens(FakeTokenizer([]), CASES["single turn"]) == 0


# --- Anything below needs the real tokenizer (network + transformers). ---

transformers = pytest.importorskip("transformers", reason="full dev env only")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as exc:  # offline CI
        pytest.skip(f"cannot fetch TinyLlama tokenizer: {exc}")


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_patched_template_renders_identically_to_stock(tokenizer, case, add_generation_prompt):
    stock = tokenizer.chat_template
    patched = TEMPLATE.read_text(encoding="utf-8")
    messages = CASES[case]

    tokenizer.chat_template = stock
    expected = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    tokenizer.chat_template = patched
    actual = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    tokenizer.chat_template = stock
    assert actual == expected


def test_stock_template_supervises_nothing(tokenizer):
    """The bug this guards against: the documented setup trained on zero tokens."""
    assert count_supervised_tokens(tokenizer, CASES["single turn"]) == 0


@pytest.mark.parametrize("case", list(CASES))
def test_patched_template_supervises_only_assistant_turns(tokenizer, case):
    stock = tokenizer.chat_template
    tokenizer.chat_template = TEMPLATE.read_text(encoding="utf-8")
    try:
        messages = CASES[case]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
        )
        supervised = [
            i for i, flag in zip(encoded["input_ids"], encoded["assistant_masks"]) if flag
        ]
        assert supervised, "patched template must mark assistant tokens"
        decoded = tokenizer.decode(supervised)
        expected = "".join(
            m["content"] + tokenizer.eos_token for m in messages if m["role"] == "assistant"
        )
        assert decoded == expected
    finally:
        tokenizer.chat_template = stock
