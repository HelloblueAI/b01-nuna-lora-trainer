import pytest

from b01_nuna_lora.preferences import load_preferences, preference_example, split_preference


def test_split_preference_keeps_the_last_assistant_turn():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "sure"},
    ]
    prompt, chosen = split_preference(messages)
    assert prompt[-1]["content"] == "more"
    assert chosen == [{"role": "assistant", "content": "sure"}]


def test_preference_example_rejects_different_prompts():
    with pytest.raises(ValueError, match="differ"):
        preference_example(
            {
                "chosen": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "yes"},
                ],
                "rejected": [
                    {"role": "user", "content": "b"},
                    {"role": "assistant", "content": "no"},
                ],
            }
        )


def test_load_preferences_rejects_a_noncommercial_license():
    with pytest.raises(SystemExit, match="hf_license"):
        load_preferences("some/repo", "train", max_samples=10, license_id="cc-by-nc")
