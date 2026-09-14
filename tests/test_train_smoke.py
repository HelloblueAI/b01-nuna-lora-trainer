"""End-to-end SFTTrainer run on CPU with a tiny random model.

CI previously only ran `train --dry-run`, which returns before importing torch, so the
TRL/PEFT integration was never exercised. This trains for real (seconds, no GPU, no 1.1B
download) against the bundled dataset and the shipped chat template.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("torch", reason="full dev env only")
transformers = pytest.importorskip("transformers", reason="full dev env only")
trl = pytest.importorskip("trl", reason="full dev env only")
datasets = pytest.importorskip("datasets", reason="full dev env only")

from b01_nuna_lora.data import load_records  # noqa: E402
from b01_nuna_lora.train import check_assistant_masks  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as exc:  # offline CI
        pytest.skip(f"cannot fetch TinyLlama tokenizer: {exc}")
    tok.pad_token = tok.eos_token
    return tok


def _tiny_model(tokenizer):
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return transformers.LlamaForCausalLM(config)


def _train(tokenizer, tmp_path, records):
    args = trl.SFTConfig(
        output_dir=str(tmp_path),
        assistant_only_loss=True,
        max_length=256,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        report_to="none",
        logging_steps=100,
        use_cpu=True,
        save_strategy="no",
    )
    trainer = trl.SFTTrainer(
        model=_tiny_model(tokenizer),
        args=args,
        train_dataset=datasets.Dataset.from_list(records),
        processing_class=tokenizer,
    )
    return trainer.train()


def test_bundled_dataset_trains_with_the_shipped_template(tokenizer, tmp_path):
    tokenizer.chat_template = (ROOT / "configs/tinyllama_chat_template.jinja").read_text()
    records = load_records(ROOT / "datasets/train.json")

    assert check_assistant_masks(tokenizer, records) > 0
    result = _train(tokenizer, tmp_path, records)

    assert result.global_step > 0
    assert result.training_loss == pytest.approx(result.training_loss)  # not NaN


def test_stock_template_is_caught_before_trl_raises(tokenizer):
    """Regression: the documented config used to reach TRL and die there."""
    stock = transformers.AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ).chat_template
    tokenizer.chat_template = stock
    records = load_records(ROOT / "datasets/train.json")

    with pytest.raises(SystemExit, match="nothing to learn from"):
        check_assistant_masks(tokenizer, records)
