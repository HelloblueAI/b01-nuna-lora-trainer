"""4-bit QLoRA config. No GPU and no bitsandbytes: the decisions are pure."""

from __future__ import annotations

import json

import pytest

from b01_nuna_lora.eval import resolve_quantization
from b01_nuna_lora.train import (
    apply_4bit_training_precision,
    quantization_kwargs,
    recorded_quantization,
    require_trainer_accepts_quantization,
)


def test_four_bit_off_by_default():
    assert quantization_kwargs({}) is None
    assert quantization_kwargs({"load_in_4bit": False}) is None


def test_nf4_double_quant_is_the_default_qlora_setup():
    quant = quantization_kwargs({"load_in_4bit": True})
    assert quant == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }


def test_unknown_quant_type_is_rejected():
    with pytest.raises(SystemExit, match="nf4"):
        quantization_kwargs({"load_in_4bit": True, "bnb_4bit_quant_type": "int8"})


def test_unknown_compute_dtype_is_rejected():
    with pytest.raises(SystemExit, match="float16"):
        quantization_kwargs({"load_in_4bit": True, "bnb_4bit_compute_dtype": "int4"})


def test_four_bit_switches_training_to_bf16_on_a_capable_gpu():
    sft = {"fp16": True, "bf16": False}
    quant = quantization_kwargs({"load_in_4bit": True})
    precision = apply_4bit_training_precision(sft, quant, bf16_supported=True)
    assert precision == "bf16"
    assert sft["fp16"] is False and sft["bf16"] is True
    assert quant["bnb_4bit_compute_dtype"] == "bfloat16"


def test_four_bit_refuses_a_gpu_without_bf16():
    quant = quantization_kwargs({"load_in_4bit": True})
    with pytest.raises(SystemExit, match="bfloat16"):
        apply_4bit_training_precision({"fp16": True, "bf16": False}, quant, bf16_supported=False)


def test_full_precision_recipe_is_left_alone():
    sft = {"fp16": True, "bf16": False}
    precision = apply_4bit_training_precision(sft, None, bf16_supported=True)
    assert precision == "fp16"
    assert sft == {"fp16": True, "bf16": False}


def test_dropped_quantization_config_fails_closed():
    with pytest.raises(SystemExit, match="silently stay in full precision"):
        require_trainer_accepts_quantization(["quantization_config"], enabled=True)


def test_dropped_quantization_config_is_irrelevant_when_4bit_is_off():
    require_trainer_accepts_quantization(["quantization_config"], enabled=False)


def test_recorded_quantization_reads_the_train_log(tmp_path):
    quant = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    (tmp_path / "train_run.json").write_text(json.dumps({"quantization": quant}))
    assert recorded_quantization(tmp_path) == quant


def test_full_precision_train_log_means_no_quantization(tmp_path):
    (tmp_path / "train_run.json").write_text(json.dumps({"load_in_4bit": False}))
    assert recorded_quantization(tmp_path) is None


def test_missing_train_log_means_no_quantization(tmp_path):
    assert recorded_quantization(tmp_path) is None


def test_eval_follows_the_adapter_unless_overridden(tmp_path):
    quant = quantization_kwargs({"load_in_4bit": True})
    (tmp_path / "train_run.json").write_text(json.dumps({"quantization": quant}))
    assert resolve_quantization(tmp_path, None) == quant
    assert resolve_quantization(tmp_path, False) is None


def test_eval_flag_turns_4bit_on_without_a_train_log(tmp_path):
    resolved = resolve_quantization(tmp_path, True)
    assert resolved is not None and resolved["load_in_4bit"] is True
