import torch
from safetensors.torch import save_file

from b01_nuna_lora.train import nonfinite_weight_keys


def test_finite_weights_pass(tmp_path):
    save_file({"w": torch.zeros(2, 2)}, tmp_path / "model.safetensors")
    assert nonfinite_weight_keys(tmp_path) == []


def test_nonfinite_weights_are_named(tmp_path):
    save_file({"layer.weight": torch.tensor([1.0, float("nan")])}, tmp_path / "a.safetensors")
    found = nonfinite_weight_keys(tmp_path)
    assert found == ["a.safetensors:layer.weight"]
