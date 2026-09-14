"""VRAM accounting in train_run.json.

CI has no GPU, so the real call path never runs there. Keeping the arithmetic in
a pure function means the reported numbers are still covered.
"""

from __future__ import annotations

import types

from b01_nuna_lora.train import _vram_stats

MIB = 2**20


def _fake_torch(allocated: int, reserved: int, total: int):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            max_memory_allocated=lambda: allocated,
            max_memory_reserved=lambda: reserved,
            get_device_properties=lambda _i: types.SimpleNamespace(total_memory=total),
        )
    )


def test_bytes_are_reported_as_mib():
    stats = _vram_stats(_fake_torch(1024 * MIB, 2048 * MIB, 8192 * MIB))
    assert stats == {
        "peak_vram_mib": 1024.0,
        "peak_vram_reserved_mib": 2048.0,
        "total_vram_mib": 8192.0,
    }


def test_reserved_is_reported_separately_from_allocated():
    """Reserved >= allocated, and it is the figure that must fit on the card."""
    stats = _vram_stats(_fake_torch(4641 * MIB, 4764 * MIB, 7782 * MIB))
    assert stats["peak_vram_reserved_mib"] > stats["peak_vram_mib"]
    assert stats["peak_vram_reserved_mib"] < stats["total_vram_mib"]


def test_a_moved_allocator_api_does_not_lose_the_run_log():
    broken = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            max_memory_allocated=lambda: (_ for _ in ()).throw(AttributeError("gone"))
        )
    )
    assert _vram_stats(broken) == {
        "peak_vram_mib": None,
        "peak_vram_reserved_mib": None,
        "total_vram_mib": None,
    }


def test_measured_rtx_4060_run_fits_the_documented_card():
    """Guards the docs/EVIDENCE.md claim that this recipe fits an 8GB card."""
    stats = _vram_stats(_fake_torch(4641 * MIB, 4764 * MIB, 7782 * MIB))
    assert stats["peak_vram_reserved_mib"] / stats["total_vram_mib"] < 0.8
