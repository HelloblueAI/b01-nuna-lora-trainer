"""_filter_kwargs silently drops unknown kwargs, so pin drift can change training.

pyproject allows a wide TRL range. These tests turn "the installed TRL quietly ignored
your config" into a CI failure instead of a differently-trained adapter.
"""

import pytest

from b01_nuna_lora.train import (
    CRITICAL_SFT_KWARGS,
    EITHER_OR_SFT_KWARGS,
    _check_dropped_sft_kwargs,
    _filter_kwargs,
)

try:
    import trl
except ImportError:  # pragma: no cover - depends on the install profile
    trl = None

requires_trl = pytest.mark.skipif(trl is None, reason="TRL only in the full dev env")


def _accepted(name: str) -> bool:
    kept, _ = _filter_kwargs(trl.SFTConfig, {name: None})
    return name in kept


@requires_trl
def test_sequence_length_kwarg_still_exists():
    for pair in EITHER_OR_SFT_KWARGS:
        assert any(_accepted(name) for name in pair), (
            f"installed trl {trl.__version__} accepts none of {pair}; "
            "max_length would fall back to a silent default"
        )


@requires_trl
@pytest.mark.parametrize("name", ["seed", "learning_rate", "num_train_epochs", "output_dir"])
def test_core_training_kwargs_still_exist(name):
    assert _accepted(name), f"installed trl {trl.__version__} dropped SFTConfig.{name}"


@requires_trl
def test_assistant_only_loss_support_is_explicit():
    """Not a hard failure: older supported TRL lacks it, but the run must say so."""
    if _accepted("assistant_only_loss"):
        return
    warnings = _check_dropped_sft_kwargs(["assistant_only_loss"])
    assert any("loss will include prompt tokens" in w for w in warnings)
    pytest.skip(
        f"trl {trl.__version__} has no assistant_only_loss; training masks nothing. "
        "Bump the TRL floor in pyproject to train as documented."
    )


def test_filter_kwargs_reports_what_it_dropped():
    def target(alpha, beta):
        return alpha, beta

    kept, dropped = _filter_kwargs(target, {"alpha": 1, "gamma": 2, "delta": 3})
    assert kept == {"alpha": 1}
    assert dropped == ["delta", "gamma"]


def test_filter_kwargs_keeps_everything_for_var_keyword_callables():
    def target(**kwargs):
        return kwargs

    kept, dropped = _filter_kwargs(target, {"anything": 1})
    assert kept == {"anything": 1}
    assert dropped == []


def test_every_critical_kwarg_produces_a_warning_when_dropped():
    warnings = _check_dropped_sft_kwargs(list(CRITICAL_SFT_KWARGS))
    assert len(warnings) == len(CRITICAL_SFT_KWARGS)
    assert any("loss will include prompt tokens" in w for w in warnings)


def test_no_warnings_when_nothing_critical_dropped():
    assert _check_dropped_sft_kwargs(["report_to", "dataloader_pin_memory"]) == []


def test_losing_both_sequence_length_kwargs_is_fatal():
    with pytest.raises(SystemExit, match="sequence length would be a silent default"):
        _check_dropped_sft_kwargs(["max_length", "max_seq_length"])
