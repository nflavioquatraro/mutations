"""Validation contracts for model-level adapter targeting and prompt routing."""

import pytest

from anchored_moe import AdapterRoute, AdapterTargetConfig, PromptRoutingConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"last_n": 0},
        {"last_n": True},
        {"rank": 0},
        {"alpha": float("nan")},
        {"dropout": 1},
        {"layers": "bad"},
        {"layers": (1, 1), "last_n": None},
        {"layers": "all"},
        {"components": ("normalization",)},
        {"module_paths": ("mlp.up_proj",)},
        {"components": ()},
        {"rank_pattern": (("x", 0),)},
        {"rank_pattern": (("x", 2), ("x", 3))},
    ],
)
def test_invalid_target_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        AdapterTargetConfig(**kwargs)


@pytest.mark.parametrize("confidence", [-1, 2, float("nan"), True, "0.5"])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        AdapterRoute("domain", confidence)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_history_limit(count):
    with pytest.raises(ValueError):
        PromptRoutingConfig(history_messages=count)
