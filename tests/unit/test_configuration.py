"""Tests for deterministic strategy and layer configuration."""

from __future__ import annotations

import pytest

from anchored_moe import AnchoredMoEConfig


def test_upper_half_resolution_is_deterministic_for_odd_depth() -> None:
    config = AnchoredMoEConfig(converted_layers="upper_half")

    assert config.resolve_converted_layers(5) == (2, 3, 4)


def test_explicit_layers_are_sorted_and_round_trip() -> None:
    config = AnchoredMoEConfig(
        base_model_name_or_path="Qwen/Qwen3-4B",
        base_model_revision="pinned-revision",
        converted_layers=[3, 1],
    )

    restored = AnchoredMoEConfig.from_dict(config.to_dict())

    assert restored.converted_layers == [1, 3]
    assert restored.resolve_converted_layers(4) == (1, 3)
    assert restored.resolved_strategies.selection["policy"] == "forced_anchor"
    assert restored.base_model_revision == "pinned-revision"


@pytest.mark.parametrize(
    ("converted_layers", "exception"),
    [
        ([], ValueError),
        ([1, 1], ValueError),
        ([True], TypeError),
        ("lower_half", ValueError),
    ],
)
def test_invalid_layer_selectors_are_rejected(
    converted_layers: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        AnchoredMoEConfig(converted_layers=converted_layers)  # type: ignore[arg-type]


def test_out_of_range_layer_is_rejected_when_model_depth_is_known() -> None:
    config = AnchoredMoEConfig(converted_layers=[4])

    with pytest.raises(ValueError, match="outside the model depth"):
        config.resolve_converted_layers(4)


def test_unimplemented_strategy_cannot_silently_fall_back() -> None:
    with pytest.raises(ValueError, match="supports only 'forced_anchor'"):
        AnchoredMoEConfig(
            strategy_config={"selection": {"policy": "anchored_top1"}}
        )
