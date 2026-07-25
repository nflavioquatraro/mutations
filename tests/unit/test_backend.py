"""Tests for Qwen3 discovery and safe MLP replacement."""

from __future__ import annotations

import pytest
from torch import nn
from transformers import Qwen3ForCausalLM, Qwen3Model

from anchored_moe import AnchoredMoEConfig, RoutedMLP
from anchored_moe.backends import Qwen3Backend


def test_backend_discovers_qwen3_mlp_modules_in_layer_order(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    references = tuple(Qwen3Backend().iter_mlp_references(tiny_qwen3_model))

    assert tuple(reference.layer_index for reference in references) == (0, 1, 2, 3)
    assert all(reference.module is reference.parent.mlp for reference in references)


def test_backend_supports_bare_qwen3_model(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    bare_model = Qwen3Model(tiny_qwen3_model.config)
    backend = Qwen3Backend()

    assert backend.supports_model(bare_model)
    assert len(tuple(backend.iter_mlp_references(bare_model))) == 4


def test_conversion_wraps_only_selected_layers_and_freezes_model(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    layers = tiny_qwen3_model.model.layers
    originals = tuple(layer.mlp for layer in layers)

    report = Qwen3Backend().convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[1, 3]),
    )

    assert report.converted_layers == (1, 3)
    assert layers[0].mlp is originals[0]
    assert isinstance(layers[1].mlp, RoutedMLP)
    assert layers[1].mlp.anchor_mlp is originals[1]
    assert layers[2].mlp is originals[2]
    assert isinstance(layers[3].mlp, RoutedMLP)
    assert layers[3].mlp.anchor_mlp is originals[3]
    assert all(not parameter.requires_grad for parameter in tiny_qwen3_model.parameters())


def test_double_conversion_is_rejected(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend = Qwen3Backend()
    config = AnchoredMoEConfig(converted_layers=[2])
    backend.convert(tiny_qwen3_model, config)

    with pytest.raises(RuntimeError, match="already routed"):
        backend.convert(tiny_qwen3_model, config)


def test_stale_reference_replacement_is_rejected(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    backend = Qwen3Backend()
    reference = next(backend.iter_mlp_references(tiny_qwen3_model))
    reference.parent.mlp = nn.Identity()

    with pytest.raises(RuntimeError, match="changed after discovery"):
        backend.replace_mlp(reference, nn.Identity())
