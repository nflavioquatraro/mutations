"""Acceptance tests for the RTR-003 logical append-only router."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    LogicalAppendOnlyRouter,
    RouterScoreBank,
    compile_logical_router,
)


def _head(
    *,
    layer_index: int,
    expert_id: str,
    weight: tuple[float, float, float],
    bias: float,
    temperature: float = 1.0,
    threshold: float = 0.0,
) -> AffineRouterHead:
    head = AffineRouterHead(
        3,
        layer_index=layer_index,
        expert_id=expert_id,
        temperature=temperature,
        threshold=threshold,
    )
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([weight]))
        head.linear.bias.fill_(bias)
    return head


def _independent_bank() -> AppendOnlyRouterHeads:
    bank = AppendOnlyRouterHeads(3)
    bank.append(
        _head(
            layer_index=2,
            expert_id="code:0",
            weight=(0.5, -0.25, 0.75),
            bias=0.1,
            temperature=2.0,
            threshold=0.2,
        )
    )
    bank.append(
        _head(
            layer_index=3,
            expert_id="code:0",
            weight=(0.25, 0.5, -0.5),
            bias=-0.2,
        )
    )
    bank.append(
        _head(
            layer_index=2,
            expert_id="math:0",
            weight=(-0.5, 0.75, 0.25),
            bias=0.3,
            temperature=0.5,
            threshold=-0.1,
        )
    )
    return bank


def test_compile_preserves_scores_ownership_and_frozen_matrix_shapes() -> None:
    independent = _independent_bank()
    logical = compile_logical_router(independent)
    features = torch.tensor(
        [[[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]]],
        requires_grad=True,
    )

    scores = logical(features, layer_index=2)

    assert isinstance(logical, RouterScoreBank)
    assert logical.architecture == "logical_append_only_columns"
    assert logical.head_keys == independent.head_keys
    assert logical.expert_ids_for_layer(2) == ("code:0", "math:0")
    assert logical.expert_ids_for_layer(3) == ("code:0",)
    assert torch.equal(scores, independent(features, layer_index=2))
    assert logical.layers["2"].weight.shape == (2, 3)
    assert logical.layers["2"].bias.shape == (2,)
    assert logical.layers["2"].temperature.shape == (2,)
    assert logical.layers["2"].threshold.shape == (2,)
    assert tuple(logical.parameters()) == ()

    scores.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_compile_copies_source_state_without_storage_aliases() -> None:
    independent = _independent_bank()
    logical = LogicalAppendOnlyRouter.from_independent_heads(independent)
    features = torch.randn(2, 3)
    expected = logical(features, layer_index=2).clone()
    compiled_pointers = {
        value.data_ptr() for value in logical.state_dict().values()
    }
    source_pointers = {
        value.data_ptr() for value in independent.state_dict().values()
    }

    with torch.no_grad():
        independent.heads[0].linear.weight.add_(100.0)
        independent.heads[2].linear.bias.fill_(-100.0)

    assert compiled_pointers.isdisjoint(source_pointers)
    assert torch.equal(logical(features, layer_index=2), expected)


def test_append_preserves_incumbent_scores_and_global_filtered_order() -> None:
    logical = LogicalAppendOnlyRouter(3)
    logical.append(
        _head(
            layer_index=2,
            expert_id="code:0",
            weight=(0.5, -0.25, 0.75),
            bias=0.1,
        )
    )
    features = torch.randn(4, 3)
    incumbent = logical(features, layer_index=2)

    logical.append(
        _head(
            layer_index=4,
            expert_id="code:0",
            weight=(0.1, 0.2, 0.3),
            bias=0.0,
        )
    )
    logical.append(
        _head(
            layer_index=2,
            expert_id="math:0",
            weight=(-0.5, 0.75, 0.25),
            bias=0.3,
        )
    )
    appended = logical(features, layer_index=2)

    assert torch.equal(appended[..., :1], incumbent)
    assert logical.head_keys == (
        (2, "code:0"),
        (4, "code:0"),
        (2, "math:0"),
    )
    assert logical.expert_ids_for_layer(2) == ("code:0", "math:0")


def test_logical_router_moves_buffers_and_preserves_feature_gradients() -> None:
    logical = compile_logical_router(_independent_bank()).to(dtype=torch.float64)
    features = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    scores = logical(features, layer_index=3)
    scores.sum().backward()

    assert scores.dtype == torch.float64
    assert features.grad is not None
    assert features.grad.dtype == torch.float64
    assert all(value.dtype == torch.float64 for value in logical.buffers())


def test_logical_router_rejects_invalid_heads_features_and_layers() -> None:
    logical = LogicalAppendOnlyRouter(3)
    head = _head(
        layer_index=2,
        expert_id="code:0",
        weight=(0.5, -0.25, 0.75),
        bias=0.1,
    )
    logical.append(head)

    with pytest.raises(TypeError, match="AffineRouterHead"):
        logical.append(torch.nn.Linear(3, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already compiled"):
        logical.append(head)
    with pytest.raises(ValueError, match="feature size"):
        logical.append(
            AffineRouterHead(4, layer_index=2, expert_id="math:0")
        )
    with pytest.raises(ValueError, match="one dtype"):
        logical.append(
            _head(
                layer_index=2,
                expert_id="math:0",
                weight=(0.1, 0.2, 0.3),
                bias=0.0,
            ).to(dtype=torch.float64)
        )

    nonfinite = _head(
        layer_index=2,
        expert_id="science:0",
        weight=(0.1, 0.2, 0.3),
        bias=0.0,
    )
    with torch.no_grad():
        nonfinite.linear.weight[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        logical.append(nonfinite)

    with pytest.raises(LookupError, match="layer 5"):
        logical(torch.randn(2, 3), layer_index=5)
    with pytest.raises(TypeError, match="floating-point"):
        logical(torch.ones(2, 3, dtype=torch.int64), layer_index=2)
    with pytest.raises(ValueError, match="final dimension"):
        logical(torch.randn(2, 4), layer_index=2)
    with pytest.raises(ValueError, match="dtype"):
        logical(torch.randn(2, 3, dtype=torch.float64), layer_index=2)


def test_empty_compilation_and_invalid_construction_are_explicit() -> None:
    empty = AppendOnlyRouterHeads(3)
    logical = compile_logical_router(empty)

    assert len(logical) == 0
    assert logical.head_keys == ()
    assert logical.expert_ids_for_layer(2) == ()

    with pytest.raises(TypeError, match="AppendOnlyRouterHeads"):
        compile_logical_router(torch.nn.Linear(3, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        LogicalAppendOnlyRouter(0)
    with pytest.raises(TypeError, match="integer"):
        LogicalAppendOnlyRouter(True)  # type: ignore[arg-type]
