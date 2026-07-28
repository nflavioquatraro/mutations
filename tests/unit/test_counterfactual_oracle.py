"""Acceptance tests for CAL-007 counterfactual oracle and router regret."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    CounterfactualRoute,
    CounterfactualRouteLosses,
    evaluate_counterfactual_oracle,
)


def _routes() -> tuple[CounterfactualRoute, ...]:
    """Return the canonical two-expert candidate order."""

    return (
        CounterfactualRoute.anchor(),
        CounterfactualRoute.single("code"),
        CounterfactualRoute.single("math"),
        CounterfactualRoute.pair("code", "math"),
    )


def test_counterfactual_oracle_uses_stable_ties_and_reports_regret() -> None:
    losses = CounterfactualRouteLosses(
        torch.tensor(
            [
                [1.0, 0.5, 0.7, 0.4],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 1.5, 0.8, 1.0],
            ]
        ),
        expert_ids=("code", "math"),
        routes=_routes(),
        loss_name="cross_entropy",
    )

    result = evaluate_counterfactual_oracle(
        losses,
        torch.tensor([1, 3, 2]),
    )

    assert torch.equal(result.oracle_candidate_indices, torch.tensor([3, 0, 2]))
    assert torch.allclose(result.oracle_losses, torch.tensor([0.4, 1.0, 0.8]))
    assert torch.allclose(result.selected_losses, torch.tensor([0.5, 1.0, 0.8]))
    assert torch.allclose(result.regret, torch.tensor([0.1, 0.0, 0.0]))
    assert torch.equal(result.exact_oracle_mask, torch.tensor([False, True, True]))
    assert result.mean_regret == pytest.approx(1.0 / 30.0)
    assert result.oracle_agreement_fraction == pytest.approx(1.0 / 3.0)


def test_counterfactual_loss_table_is_detached_cpu_and_defensive() -> None:
    source = torch.tensor(
        [[1.0, 0.8, 0.9, 0.7]],
        requires_grad=True,
    )
    table = CounterfactualRouteLosses(
        source,
        expert_ids=("code", "math"),
        routes=_routes(),
        loss_name="cross_entropy",
    )

    source.detach()[0, 0] = 99.0
    first_read = table.losses
    first_read[0, 1] = 88.0

    assert table.losses.device.type == "cpu"
    assert table.losses.requires_grad is False
    assert torch.equal(table.losses, torch.tensor([[1.0, 0.8, 0.9, 0.7]]))
    assert table.token_shape == torch.Size([1])
    assert table.token_count == 1
    assert table.candidate_count == 4
    assert table.pair_count == 1
    assert table.index_for_route(CounterfactualRoute.single("math")) == 2


@pytest.mark.parametrize(
    ("routes", "message"),
    [
        (
            (
                CounterfactualRoute.single("code"),
                CounterfactualRoute.anchor(),
                CounterfactualRoute.single("math"),
                CounterfactualRoute.pair("code", "math"),
            ),
            "begin with the anchor",
        ),
        (
            (
                CounterfactualRoute.anchor(),
                CounterfactualRoute.single("code"),
                CounterfactualRoute.single("math"),
                CounterfactualRoute.pair("math", "code"),
            ),
            "ascending registry order",
        ),
    ],
)
def test_counterfactual_loss_table_requires_canonical_routes(
    routes: tuple[CounterfactualRoute, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CounterfactualRouteLosses(
            torch.ones(2, 4),
            expert_ids=("code", "math"),
            routes=routes,
            loss_name="cross_entropy",
        )


def test_counterfactual_oracle_rejects_invalid_losses_and_selection() -> None:
    with pytest.raises(ValueError, match="finite"):
        CounterfactualRouteLosses(
            torch.tensor([[1.0, 0.8, 0.9, float("nan")]]),
            expert_ids=("code", "math"),
            routes=_routes(),
            loss_name="cross_entropy",
        )

    table = CounterfactualRouteLosses(
        torch.ones(2, 4),
        expert_ids=("code", "math"),
        routes=_routes(),
        loss_name="cross_entropy",
    )
    with pytest.raises(TypeError, match="torch.long"):
        evaluate_counterfactual_oracle(table, torch.zeros(2))
    with pytest.raises(ValueError, match="token shape"):
        evaluate_counterfactual_oracle(table, torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="unevaluated"):
        evaluate_counterfactual_oracle(table, torch.tensor([0, 4]))


def test_counterfactual_route_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="at most two"):
        CounterfactualRoute(("code", "math", "bio"))
    with pytest.raises(ValueError, match="repeat"):
        CounterfactualRoute(("code", "code"))
    with pytest.raises(KeyError, match="do not contain route"):
        CounterfactualRouteLosses(
            torch.ones(1, 4),
            expert_ids=("code", "math"),
            routes=_routes(),
            loss_name="cross_entropy",
        ).index_for_route(CounterfactualRoute.single("bio"))
