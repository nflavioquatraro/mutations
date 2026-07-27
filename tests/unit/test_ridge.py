"""Acceptance tests for CAL-004 independent ridge fitting."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    RidgeFittingError,
    RouterFeatureBank,
    UtilityLabelBank,
    fit_independent_ridge,
)


def _feature_bank(
    features_by_layer: dict[int, torch.Tensor] | None = None,
) -> RouterFeatureBank:
    features = features_by_layer or {
        2: torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        )
    }
    return RouterFeatureBank(
        features,
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 2,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "synthetic-ridge-reference",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=53,
    )


def _utility_bank(
    feature_bank: RouterFeatureBank,
    utilities_by_layer: dict[int, torch.Tensor],
    *,
    expert_ids_by_layer: dict[int, tuple[str, ...]] | None = None,
) -> UtilityLabelBank:
    expert_ids = expert_ids_by_layer or {
        layer_index: ("code:0", "math:0")
        for layer_index in feature_bank.layer_indices
    }
    return UtilityLabelBank(
        utilities_by_layer,
        expert_ids_by_layer=expert_ids,
        row_ids=feature_bank.row_ids,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def test_independent_ridge_matches_centered_closed_form() -> None:
    feature_bank = _feature_bank()
    features = feature_bank.features_for_layer(2)
    true_weights = torch.tensor([[3.0, -1.0], [1.0, 2.0]])
    true_bias = torch.tensor([0.5, -0.25])
    utilities = features @ true_weights + true_bias
    labels = _utility_bank(feature_bank, {2: utilities})

    router = fit_independent_ridge(
        feature_bank,
        labels,
        regularization=1.0,
    )

    # X is centered and X^T X = 2I, so lambda=1 shrinks each slope by 2/3.
    expected_weights = true_weights * (2.0 / 3.0)
    assert router.head_keys == ((2, "code:0"), (2, "math:0"))
    assert torch.allclose(
        router.heads[0].linear.weight,
        expected_weights[:, 0].unsqueeze(0),
    )
    assert torch.allclose(
        router.heads[1].linear.weight,
        expected_weights[:, 1].unsqueeze(0),
    )
    assert torch.equal(router.heads[0].linear.bias, true_bias[0].reshape(1))
    assert torch.equal(router.heads[1].linear.bias, true_bias[1].reshape(1))
    assert all(not parameter.requires_grad for parameter in router.parameters())


def test_ridge_preserves_layer_and_expert_order_without_mutating_banks() -> None:
    feature_bank = _feature_bank(
        {
            3: torch.tensor(
                [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
            ),
            1: torch.tensor(
                [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]]
            ),
        }
    )
    utilities = {
        layer_index: torch.tensor(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
                [0.5, -0.5],
                [-0.5, 0.5],
            ]
        )
        for layer_index in feature_bank.layer_indices
    }
    labels = _utility_bank(
        feature_bank,
        utilities,
        expert_ids_by_layer={
            1: ("math:0", "code:0"),
            3: ("code:1", "code:0"),
        },
    )
    original_features = {
        layer: feature_bank.features_for_layer(layer)
        for layer in feature_bank.layer_indices
    }
    original_utilities = {
        layer: labels.utilities_for_layer(layer)
        for layer in labels.layer_indices
    }

    first = fit_independent_ridge(feature_bank, labels, regularization=0.25)
    second = fit_independent_ridge(feature_bank, labels, regularization=0.25)

    assert first.head_keys == (
        (1, "math:0"),
        (1, "code:0"),
        (3, "code:1"),
        (3, "code:0"),
    )
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name])
    for layer_index in feature_bank.layer_indices:
        assert torch.equal(
            feature_bank.features_for_layer(layer_index),
            original_features[layer_index],
        )
        assert torch.equal(
            labels.utilities_for_layer(layer_index),
            original_utilities[layer_index],
        )


def test_ridge_fits_intercept_when_features_are_rank_deficient() -> None:
    feature_bank = _feature_bank(
        {0: torch.tensor([[2.0, -3.0], [2.0, -3.0], [2.0, -3.0]])}
    )
    labels = _utility_bank(
        feature_bank,
        {0: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
    )

    router = fit_independent_ridge(
        feature_bank,
        labels,
        regularization=1e-6,
    )

    assert torch.equal(router.heads[0].linear.weight, torch.zeros(1, 2))
    assert torch.equal(router.heads[1].linear.weight, torch.zeros(1, 2))
    assert torch.equal(router.heads[0].linear.bias, torch.tensor([3.0]))
    assert torch.equal(router.heads[1].linear.bias, torch.tensor([4.0]))


@pytest.mark.parametrize(
    ("regularization", "error_type"),
    [
        (True, TypeError),
        ("0.1", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_ridge_rejects_invalid_regularization(
    regularization: object,
    error_type: type[Exception],
) -> None:
    feature_bank = _feature_bank()
    labels = _utility_bank(
        feature_bank,
        {2: torch.zeros(4, 2)},
    )

    with pytest.raises(error_type):
        fit_independent_ridge(
            feature_bank,
            labels,
            regularization=regularization,  # type: ignore[arg-type]
        )


def test_ridge_rejects_wrong_types_and_misaligned_banks() -> None:
    feature_bank = _feature_bank()
    labels = _utility_bank(feature_bank, {2: torch.zeros(4, 2)})
    other_bank = _feature_bank(
        {
            2: torch.tensor(
                [[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]]
            )
        }
    )

    with pytest.raises(TypeError, match="RouterFeatureBank"):
        fit_independent_ridge(object(), labels)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="UtilityLabelBank"):
        fit_independent_ridge(feature_bank, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different router feature bank"):
        fit_independent_ridge(other_bank, labels)


def test_ridge_reports_parameters_that_overflow_feature_dtype() -> None:
    feature_bank = RouterFeatureBank(
        {0: torch.tensor([[0.0], [0.001]], dtype=torch.float16)},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 1,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "synthetic-ridge-overflow",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
    )
    labels = UtilityLabelBank(
        {0: torch.tensor([[0.0], [60000.0]], dtype=torch.float16)},
        expert_ids_by_layer={0: ("code:0",)},
        row_ids=feature_bank.row_ids,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )

    with pytest.raises(RidgeFittingError, match="not finite"):
        fit_independent_ridge(
            feature_bank,
            labels,
            regularization=1e-12,
        )
