"""Deterministic independent ridge fitting for affine router heads.

References:
    Hoerl and Kennard, "Ridge Regression: Biased Estimation for
    Nonorthogonal Problems." https://doi.org/10.1080/00401706.1970.10488634

Differences from the paper:
    CAL-004 fits all expert targets at one layer in a deterministic
    multi-output solve, leaves the intercept unregularized, and emits
    independently owned frozen router heads rather than a general regression
    estimator.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from anchored_moe.calibration.features import RouterFeatureBank
from anchored_moe.calibration.labels import UtilityLabelBank
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads


class RidgeFittingError(ValueError):
    """Raised when aligned calibration data cannot produce finite router heads."""


def fit_independent_ridge(
    feature_bank: RouterFeatureBank,
    utility_labels: UtilityLabelBank,
    *,
    regularization: float = 1e-3,
) -> AppendOnlyRouterHeads:
    """Fit one frozen affine router head per layer and expert.

    For layer design matrix ``X_l in R^(N x D)`` and individual-utility
    targets ``Y_l in R^(N x K_l)``, the function centers both matrices and
    solves

    ``W_l = (X_c^T X_c + lambda I)^(-1) X_c^T Y_c``

    followed by the unregularized intercept

    ``b_l = mean(Y_l) - mean(X_l) W_l``.

    Centering is algebraically equivalent to adding an unregularized constant
    column, while the strictly positive ``regularization`` makes the feature
    system positive definite even when cached features are rank deficient.
    The solve always runs on CPU in ``torch.float64``. Fitted parameters are
    converted to the feature-bank dtype and copied into frozen
    :class:`~anchored_moe.routers.AffineRouterHead` objects with temperature
    one and threshold zero.

    Heads are appended in ascending layer order and then in the exact expert
    column order declared by ``utility_labels``. Inputs are accessed through
    defensive copies and are never mutated.

    Args:
        feature_bank:
            Immutable CAL-001 feature bank containing one ``[N, D]`` design
            matrix per layer.
        utility_labels:
            CAL-002 individual-utility targets with matrices ``[N, K_l]``.
            Positive targets mean that an expert improved over the incumbent.
        regularization:
            Finite positive ridge penalty ``lambda`` applied to feature
            weights only.

    Returns:
        An :class:`~anchored_moe.routers.AppendOnlyRouterHeads` bank containing
        frozen, independently owned affine heads.

    Raises:
        TypeError:
            If either bank has the wrong type or ``regularization`` is not a
            real number.
        ValueError:
            If the banks are not exactly aligned or regularization is invalid.
        RidgeFittingError:
            If the solve fails or fitted parameters cannot be represented as
            finite values in the feature-bank dtype.

    References:
        Hoerl and Kennard, "Ridge Regression: Biased Estimation for
        Nonorthogonal Problems." https://doi.org/10.1080/00401706.1970.10488634

    Differences from the paper:
        The intercept is not regularized, layer targets are solved together
        for reproducibility, and the result is emitted as immutable deployed
        router heads rather than exposed as a mutable regression estimator.
    """

    if not isinstance(feature_bank, RouterFeatureBank):
        raise TypeError("feature_bank must be a RouterFeatureBank.")
    if not isinstance(utility_labels, UtilityLabelBank):
        raise TypeError("utility_labels must be a UtilityLabelBank.")
    penalty = _validate_regularization(regularization)
    utility_labels.assert_aligned(feature_bank)

    router = AppendOnlyRouterHeads(feature_size=feature_bank.feature_size)
    for layer_index in feature_bank.layer_indices:
        features = feature_bank.features_for_layer(layer_index).to(
            device="cpu",
            dtype=torch.float64,
        )
        targets = utility_labels.utilities_for_layer(layer_index).to(
            device="cpu",
            dtype=torch.float64,
        )
        weights, biases = _solve_centered_ridge(
            features,
            targets,
            regularization=penalty,
            layer_index=layer_index,
        )
        expert_ids = utility_labels.expert_ids_for_layer(layer_index)
        for expert_column, expert_id in enumerate(expert_ids):
            head = AffineRouterHead(
                feature_bank.feature_size,
                layer_index=layer_index,
                expert_id=expert_id,
                trainable=False,
            ).to(device="cpu", dtype=feature_bank.dtype)
            weight = weights[:, expert_column].to(dtype=feature_bank.dtype)
            bias = biases[expert_column].to(dtype=feature_bank.dtype)
            if not torch.isfinite(weight).all().item() or not torch.isfinite(
                bias
            ).item():
                raise RidgeFittingError(
                    "Fitted ridge parameters for layer "
                    f"{layer_index}, expert {expert_id!r} are not finite in "
                    f"{feature_bank.dtype}."
                )
            with torch.no_grad():
                head.linear.weight.copy_(weight.unsqueeze(0))
                head.linear.bias.copy_(bias.reshape(1))
            router.append(head)

    return router


def _solve_centered_ridge(
    features: Tensor,
    targets: Tensor,
    *,
    regularization: float,
    layer_index: int,
) -> tuple[Tensor, Tensor]:
    """Solve one layer in float64 while leaving the intercept unregularized."""

    feature_mean = features.mean(dim=0)
    target_mean = targets.mean(dim=0)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    feature_size = features.shape[1]
    gram = centered_features.T @ centered_features
    # Regularizing only the D x D feature system is equivalent to an augmented
    # design with a zero penalty on its constant intercept column.
    system = gram + regularization * torch.eye(
        feature_size,
        dtype=torch.float64,
        device="cpu",
    )
    right_hand_side = centered_features.T @ centered_targets
    try:
        weights = torch.linalg.solve(system, right_hand_side)
    except RuntimeError as error:
        raise RidgeFittingError(
            f"Ridge solve failed for layer {layer_index}: {error}"
        ) from error
    biases = target_mean - feature_mean @ weights
    if not torch.isfinite(weights).all().item() or not torch.isfinite(
        biases
    ).all().item():
        raise RidgeFittingError(
            f"Ridge solve produced non-finite parameters for layer {layer_index}."
        )
    return weights, biases


def _validate_regularization(value: Any) -> float:
    """Return a finite, strictly positive ridge penalty."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("regularization must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("regularization must be finite and strictly positive.")
    return normalized
