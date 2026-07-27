"""Acceptance tests for score-softmax activation combination."""

from __future__ import annotations

import math

import pytest
import torch

from anchored_moe import (
    ActivationCombinationResult,
    CombinationPolicy,
    ScoreSoftmaxCombination,
)


def test_score_softmax_combines_selected_outputs_and_reports_weights() -> None:
    selected_scores = torch.tensor(
        [[0.0, 0.0], [math.log(3.0), 0.0], [-1000.0, 1000.0]],
        dtype=torch.float64,
    )
    selected_outputs = torch.tensor(
        [
            [[2.0, 4.0], [6.0, 8.0]],
            [[4.0, 8.0], [0.0, 4.0]],
            [[1.0, 2.0], [9.0, 10.0]],
        ],
        dtype=torch.float64,
    )

    result = ScoreSoftmaxCombination()(
        selected_outputs,
        selected_scores=selected_scores,
    )

    assert isinstance(ScoreSoftmaxCombination(), CombinationPolicy)
    assert torch.allclose(
        result.weights,
        torch.tensor(
            [[0.5, 0.5], [0.75, 0.25], [0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        result.output,
        torch.tensor(
            [[4.0, 6.0], [3.0, 7.0], [9.0, 10.0]],
            dtype=torch.float64,
        ),
    )


def test_mix_temperature_controls_weight_sharpness() -> None:
    scores = torch.tensor([[2.0, 0.0]])
    outputs = torch.tensor([[[1.0], [3.0]]])

    cold = ScoreSoftmaxCombination(mix_temperature=0.5)(
        outputs,
        selected_scores=scores,
    )
    warm = ScoreSoftmaxCombination(mix_temperature=2.0)(
        outputs,
        selected_scores=scores,
    )

    assert cold.weights[0, 0] > warm.weights[0, 0]
    assert cold.output[0, 0] < warm.output[0, 0]


def test_combination_preserves_leading_and_trailing_shapes_dtype_and_device() -> None:
    scores = torch.randn(2, 3, 2, dtype=torch.float64)
    outputs = torch.randn(2, 3, 2, 4, 5, dtype=torch.float64)

    result = ScoreSoftmaxCombination().combine(
        outputs,
        selected_scores=scores,
    )

    assert result.weights.shape == (2, 3, 2)
    assert result.output.shape == (2, 3, 4, 5)
    assert result.weights.dtype == outputs.dtype
    assert result.output.dtype == outputs.dtype
    assert result.weights.device == outputs.device
    assert result.output.device == outputs.device
    assert torch.allclose(
        result.weights.sum(dim=-1),
        torch.ones(2, 3, dtype=torch.float64),
    )


def test_combination_is_differentiable_and_does_not_mutate_inputs() -> None:
    scores = torch.tensor(
        [[0.4, 0.2], [0.1, 0.8]],
        requires_grad=True,
    )
    outputs = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0]], [[2.0, 4.0], [7.0, 11.0]]],
        requires_grad=True,
    )
    original_scores = scores.detach().clone()
    original_outputs = outputs.detach().clone()

    result = ScoreSoftmaxCombination()(outputs, selected_scores=scores)
    result.output.sum().backward()

    assert torch.equal(scores.detach(), original_scores)
    assert torch.equal(outputs.detach(), original_outputs)
    assert scores.grad is not None
    assert outputs.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(outputs.grad).all()


@pytest.mark.parametrize(
    "temperature",
    [True, "1.0", 0.0, -1.0, float("nan"), float("inf")],
)
def test_combination_rejects_invalid_temperatures(temperature: object) -> None:
    with pytest.raises((TypeError, ValueError), match="mix_temperature"):
        ScoreSoftmaxCombination(temperature)  # type: ignore[arg-type]


def test_combination_rejects_inconsistent_inputs() -> None:
    policy = ScoreSoftmaxCombination()
    outputs = torch.randn(2, 2, 3)
    scores = torch.randn(2, 2)

    with pytest.raises(TypeError, match="selected_outputs"):
        policy([[1.0]], selected_scores=scores)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        policy(outputs.to(torch.int64), selected_scores=scores)
    with pytest.raises(TypeError, match="selected_scores"):
        policy(outputs, selected_scores=[[0.0, 1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size two"):
        policy(torch.randn(2, 3, 4), selected_scores=torch.randn(2, 3))
    with pytest.raises(ValueError, match="trailing output"):
        policy(scores, selected_scores=scores)
    with pytest.raises(ValueError, match="begin with"):
        policy(torch.randn(3, 2, 4), selected_scores=scores)
    with pytest.raises(ValueError, match="one dtype"):
        policy(outputs.double(), selected_scores=scores)
    with pytest.raises(ValueError, match="finite"):
        policy(outputs, selected_scores=torch.tensor([[float("nan"), 0.0]]).expand(2, 2))


def test_activation_combination_result_rejects_invalid_public_construction() -> None:
    with pytest.raises(ValueError, match="size two"):
        ActivationCombinationResult(
            output=torch.ones(2, 3),
            weights=torch.ones(2, 3),
        )
    with pytest.raises(ValueError, match="token-leading"):
        ActivationCombinationResult(
            output=torch.ones(3, 4),
            weights=torch.full((2, 2), 0.5),
        )
    with pytest.raises(ValueError, match="sum to one"):
        ActivationCombinationResult(
            output=torch.ones(2, 4),
            weights=torch.full((2, 2), 0.25),
        )
