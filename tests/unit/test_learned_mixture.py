"""Acceptance tests for COM-003 learned mixture-weight fitting."""

from __future__ import annotations

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from anchored_moe import (
    CombinationPolicy,
    LearnedMixtureSerializationError,
    LearnedMixtureWeightCombination,
    LearnedMixtureWeightHead,
    MixtureWeightFittingConfig,
    MixtureWeightFittingReport,
    fit_learned_mixture_head,
    plan_expert_pairs,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.tensor(
        [
            [-2.0, 1.0],
            [-1.5, 0.5],
            [-1.0, -0.5],
            [-0.5, 1.5],
            [0.0, 0.0],
            [0.5, -1.0],
            [1.0, 0.5],
            [1.5, -0.5],
            [2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    target_first = torch.sigmoid(
        1.5 * scores[:, 0] - 0.75 * scores[:, 1] + 0.3
    )
    target_weights = torch.stack((target_first, 1.0 - target_first), dim=-1)
    pair_indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    return scores, target_weights, pair_indices


def test_fitting_learns_oracle_weights_and_returns_frozen_head() -> None:
    plan = plan_expert_pairs(("code", "math", "bio"), max_pairs=None)
    scores, target_weights, pair_indices = _training_data()

    head, report = fit_learned_mixture_head(
        plan,
        selected_scores=scores,
        target_weights=target_weights,
        pair_indices=pair_indices,
        config=MixtureWeightFittingConfig(
            learning_rate=0.1,
            optimization_steps=400,
            l2_weight=0.0,
        ),
    )

    assert isinstance(head, LearnedMixtureWeightHead)
    assert tuple(head.parameters()) == ()
    assert head.training is False
    assert head.weight.dtype == scores.dtype
    assert head.weight.device.type == "cpu"
    assert report.row_count == scores.shape[0]
    assert report.pair_counts == (3, 3, 3)
    assert report.final_cross_entropy < report.initial_cross_entropy
    assert report.final_mean_absolute_error < 0.01
    assert torch.allclose(head(scores), target_weights, atol=0.02, rtol=0.0)


def test_fitting_is_deterministic_and_inputs_and_reports_are_stable() -> None:
    plan = plan_expert_pairs(("code", "math", "bio"), max_pairs=None)
    scores, target_weights, pair_indices = _training_data()
    original_scores = scores.clone()
    original_targets = target_weights.clone()
    original_indices = pair_indices.clone()
    config = MixtureWeightFittingConfig(optimization_steps=80)

    first, first_report = fit_learned_mixture_head(
        plan,
        selected_scores=scores,
        target_weights=target_weights,
        pair_indices=pair_indices,
        config=config,
    )
    second, second_report = fit_learned_mixture_head(
        plan,
        selected_scores=scores,
        target_weights=target_weights,
        pair_indices=pair_indices,
        config=config,
    )

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)
    assert first_report == second_report
    assert MixtureWeightFittingReport.from_dict(first_report.to_dict()) == first_report
    json.dumps(first_report.to_dict(), allow_nan=False)
    assert torch.equal(scores, original_scores)
    assert torch.equal(target_weights, original_targets)
    assert torch.equal(pair_indices, original_indices)


def test_learned_combination_uses_separate_logits_and_preserves_gradients() -> None:
    head = LearnedMixtureWeightHead(
        weight=torch.zeros(2),
        bias=torch.tensor(math.log(3.0)),
    )
    policy = LearnedMixtureWeightCombination(head)
    scores = torch.tensor([[9.0, 1.0], [1.0, 9.0]], requires_grad=True)
    outputs = torch.tensor(
        [[[2.0, 4.0], [-2.0, -4.0]], [[3.0, 6.0], [-3.0, -6.0]]],
        requires_grad=True,
    )

    result = policy(outputs, selected_scores=scores)
    (result.output.square().sum() + result.weights.square().sum()).backward()

    assert isinstance(policy, CombinationPolicy)
    assert policy.policy_name == "learned_mixture_weight"
    assert torch.allclose(
        result.weights,
        torch.tensor([[0.75, 0.25], [0.75, 0.25]]),
    )
    assert torch.allclose(
        result.output,
        torch.tensor([[1.0, 2.0], [1.5, 3.0]]),
    )
    assert scores.grad is not None
    assert outputs.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(outputs.grad).all()


def test_learned_head_safe_serialization_round_trip_is_deterministic(tmp_path) -> None:
    head = LearnedMixtureWeightHead(
        weight=torch.tensor([1.25, -0.5], dtype=torch.float64),
        bias=torch.tensor(0.75, dtype=torch.float64),
    )
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    head.save_safetensors(first_path)
    head.save_safetensors(second_path)

    restored = LearnedMixtureWeightHead.from_safetensors(first_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert tuple(restored.parameters()) == ()
    assert restored.training is False
    assert torch.equal(restored.weight, head.weight)
    assert torch.equal(restored.bias, head.bias)
    scores = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    assert torch.equal(restored(scores), head(scores))
    with pytest.raises(FileExistsError):
        head.save_safetensors(first_path)
    with pytest.raises(LearnedMixtureSerializationError, match="end"):
        head.save_safetensors(tmp_path / "head.bin")


def test_learned_head_rejects_incompatible_safe_payload(tmp_path) -> None:
    path = tmp_path / "invalid.safetensors"
    save_file(
        {
            "weight": torch.ones(2),
            "bias": torch.zeros(()),
        },
        path,
        metadata={"anchored_moe": json.dumps({"format": "wrong"})},
    )

    with pytest.raises(LearnedMixtureSerializationError, match="metadata"):
        LearnedMixtureWeightHead.from_safetensors(path)


@pytest.mark.parametrize(
    ("change", "error_type", "message"),
    [
        ("score_shape", ValueError, "shape"),
        ("target_sum", ValueError, "sum to one"),
        ("target_dtype", ValueError, "dtype"),
        ("index_dtype", TypeError, "torch.long"),
        ("index_range", ValueError, "outside"),
    ],
)
def test_fitting_rejects_misaligned_or_invalid_training_data(
    change: str,
    error_type: type[Exception],
    message: str,
) -> None:
    plan = plan_expert_pairs(("code", "math", "bio"), max_pairs=None)
    scores, targets, indices = _training_data()
    if change == "score_shape":
        scores = scores[:, :1]
    elif change == "target_sum":
        targets[0] = torch.tensor([0.2, 0.2])
    elif change == "target_dtype":
        targets = targets.to(dtype=torch.float64)
    elif change == "index_dtype":
        indices = indices.to(dtype=torch.int32)
    elif change == "index_range":
        indices[0] = plan.pair_count

    with pytest.raises(error_type, match=message):
        fit_learned_mixture_head(
            plan,
            selected_scores=scores,
            target_weights=targets,
            pair_indices=indices,
        )


@pytest.mark.parametrize(
    ("weight", "bias", "message"),
    [
        (torch.ones(3), torch.zeros(()), "shape"),
        (torch.ones(2), torch.zeros(1), "scalar"),
        (torch.tensor([1.0, float("nan")]), torch.zeros(()), "finite"),
    ],
)
def test_learned_head_rejects_invalid_coefficients(
    weight: torch.Tensor,
    bias: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LearnedMixtureWeightHead(weight=weight, bias=bias)
