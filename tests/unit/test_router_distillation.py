"""Acceptance tests for RTR-007 hard-route router distillation."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    HardRouteDistillationConfig,
    LogicalAppendOnlyRouter,
    RouterDistillationReport,
    RouterFeatureBank,
    RouterScoreBank,
    compile_logical_router,
    distill_frozen_router,
)


def _feature_bank(*, feature_size: int = 2) -> RouterFeatureBank:
    layer_2 = torch.tensor(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    layer_4 = torch.tensor(
        [
            [0.0, -2.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ]
    )
    if feature_size != 2:
        layer_2 = torch.zeros(5, feature_size)
        layer_4 = torch.zeros(5, feature_size)
    return RouterFeatureBank(
        {2: layer_2, 4: layer_4},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": feature_size,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "rtr-007-fixture",
            "filters": {},
            "revision": "local",
            "split": "calibration",
        },
        seed=17,
    )


def _teacher_head(
    *,
    layer_index: int,
    expert_id: str,
    weight: tuple[float, float],
) -> AffineRouterHead:
    head = AffineRouterHead(
        2,
        layer_index=layer_index,
        expert_id=expert_id,
        threshold=0.5,
        trainable=False,
    )
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([weight]))
    return head


def _teacher() -> LogicalAppendOnlyRouter:
    heads = AppendOnlyRouterHeads(2)
    for layer_index, axis in ((2, 0), (4, 1)):
        positive = [0.0, 0.0]
        positive[axis] = 1.0
        negative = [0.0, 0.0]
        negative[axis] = -1.0
        heads.append(
            _teacher_head(
                layer_index=layer_index,
                expert_id="positive:0",
                weight=tuple(positive),
            )
        )
        heads.append(
            _teacher_head(
                layer_index=layer_index,
                expert_id="negative:0",
                weight=tuple(negative),
            )
        )
    return compile_logical_router(heads).eval()


def _anchored_routes(
    router: RouterScoreBank,
    features: Tensor,
    *,
    layer_index: int,
) -> Tensor:
    scores = router(features, layer_index=layer_index)
    return torch.argmax(
        torch.cat((torch.zeros(scores.shape[0], 1), scores), dim=-1),
        dim=-1,
    )


def test_distillation_learns_hard_routes_and_returns_frozen_logical_router() -> None:
    teacher = _teacher()
    feature_bank = _feature_bank()
    config = HardRouteDistillationConfig(
        learning_rate=0.1,
        optimization_steps=200,
        l2_weight=1e-4,
    )

    student, report = distill_frozen_router(
        teacher,
        feature_bank,
        config=config,
    )

    assert isinstance(student, LogicalAppendOnlyRouter)
    assert student.training is False
    assert tuple(student.parameters()) == ()
    assert student.head_keys == (
        (2, "positive:0"),
        (2, "negative:0"),
        (4, "positive:0"),
        (4, "negative:0"),
    )
    assert report.config == config
    assert report.feature_bank_metadata_sha256 == feature_bank.metadata_sha256
    assert report.teacher_architecture == teacher.architecture
    assert report.row_count == 5
    assert report.final_route_agreement == 1.0

    for layer_report in report.layer_reports:
        features = feature_bank.features_for_layer(layer_report.layer_index)
        assert torch.equal(
            _anchored_routes(
                student,
                features,
                layer_index=layer_report.layer_index,
            ),
            _anchored_routes(
                teacher,
                features,
                layer_index=layer_report.layer_index,
            ),
        )
        assert layer_report.target_counts == (1, 2, 2)
        assert layer_report.anchor_target_count == 1
        assert layer_report.final_agreement_count == 5
        assert layer_report.final_cross_entropy < (
            layer_report.initial_cross_entropy
        )
        assert layer_report.final_total_loss <= (
            layer_report.initial_cross_entropy
        )


def test_distillation_is_deterministic_and_does_not_mutate_or_alias_teacher() -> None:
    teacher = _teacher()
    feature_bank = _feature_bank()
    teacher_state = {
        name: value.clone() for name, value in teacher.state_dict().items()
    }
    teacher_pointers = {
        value.data_ptr() for value in teacher.state_dict().values()
    }
    config = HardRouteDistillationConfig(optimization_steps=64)

    first, first_report = distill_frozen_router(
        teacher,
        feature_bank,
        config=config,
    )
    second, second_report = distill_frozen_router(
        teacher,
        feature_bank,
        config=config,
    )

    assert first_report == second_report
    assert first_report.teacher_assignments_sha256 == (
        second_report.teacher_assignments_sha256
    )
    assert all(
        torch.equal(value, teacher_state[name])
        for name, value in teacher.state_dict().items()
    )
    assert {
        value.data_ptr() for value in first.state_dict().values()
    }.isdisjoint(teacher_pointers)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_anchor_first_ties_and_report_round_trip_are_explicit() -> None:
    teacher = _teacher()
    with torch.no_grad():
        for layer in teacher.layers.values():
            layer.weight.zero_()
            layer.bias.zero_()
            layer.threshold.zero_()
    feature_bank = _feature_bank()

    student, report = distill_frozen_router(
        teacher,
        feature_bank,
        config=HardRouteDistillationConfig(optimization_steps=8),
    )
    reconstructed = RouterDistillationReport.from_dict(report.to_dict())

    assert reconstructed == report
    assert report.final_route_agreement == 1.0
    for layer_report in report.layer_reports:
        assert layer_report.target_counts == (5, 0, 0)
        assert layer_report.initial_agreement_count == 5
        assert layer_report.final_agreement_count == 5
        features = feature_bank.features_for_layer(layer_report.layer_index)
        assert torch.equal(
            _anchored_routes(
                student,
                features,
                layer_index=layer_report.layer_index,
            ),
            torch.zeros(5, dtype=torch.long),
        )


class _NonFiniteTeacher(RouterScoreBank):
    """Minimal invalid teacher used to exercise output validation."""

    architecture = "nonfinite_test_teacher"

    def __init__(self) -> None:
        super().__init__()
        self.feature_size = 2

    def expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        return ("bad:0",)

    def forward(self, features: Tensor, *, layer_index: int) -> Tensor:
        return torch.full(
            (features.shape[0], 1),
            torch.nan,
            dtype=features.dtype,
        )


def test_distillation_rejects_mutable_and_incompatible_inputs() -> None:
    feature_bank = _feature_bank()
    training_teacher = _teacher().train()
    with pytest.raises(ValueError, match="evaluation mode"):
        distill_frozen_router(training_teacher, feature_bank)

    wrong_width_teacher = LogicalAppendOnlyRouter(3).eval()
    with pytest.raises(ValueError, match="feature size"):
        distill_frozen_router(wrong_width_teacher, feature_bank)

    with pytest.raises(ValueError, match="finite"):
        distill_frozen_router(_NonFiniteTeacher().eval(), feature_bank)

    with pytest.raises(TypeError, match="HardRouteDistillationConfig"):
        distill_frozen_router(
            _teacher(),
            feature_bank,
            config={"optimization_steps": 8},  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="positive integer"):
        HardRouteDistillationConfig(optimization_steps=0)
    with pytest.raises(ValueError, match="non-negative"):
        HardRouteDistillationConfig(l2_weight=-1.0)


def test_distillation_report_schema_rejects_unknown_fields() -> None:
    _, report = distill_frozen_router(
        _teacher(),
        _feature_bank(),
        config=HardRouteDistillationConfig(optimization_steps=8),
    )
    malformed = report.to_dict()
    malformed["unexpected"] = True

    with pytest.raises(ValueError, match="fields do not match"):
        RouterDistillationReport.from_dict(malformed)
