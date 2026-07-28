"""Verify deterministic RTR-007 anchored hard-route distillation."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    HardRouteDistillationConfig,
    RouterFeatureBank,
    compile_logical_router,
    distill_frozen_router,
)


def _anchored_routes(scores: torch.Tensor) -> torch.Tensor:
    """Return anchor-first routes for specialist score columns."""

    return torch.argmax(
        torch.cat((torch.zeros(scores.shape[0], 1), scores), dim=-1),
        dim=-1,
    )


def main() -> None:
    """Distill a frozen teacher and verify exact replay-route agreement."""

    features = RouterFeatureBank(
        {
            2: torch.tensor(
                [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
            )
        },
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 1,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "examples/offline-router-distillation",
            "filters": {},
            "revision": "v1",
            "split": "calibration",
        },
        seed=23,
    )
    heads = AppendOnlyRouterHeads(feature_size=1)
    for expert_id, weight in (("positive:0", 1.0), ("negative:0", -1.0)):
        head = AffineRouterHead(
            1,
            layer_index=2,
            expert_id=expert_id,
            threshold=0.5,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.fill_(weight)
        heads.append(head)
    teacher = compile_logical_router(heads).eval()

    student, report = distill_frozen_router(
        teacher,
        features,
        config=HardRouteDistillationConfig(
            learning_rate=0.1,
            optimization_steps=128,
        ),
    )
    replay = features.features_for_layer(2)
    teacher_routes = _anchored_routes(teacher(replay, layer_index=2))
    student_routes = _anchored_routes(student(replay, layer_index=2))

    assert torch.equal(student_routes, teacher_routes)
    assert report.final_route_agreement == 1.0
    assert tuple(student.parameters()) == ()
    assert student.training is False
    print(
        "RTR-007 validation passed: "
        f"strategy={report.strategy_name}, "
        f"teacher_policy={report.teacher_policy}, "
        f"agreement={report.final_route_agreement:.3f}, "
        f"assignments_sha256={report.teacher_assignments_sha256}, "
        "trainable_parameters=0"
    )


if __name__ == "__main__":
    main()
