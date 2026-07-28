"""Acceptance tests for RTR-004 global router-only recalibration."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    GlobalRouterRecalibrationReport,
    RouterFeatureBank,
    UtilityLabelBank,
    compile_logical_router,
    recalibrate_global_router,
)


def _feature_bank() -> RouterFeatureBank:
    return RouterFeatureBank(
        {
            2: torch.tensor(
                [
                    [-2.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ]
            )
        },
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 2,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "tests/router-replay",
            "revision": "v1",
            "split": "validation",
            "filters": ["synthetic"],
        },
        seed=7,
    )


def _utility_bank(
    features: RouterFeatureBank,
    *,
    expert_ids: tuple[str, ...] = ("code:0", "math:0"),
    incumbent_route: str = "anchor",
) -> UtilityLabelBank:
    return UtilityLabelBank(
        {
            2: torch.tensor(
                [
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [-1.0, 1.0],
                ]
            )
        },
        expert_ids_by_layer={2: expert_ids},
        row_ids=features.row_ids,
        feature_bank_metadata_sha256=features.metadata_sha256,
        loss_name="token_nll",
        incumbent_route=incumbent_route,
    )


def _source_router():
    heads = AppendOnlyRouterHeads(2)
    heads.append(AffineRouterHead(2, layer_index=2, expert_id="code:0"))
    heads.append(AffineRouterHead(2, layer_index=2, expert_id="math:0"))
    return compile_logical_router(heads)


def test_global_recalibration_improves_routes_without_mutating_source() -> None:
    features = _feature_bank()
    utilities = _utility_bank(features)
    source = _source_router()
    source_state = {
        name: tensor.clone() for name, tensor in source.state_dict().items()
    }

    recalibrated, report = recalibrate_global_router(
        source,
        features,
        utilities,
        registry_sha256="a" * 64,
        compatibility_sha256="b" * 64,
        learning_rate=0.1,
        optimization_steps=128,
        drift_weight=1e-4,
    )

    layer = report.layer_reports[0]
    assert report.strategy_name == "router_only_global"
    assert report.registry_sha256 == "a" * 64
    assert layer.expert_ids == ("code:0", "math:0")
    assert layer.anchor_target_count == 1
    assert layer.final_cross_entropy < layer.initial_cross_entropy
    assert layer.final_correct_count > layer.initial_correct_count
    assert layer.changed_route_count > 0
    assert tuple(recalibrated.parameters()) == ()
    assert recalibrated.head_keys == source.head_keys
    assert all(
        torch.equal(source.state_dict()[name], value)
        for name, value in source_state.items()
    )
    assert {
        value.data_ptr() for value in source.state_dict().values()
    }.isdisjoint(
        value.data_ptr() for value in recalibrated.state_dict().values()
    )


def test_global_recalibration_is_deterministic_and_report_round_trips() -> None:
    features = _feature_bank()
    utilities = _utility_bank(features)

    first_router, first_report = recalibrate_global_router(
        _source_router(),
        features,
        utilities,
        registry_sha256="c" * 64,
        compatibility_sha256="d" * 64,
        optimization_steps=48,
    )
    second_router, second_report = recalibrate_global_router(
        _source_router(),
        features,
        utilities,
        registry_sha256="c" * 64,
        compatibility_sha256="d" * 64,
        optimization_steps=48,
    )

    assert first_report == second_report
    assert GlobalRouterRecalibrationReport.from_dict(
        first_report.to_dict()
    ) == first_report
    assert all(
        torch.equal(first_router.state_dict()[name], tensor)
        for name, tensor in second_router.state_dict().items()
    )


def test_global_recalibration_rejects_semantically_incompatible_inputs() -> None:
    features = _feature_bank()
    source = _source_router()

    with pytest.raises(ValueError, match="incumbent_route"):
        recalibrate_global_router(
            source,
            features,
            _utility_bank(features, incumbent_route="registry:old"),
            registry_sha256="a" * 64,
            compatibility_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="expert order"):
        recalibrate_global_router(
            source,
            features,
            _utility_bank(features, expert_ids=("math:0", "code:0")),
            registry_sha256="a" * 64,
            compatibility_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        recalibrate_global_router(
            source,
            features,
            _utility_bank(features),
            registry_sha256="invalid",
            compatibility_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="positive"):
        recalibrate_global_router(
            source,
            features,
            _utility_bank(features),
            registry_sha256="a" * 64,
            compatibility_sha256="b" * 64,
            optimization_steps=0,
        )
