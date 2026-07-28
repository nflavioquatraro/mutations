"""Acceptance tests for CAL-006 stability-margin refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    RouterFeatureBank,
    RouterScoreBank,
    UtilityLabelBank,
    compile_logical_router,
    refine_stability_margins,
)


_FEATURES = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
_ROW_IDS = torch.tensor([10, 11, 12, 13])


def _feature_bank(
    features_by_layer: dict[int, torch.Tensor] | None = None,
) -> RouterFeatureBank:
    return RouterFeatureBank(
        features_by_layer or {2: _FEATURES},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 1,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "synthetic-stability-reference",
            "revision": "1",
            "split": "validation",
            "filters": [],
        },
        seed=79,
        row_ids=_ROW_IDS,
    )


def _labels(
    bank: RouterFeatureBank,
    utilities_by_layer: dict[int, torch.Tensor] | None = None,
    *,
    expert_ids_by_layer: dict[int, tuple[str, ...]] | None = None,
) -> UtilityLabelBank:
    utilities = utilities_by_layer or {
        layer: torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
        for layer in bank.layer_indices
    }
    expert_ids = expert_ids_by_layer or {
        layer: ("code:new",) for layer in bank.layer_indices
    }
    return UtilityLabelBank(
        utilities,
        expert_ids_by_layer=expert_ids,
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="registry:old",
    )


def _head(
    expert_id: str,
    *,
    layer_index: int = 2,
    weight: float = 0.0,
    bias: float = 0.0,
    trainable: bool = False,
) -> AffineRouterHead:
    head = AffineRouterHead(
        1,
        layer_index=layer_index,
        expert_id=expert_id,
        temperature=1.0,
        threshold=0.0,
        trainable=trainable,
    )
    with torch.no_grad():
        head.linear.weight.fill_(weight)
        head.linear.bias.fill_(bias)
    return head


def _routers() -> tuple[AppendOnlyRouterHeads, AppendOnlyRouterHeads]:
    incumbent = AppendOnlyRouterHeads(1)
    incumbent.append(_head("code:old", bias=0.2))
    candidate = AppendOnlyRouterHeads(1)
    candidate.append(_head("code:new"))
    return incumbent, candidate


def _refine(
    candidate: AppendOnlyRouterHeads | None = None,
    incumbent: RouterScoreBank | None = None,
    bank: RouterFeatureBank | None = None,
    labels: UtilityLabelBank | None = None,
) -> tuple[AppendOnlyRouterHeads, object]:
    source_bank = bank or _feature_bank()
    source_incumbent, source_candidate = _routers()
    return refine_stability_margins(
        candidate or source_candidate,
        incumbent if incumbent is not None else source_incumbent,
        source_bank,
        labels or _labels(source_bank),
        stability_row_ids=[11, 10],
        win_row_ids=[13, 12],
        stability_margin=0.3,
        win_margin=0.3,
        drift_weight=1e-3,
        learning_rate=0.05,
        optimization_steps=128,
    )


def test_refinement_reduces_margin_violations_and_reports_routes() -> None:
    refined, report = _refine()
    head_report = report.head_reports[0]  # type: ignore[union-attr]

    assert report.stability_row_ids == (10, 11)  # type: ignore[union-attr]
    assert report.win_row_ids == (12, 13)  # type: ignore[union-attr]
    assert head_report.status == "optimized"
    assert head_report.final_total_loss < head_report.initial_total_loss
    assert head_report.stability_violation_count_before == 2
    assert head_report.stability_violation_count_after == 0
    assert head_report.win_violation_count_before == 2
    assert head_report.win_violation_count_after == 0
    assert head_report.stability_activation_count_after == 0
    assert head_report.win_activation_count_after == 2
    assert head_report.useful_win_row_count == 2
    assert head_report.ignored_win_row_count == 0
    scores = refined(_FEATURES, layer_index=2).squeeze(-1)
    assert torch.all(scores[:2] < 0.2)
    assert torch.all(scores[2:] > 0.2)
    json.dumps(report.to_dict(), allow_nan=False)  # type: ignore[union-attr]


def test_refinement_is_deterministic_frozen_and_source_immutable(
    tmp_path: Path,
) -> None:
    incumbent, candidate = _routers()
    incumbent_state = {
        name: tensor.detach().clone()
        for name, tensor in incumbent.state_dict().items()
    }
    candidate_state = {
        name: tensor.detach().clone()
        for name, tensor in candidate.state_dict().items()
    }
    candidate_pointers = tuple(
        parameter.data_ptr() for parameter in candidate.parameters()
    )

    first, first_report = _refine(candidate=candidate, incumbent=incumbent)
    second, second_report = _refine(candidate=candidate, incumbent=incumbent)

    assert first_report == second_report
    assert first.head_keys == candidate.head_keys
    assert all(not parameter.requires_grad for parameter in first.parameters())
    assert tuple(parameter.data_ptr() for parameter in first.parameters()) != (
        candidate_pointers
    )
    for name, tensor in incumbent_state.items():
        assert torch.equal(incumbent.state_dict()[name], tensor)
    for name, tensor in candidate_state.items():
        assert torch.equal(candidate.state_dict()[name], tensor)
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name])

    payload = tmp_path / "refined-router.safetensors"
    first.save_safetensors(payload)
    restored = AppendOnlyRouterHeads.from_safetensors(payload)
    assert restored.head_keys == first.head_keys
    assert torch.equal(
        restored(_FEATURES, layer_index=2),
        first(_FEATURES, layer_index=2),
    )


def test_anchor_only_refinement_ignores_non_positive_win_rows() -> None:
    bank = _feature_bank()
    candidate = AppendOnlyRouterHeads(1)
    candidate.append(_head("code:new", bias=0.5))
    labels = _labels(
        bank,
        {2: torch.tensor([[-1.0], [-1.0], [1.0], [0.0]])},
    )

    refined, report = refine_stability_margins(
        candidate,
        None,
        bank,
        labels,
        stability_row_ids=[10, 11],
        win_row_ids=[12, 13],
        stability_margin=0.2,
        win_margin=0.2,
        optimization_steps=128,
    )
    head_report = report.head_reports[0]

    assert head_report.useful_win_row_count == 1
    assert head_report.ignored_win_row_count == 1
    assert head_report.stability_activation_count_after == 0
    assert head_report.win_activation_count_after == 1
    assert refined.heads[0].temperature == candidate.heads[0].temperature
    assert refined.heads[0].threshold == candidate.heads[0].threshold


def test_refinement_accepts_compiled_logical_incumbent() -> None:
    incumbent, candidate = _routers()
    logical_incumbent = compile_logical_router(incumbent)

    independent_refined, independent_report = _refine(
        candidate=candidate,
        incumbent=incumbent,
    )
    logical_refined, logical_report = _refine(
        candidate=candidate,
        incumbent=logical_incumbent,
    )

    assert independent_report == logical_report
    for name, tensor in independent_refined.state_dict().items():
        assert torch.equal(tensor, logical_refined.state_dict()[name])


def test_refinement_preserves_interleaved_global_head_order() -> None:
    bank = _feature_bank({1: _FEATURES, 3: _FEATURES * 2.0})
    candidate = AppendOnlyRouterHeads(1)
    for layer_index, expert_id in (
        (1, "code:0"),
        (3, "code:0"),
        (1, "code:1"),
        (3, "code:1"),
    ):
        candidate.append(_head(expert_id, layer_index=layer_index))
    utilities = torch.tensor(
        [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    labels = _labels(
        bank,
        {1: utilities, 3: utilities},
        expert_ids_by_layer={
            1: ("code:0", "code:1"),
            3: ("code:0", "code:1"),
        },
    )

    refined, report = refine_stability_margins(
        candidate,
        None,
        bank,
        labels,
        stability_row_ids=[10, 11],
        win_row_ids=[12, 13],
        optimization_steps=64,
    )

    assert refined.head_keys == candidate.head_keys
    assert tuple(
        (item.layer_index, item.expert_id) for item in report.head_reports
    ) == candidate.head_keys


@pytest.mark.parametrize(
    ("stability_ids", "win_ids", "message"),
    [
        ([], [12, 13], "empty"),
        ([10, 10], [12, 13], "duplicate"),
        ([10, 99], [12, 13], "unknown"),
        ([10, 11], [11, 12], "disjoint"),
    ],
)
def test_refinement_rejects_invalid_row_partitions(
    stability_ids: list[int],
    win_ids: list[int],
    message: str,
) -> None:
    incumbent, candidate = _routers()
    bank = _feature_bank()

    with pytest.raises(ValueError, match=message):
        refine_stability_margins(
            candidate,
            incumbent,
            bank,
            _labels(bank),
            stability_row_ids=stability_ids,
            win_row_ids=win_ids,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("stability_margin", -0.1, ValueError),
        ("win_margin", float("nan"), ValueError),
        ("stability_weight", 0.0, ValueError),
        ("win_weight", "1", TypeError),
        ("drift_weight", -1.0, ValueError),
        ("learning_rate", float("inf"), ValueError),
        ("optimization_steps", 0, ValueError),
        ("optimization_steps", True, TypeError),
    ],
)
def test_refinement_rejects_invalid_hyperparameters(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    incumbent, candidate = _routers()
    bank = _feature_bank()
    kwargs = {field_name: value}

    with pytest.raises(error_type):
        refine_stability_margins(
            candidate,
            incumbent,
            bank,
            _labels(bank),
            stability_row_ids=[10, 11],
            win_row_ids=[12, 13],
            **kwargs,  # type: ignore[arg-type]
        )


def test_refinement_rejects_trainable_misaligned_and_duplicate_heads() -> None:
    incumbent, candidate = _routers()
    bank = _feature_bank()
    trainable = AppendOnlyRouterHeads(1)
    trainable.append(_head("code:new", trainable=True))
    duplicate_incumbent = AppendOnlyRouterHeads(1)
    duplicate_incumbent.append(_head("code:new"))

    with pytest.raises(ValueError, match="frozen"):
        _refine(candidate=trainable, incumbent=incumbent)
    with pytest.raises(ValueError, match="disjoint"):
        _refine(candidate=candidate, incumbent=duplicate_incumbent)
    with pytest.raises(ValueError, match="expert order"):
        _refine(
            candidate=candidate,
            incumbent=incumbent,
            labels=_labels(
                bank,
                expert_ids_by_layer={2: ("math:new",)},
            ),
        )
    with pytest.raises(TypeError, match="candidate_router"):
        refine_stability_margins(
            object(),  # type: ignore[arg-type]
            incumbent,
            bank,
            _labels(bank),
            stability_row_ids=[10, 11],
            win_row_ids=[12, 13],
        )
