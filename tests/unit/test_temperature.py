"""Acceptance tests for CAL-005 temperature and threshold fitting."""

from __future__ import annotations

import json

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
)


_ROW_IDS = torch.arange(10, 20)
_FEATURES = torch.tensor(
    [[-2.0], [-1.0], [1.0], [2.0], [-0.5], [0.5], [1.0], [2.0], [3.0], [4.0]]
)
_UTILITIES = torch.tensor(
    [[-1.0], [-1.0], [1.0], [1.0], [-1.0], [1.0], [0.0], [1.0], [-1.0], [1.0]]
)


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
            "dataset_id": "synthetic-temperature-reference",
            "revision": "1",
            "split": "validation",
            "filters": [],
        },
        seed=67,
        row_ids=_ROW_IDS,
    )


def _utility_bank(
    feature_bank: RouterFeatureBank,
    utilities_by_layer: dict[int, torch.Tensor] | None = None,
    *,
    expert_ids_by_layer: dict[int, tuple[str, ...]] | None = None,
) -> UtilityLabelBank:
    utilities = utilities_by_layer or {2: _UTILITIES}
    expert_ids = expert_ids_by_layer or {
        layer: ("code:0",) for layer in feature_bank.layer_indices
    }
    return UtilityLabelBank(
        utilities,
        expert_ids_by_layer=expert_ids,
        row_ids=feature_bank.row_ids,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def _router(
    *,
    temperature: float = 1.0,
    threshold: float = 0.0,
    expert_id: str = "code:0",
) -> AppendOnlyRouterHeads:
    router = AppendOnlyRouterHeads(feature_size=1)
    head = AffineRouterHead(
        1,
        layer_index=2,
        expert_id=expert_id,
        temperature=temperature,
        threshold=threshold,
    )
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
    router.append(head)
    return router


def _calibrate(
    router: AppendOnlyRouterHeads | None = None,
    feature_bank: RouterFeatureBank | None = None,
    labels: UtilityLabelBank | None = None,
    *,
    budget: float = 1.0 / 3.0,
) -> tuple[AppendOnlyRouterHeads, object]:
    bank = feature_bank or _feature_bank()
    utility_labels = labels or _utility_bank(bank)
    return calibrate_temperature_and_thresholds(
        router or _router(),
        bank,
        utility_labels,
        temperature_row_ids=[13, 10, 12, 11],
        threshold_row_ids=[19, 14, 18, 15, 17, 16],
        false_activation_rate=budget,
    )


def test_temperature_improves_binary_log_loss_and_threshold_respects_budget() -> None:
    calibrated, report = _calibrate()
    head_report = report.head_reports[0]  # type: ignore[union-attr]

    assert report.temperature_row_ids == (10, 11, 12, 13)  # type: ignore[union-attr]
    assert report.threshold_row_ids == (14, 15, 16, 17, 18, 19)  # type: ignore[union-attr]
    assert 0.01 <= head_report.temperature < 1.0
    assert head_report.binary_log_loss_after < head_report.binary_log_loss_before
    assert head_report.temperature_status == "optimized"
    assert head_report.threshold_status == "optimized"
    assert head_report.positive_count == 3
    assert head_report.negative_count == 3
    assert head_report.true_positive_count == 2
    assert head_report.false_positive_count == 1
    assert head_report.true_positive_rate == pytest.approx(2.0 / 3.0)
    assert head_report.false_activation_rate == pytest.approx(1.0 / 3.0)
    assert head_report.threshold * head_report.temperature == pytest.approx(1.0)

    threshold_features = _FEATURES[4:]
    scores = calibrated(threshold_features, layer_index=2)
    decision = AnchoredTop1Selection()(
        scores,
        expert_ids=calibrated.expert_ids_for_layer(2),
    )
    assert torch.equal(
        decision.selected_indices,
        torch.tensor([0, 0, 0, 1, 1, 1]),
    )


def test_threshold_groups_equal_scores_before_budget_selection() -> None:
    features = _FEATURES.clone()
    features[4:, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    feature_bank = _feature_bank({2: features})
    utilities = _UTILITIES.clone()
    utilities[4:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    labels = _utility_bank(feature_bank, {2: utilities})

    calibrated, report = _calibrate(
        feature_bank=feature_bank,
        labels=labels,
        budget=1.0 / 3.0,
    )
    head_report = report.head_reports[0]  # type: ignore[union-attr]
    decisions = calibrated(features[4:], layer_index=2) > 0

    assert head_report.false_positive_count == 1
    assert head_report.true_positive_count == 1
    assert head_report.threshold * head_report.temperature == pytest.approx(1.0)
    assert torch.equal(
        decisions.squeeze(-1),
        torch.tensor([False, False, False, False, True, True]),
    )


def test_calibration_returns_frozen_copy_without_mutating_source() -> None:
    router = _router()
    source_head = router.heads[0]
    original_weight = source_head.linear.weight.detach().clone()
    original_bias = source_head.linear.bias.detach().clone()
    original_pointers = tuple(
        parameter.data_ptr() for parameter in source_head.parameters()
    )

    calibrated, report = _calibrate(router=router)

    assert calibrated is not router
    assert calibrated.heads[0] is not source_head
    assert torch.equal(source_head.linear.weight, original_weight)
    assert torch.equal(source_head.linear.bias, original_bias)
    assert source_head.temperature == 1.0
    assert source_head.threshold == 0.0
    assert original_pointers == tuple(
        parameter.data_ptr() for parameter in source_head.parameters()
    )
    assert torch.equal(calibrated.heads[0].linear.weight, original_weight)
    assert torch.equal(calibrated.heads[0].linear.bias, original_bias)
    assert all(not parameter.requires_grad for parameter in calibrated.parameters())
    json.dumps(report.to_dict(), allow_nan=False)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("threshold_utilities", "expected_status", "expected_active", "expected_rate"),
    [
        (
            torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            "no_positive_fallback",
            0,
            0.0,
        ),
        (
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "no_negative_fallback",
            6,
            0.0,
        ),
    ],
)
def test_threshold_handles_single_class_partitions_conservatively(
    threshold_utilities: torch.Tensor,
    expected_status: str,
    expected_active: int,
    expected_rate: float,
) -> None:
    feature_bank = _feature_bank()
    utilities = _UTILITIES.clone()
    utilities[4:, 0] = threshold_utilities
    labels = _utility_bank(feature_bank, {2: utilities})

    calibrated, report = _calibrate(
        feature_bank=feature_bank,
        labels=labels,
        budget=0.0,
    )
    head_report = report.head_reports[0]  # type: ignore[union-attr]
    active = calibrated(_FEATURES[4:], layer_index=2) > 0

    assert int(active.sum().item()) == expected_active
    assert head_report.false_activation_rate == expected_rate
    assert head_report.threshold_status == expected_status
    if expected_status == "no_positive_fallback":
        assert head_report.positive_count == 0
        assert head_report.true_positive_rate == 0.0
    else:
        assert head_report.negative_count == 0
        assert head_report.true_positive_rate == 1.0


def test_temperature_uses_neutral_fallback_for_single_class_targets() -> None:
    feature_bank = _feature_bank()
    utilities = _UTILITIES.clone()
    utilities[:4, 0] = 1.0
    labels = _utility_bank(feature_bank, {2: utilities})

    calibrated, report = _calibrate(feature_bank=feature_bank, labels=labels)
    head_report = report.head_reports[0]  # type: ignore[union-attr]

    assert calibrated.heads[0].temperature == 1.0
    assert head_report.temperature_status == "single_class_fallback"
    assert (
        head_report.binary_log_loss_after
        == head_report.binary_log_loss_before
    )


def test_calibration_preserves_global_append_order_across_layers() -> None:
    feature_bank = _feature_bank({1: _FEATURES, 3: _FEATURES * 2.0})
    router = AppendOnlyRouterHeads(feature_size=1)
    for layer_index, expert_id in (
        (1, "code:0"),
        (3, "code:0"),
        (1, "math:0"),
        (3, "math:0"),
    ):
        head = AffineRouterHead(1, layer_index=layer_index, expert_id=expert_id)
        with torch.no_grad():
            head.linear.weight.fill_(1.0 if expert_id == "code:0" else -1.0)
        router.append(head)
    utilities = torch.column_stack((_UTILITIES[:, 0], -_UTILITIES[:, 0]))
    labels = _utility_bank(
        feature_bank,
        {1: utilities, 3: utilities},
        expert_ids_by_layer={
            1: ("code:0", "math:0"),
            3: ("code:0", "math:0"),
        },
    )

    calibrated, report = calibrate_temperature_and_thresholds(
        router,
        feature_bank,
        labels,
        temperature_row_ids=[10, 11, 12, 13],
        threshold_row_ids=[14, 15, 16, 17, 18, 19],
        false_activation_rate=1.0,
    )

    assert calibrated.head_keys == router.head_keys
    assert tuple(
        (item.layer_index, item.expert_id) for item in report.head_reports
    ) == router.head_keys


@pytest.mark.parametrize(
    ("temperature_ids", "threshold_ids", "message"),
    [
        ([10, 11], [11, 12, 13, 14, 15, 16, 17, 18, 19], "disjoint"),
        ([10, 11], [12, 13], "cover"),
        ([10, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19], "duplicate"),
        ([10, 99], [11, 12, 13, 14, 15, 16, 17, 18, 19], "unknown"),
        ([], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "empty"),
    ],
)
def test_calibration_rejects_invalid_row_partitions(
    temperature_ids: list[int],
    threshold_ids: list[int],
    message: str,
) -> None:
    feature_bank = _feature_bank()
    labels = _utility_bank(feature_bank)

    with pytest.raises(ValueError, match=message):
        calibrate_temperature_and_thresholds(
            _router(),
            feature_bank,
            labels,
            temperature_row_ids=temperature_ids,
            threshold_row_ids=threshold_ids,
        )


@pytest.mark.parametrize(
    ("budget", "exception"),
    [
        (True, TypeError),
        ("0.1", TypeError),
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_calibration_rejects_invalid_false_activation_rate(
    budget: object,
    exception: type[Exception],
) -> None:
    feature_bank = _feature_bank()
    labels = _utility_bank(feature_bank)

    with pytest.raises(exception):
        calibrate_temperature_and_thresholds(
            _router(),
            feature_bank,
            labels,
            temperature_row_ids=[10, 11, 12, 13],
            threshold_row_ids=[14, 15, 16, 17, 18, 19],
            false_activation_rate=budget,  # type: ignore[arg-type]
        )


def test_calibration_rejects_incompatible_or_precalibrated_heads() -> None:
    feature_bank = _feature_bank()
    labels = _utility_bank(feature_bank)

    with pytest.raises(ValueError, match="temperature 1 and threshold 0"):
        _calibrate(router=_router(threshold=0.5))
    with pytest.raises(ValueError, match="expert order"):
        calibrate_temperature_and_thresholds(
            _router(expert_id="math:0"),
            feature_bank,
            labels,
            temperature_row_ids=[10, 11, 12, 13],
            threshold_row_ids=[14, 15, 16, 17, 18, 19],
        )
    with pytest.raises(TypeError, match="AppendOnlyRouterHeads"):
        calibrate_temperature_and_thresholds(
            object(),  # type: ignore[arg-type]
            feature_bank,
            labels,
            temperature_row_ids=[10, 11, 12, 13],
            threshold_row_ids=[14, 15, 16, 17, 18, 19],
        )
