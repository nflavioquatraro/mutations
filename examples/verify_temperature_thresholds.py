"""Verify CAL-004 to CAL-005 router calibration without model downloads."""

from __future__ import annotations

import json

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    fit_independent_ridge,
)


def _bank(
    features: torch.Tensor,
    *,
    dataset_id: str,
    split: str,
    row_ids: torch.Tensor,
) -> RouterFeatureBank:
    """Construct one deterministic synthetic feature bank."""

    return RouterFeatureBank(
        {2: features},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 1,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": dataset_id,
            "revision": "1",
            "split": split,
            "filters": [],
        },
        seed=73,
        row_ids=row_ids,
    )


def _labels(
    bank: RouterFeatureBank,
    utilities: torch.Tensor,
) -> UtilityLabelBank:
    """Bind one expert's utility targets to a feature bank."""

    return UtilityLabelBank(
        {2: utilities},
        expert_ids_by_layer={2: ("code:0",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def main() -> None:
    """Fit, calibrate, route, and validate immutable router metadata."""

    fit_features = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
    )
    fit_bank = _bank(
        fit_features,
        dataset_id="synthetic-local-temperature-fit",
        split="train",
        row_ids=torch.arange(6),
    )
    fitted = fit_independent_ridge(
        fit_bank,
        _labels(fit_bank, fit_features.clone()),
        regularization=1e-6,
    )

    calibration_features = torch.tensor(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
            [-0.5],
            [0.5],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]
    )
    calibration_utilities = torch.tensor(
        [
            [-1.0],
            [-1.0],
            [1.0],
            [1.0],
            [-1.0],
            [1.0],
            [-1.0],
            [1.0],
            [-1.0],
            [1.0],
        ]
    )
    calibration_bank = _bank(
        calibration_features,
        dataset_id="synthetic-local-temperature-validation",
        split="validation",
        row_ids=torch.arange(10, 20),
    )
    calibrated, report = calibrate_temperature_and_thresholds(
        fitted,
        calibration_bank,
        _labels(calibration_bank, calibration_utilities),
        temperature_row_ids=[10, 11, 12, 13],
        threshold_row_ids=[14, 15, 16, 17, 18, 19],
        false_activation_rate=1.0 / 3.0,
    )

    scores = calibrated(calibration_features[4:], layer_index=2)
    decision = AnchoredTop1Selection()(
        scores,
        expert_ids=calibrated.expert_ids_for_layer(2),
    )
    expected_routes = torch.tensor([0, 0, 0, 1, 1, 1])
    assert torch.equal(decision.selected_indices, expected_routes)
    assert fitted.heads[0].temperature == 1.0
    assert fitted.heads[0].threshold == 0.0
    json.dumps(report.to_dict(), allow_nan=False)

    head_report = report.head_reports[0]
    print(
        "Temperature/threshold validation passed: "
        f"temperature={head_report.temperature:.6g}, "
        f"threshold={head_report.threshold:.6g}, "
        f"false_activation_rate={head_report.false_activation_rate:.6g}, "
        f"routes={decision.selected_indices.tolist()}, "
        "immutable_input=True"
    )


if __name__ == "__main__":
    main()
