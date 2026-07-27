"""CAL-004 to CAL-005 calibration pipeline with separate reference banks."""

from __future__ import annotations

from pathlib import Path

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    fit_independent_ridge,
)


def _bank(
    features: torch.Tensor,
    *,
    dataset_id: str,
    row_ids: torch.Tensor,
) -> RouterFeatureBank:
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
            "split": "train" if dataset_id.endswith("fit") else "validation",
            "filters": [],
        },
        seed=71,
        row_ids=row_ids,
    )


def _labels(
    bank: RouterFeatureBank,
    utilities: torch.Tensor,
) -> UtilityLabelBank:
    return UtilityLabelBank(
        {2: utilities},
        expert_ids_by_layer={2: ("code:0",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def test_ridge_heads_are_calibrated_on_separate_rows_and_reload_exactly(
    tmp_path: Path,
) -> None:
    fit_features = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    fit_bank = _bank(
        fit_features,
        dataset_id="synthetic-temperature-fit",
        row_ids=torch.arange(6),
    )
    fitted = fit_independent_ridge(
        fit_bank,
        _labels(fit_bank, fit_features.clone()),
        regularization=1e-6,
    )
    calibration_features = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0], [-0.5], [0.5], [1.0], [2.0], [3.0], [4.0]]
    )
    calibration_utilities = torch.tensor(
        [[-1.0], [-1.0], [1.0], [1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0], [1.0]]
    )
    calibration_bank = _bank(
        calibration_features,
        dataset_id="synthetic-temperature-validation",
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

    assert torch.equal(
        decision.selected_indices,
        torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    assert report.head_reports[0].false_activation_rate <= 1.0 / 3.0
    assert fitted.heads[0].temperature == 1.0
    assert fitted.heads[0].threshold == 0.0

    payload_path = tmp_path / "calibrated-router.safetensors"
    calibrated.save_safetensors(payload_path)
    restored = AppendOnlyRouterHeads.from_safetensors(payload_path)
    assert restored.head_keys == calibrated.head_keys
    assert restored.heads[0].temperature == calibrated.heads[0].temperature
    assert restored.heads[0].threshold == calibrated.heads[0].threshold
    assert torch.equal(restored(calibration_features[4:], layer_index=2), scores)
