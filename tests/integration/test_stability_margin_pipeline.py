"""CAL-005 to CAL-006 sequential-installation refinement pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    refine_stability_margins,
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
            "split": "validation",
            "filters": [],
        },
        seed=83,
        row_ids=row_ids,
    )


def _labels(
    bank: RouterFeatureBank,
    utilities: torch.Tensor,
) -> UtilityLabelBank:
    return UtilityLabelBank(
        {2: utilities},
        expert_ids_by_layer={2: ("code:new",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="registry:old",
    )


def test_calibrated_candidate_refines_without_changing_incumbent(
    tmp_path: Path,
) -> None:
    calibration_features = torch.tensor(
        [[-2.0], [-1.0], [1.0], [2.0], [-2.0], [-1.0], [1.0], [2.0]]
    )
    calibration_utilities = torch.tensor(
        [[-1.0], [-1.0], [1.0], [1.0], [-1.0], [-1.0], [1.0], [1.0]]
    )
    calibration_bank = _bank(
        calibration_features,
        dataset_id="synthetic-stability-calibration",
        row_ids=torch.arange(100, 108),
    )
    fitted_candidate = AppendOnlyRouterHeads(1)
    candidate_head = AffineRouterHead(
        1,
        layer_index=2,
        expert_id="code:new",
        trainable=False,
    )
    with torch.no_grad():
        candidate_head.linear.weight.fill_(0.05)
    fitted_candidate.append(candidate_head)
    calibrated_candidate, calibration_report = (
        calibrate_temperature_and_thresholds(
            fitted_candidate,
            calibration_bank,
            _labels(calibration_bank, calibration_utilities),
            temperature_row_ids=[100, 101, 102, 103],
            threshold_row_ids=[104, 105, 106, 107],
            false_activation_rate=0.0,
        )
    )

    incumbent = AppendOnlyRouterHeads(1)
    incumbent_head = AffineRouterHead(
        1,
        layer_index=2,
        expert_id="code:old",
        trainable=False,
    )
    with torch.no_grad():
        incumbent_head.linear.bias.fill_(20.0)
    incumbent.append(incumbent_head)
    incumbent_state = {
        name: tensor.detach().clone()
        for name, tensor in incumbent.state_dict().items()
    }
    calibrated_state = {
        name: tensor.detach().clone()
        for name, tensor in calibrated_candidate.state_dict().items()
    }

    refinement_features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    refinement_bank = _bank(
        refinement_features,
        dataset_id="synthetic-stability-refinement",
        row_ids=torch.arange(200, 204),
    )
    refined, report = refine_stability_margins(
        calibrated_candidate,
        incumbent,
        refinement_bank,
        _labels(
            refinement_bank,
            torch.tensor([[-1.0], [-1.0], [1.0], [1.0]]),
        ),
        stability_row_ids=[200, 201],
        win_row_ids=[202, 203],
        stability_margin=0.2,
        win_margin=0.2,
        optimization_steps=192,
    )

    old_scores = incumbent(refinement_features, layer_index=2)
    before_scores = calibrated_candidate(refinement_features, layer_index=2)
    after_scores = refined(refinement_features, layer_index=2)
    selector = AnchoredTop1Selection()
    before = selector(
        torch.cat((old_scores, before_scores), dim=-1),
        expert_ids=("code:old", "code:new"),
    )
    after = selector(
        torch.cat((old_scores, after_scores), dim=-1),
        expert_ids=("code:old", "code:new"),
    )

    assert torch.equal(before.selected_indices, torch.ones(4, dtype=torch.long))
    assert torch.equal(
        after.selected_indices,
        torch.tensor([1, 1, 2, 2]),
    )
    assert report.head_reports[0].stability_violation_count_after == 0
    assert report.head_reports[0].win_violation_count_after == 0
    assert calibration_report.head_reports[0].temperature == (
        calibrated_candidate.heads[0].temperature
    )
    assert refined.heads[0].temperature == calibrated_candidate.heads[0].temperature
    assert refined.heads[0].threshold == calibrated_candidate.heads[0].threshold
    for name, tensor in incumbent_state.items():
        assert torch.equal(incumbent.state_dict()[name], tensor)
    for name, tensor in calibrated_state.items():
        assert torch.equal(calibrated_candidate.state_dict()[name], tensor)

    payload = tmp_path / "stability-refined-router.safetensors"
    refined.save_safetensors(payload)
    restored = AppendOnlyRouterHeads.from_safetensors(payload)
    assert torch.equal(
        restored(refinement_features, layer_index=2),
        after_scores,
    )
