"""Acceptance tests for CAL-002 domain and utility label banks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from anchored_moe import (
    DomainLabelBank,
    LabelBankSerializationError,
    RouterFeatureBank,
    UtilityLabelBank,
)


def _feature_bank(*, offset: float = 0.0) -> RouterFeatureBank:
    return RouterFeatureBank(
        {
            1: torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
            )
            + offset,
            2: torch.tensor(
                [[1.5, 0.0], [0.0, 1.5], [0.75, 0.75]]
            )
            + offset,
        },
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 2,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "synthetic-label-reference",
            "revision": "1",
            "split": "train",
            "filters": [],
        },
        seed=23,
        row_ids=torch.tensor([10, 20, 30]),
    )


def _domain_bank(feature_bank: RouterFeatureBank | None = None) -> DomainLabelBank:
    bank = feature_bank or _feature_bank()
    return DomainLabelBank.from_domain_names(
        bank,
        ["math", "code", "math"],
        label_source="dataset:domain",
    )


def _utility_bank(
    feature_bank: RouterFeatureBank | None = None,
) -> UtilityLabelBank:
    bank = feature_bank or _feature_bank()
    incumbent_losses = {
        1: torch.tensor([1.0, 2.0, 3.0]),
        2: torch.tensor([2.0, 3.0, 4.0]),
    }
    expert_losses = {
        1: torch.tensor(
            [[0.5, 1.5], [2.5, 1.0], [2.0, 3.5]]
        ),
        2: torch.tensor(
            [[1.5, 2.5], [2.0, 3.5], [4.5, 3.0]]
        ),
    }
    expert_ids = {1: ["code:0", "math:0"], 2: ["code:0", "math:0"]}
    return UtilityLabelBank.from_losses(
        bank,
        incumbent_losses_by_layer=incumbent_losses,
        expert_losses_by_layer=expert_losses,
        expert_ids_by_layer=expert_ids,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def test_domain_names_use_sorted_class_order_and_exact_row_alignment() -> None:
    feature_bank = _feature_bank()
    bank = _domain_bank(feature_bank)

    assert bank.class_names == ("code", "math")
    assert bank.class_count == 2
    assert torch.equal(bank.labels, torch.tensor([1, 0, 1]))
    assert bank.domain_names == ("math", "code", "math")
    assert torch.equal(bank.row_ids, feature_bank.row_ids)
    assert bank.label_source == "dataset:domain"
    assert bank.feature_bank_metadata_sha256 == feature_bank.metadata_sha256
    bank.assert_aligned(feature_bank)

    with pytest.raises(ValueError, match="different"):
        bank.assert_aligned(_feature_bank(offset=1.0))


def test_domain_accessors_are_defensive_and_sources_are_not_mutated() -> None:
    feature_bank = _feature_bank()
    labels = torch.tensor([0, 1, 0])
    row_ids = feature_bank.row_ids
    bank = DomainLabelBank(
        labels,
        class_names=["code", "math"],
        row_ids=row_ids,
        feature_bank_metadata_sha256=feature_bank.metadata_sha256,
        label_source="manual:test",
    )

    returned_labels = bank.labels
    returned_row_ids = bank.row_ids
    returned_labels.zero_()
    returned_row_ids.zero_()

    assert torch.equal(bank.labels, torch.tensor([0, 1, 0]))
    assert torch.equal(bank.row_ids, torch.tensor([10, 20, 30]))
    assert torch.equal(labels, torch.tensor([0, 1, 0]))
    assert torch.equal(row_ids, torch.tensor([10, 20, 30]))


def test_utility_from_losses_matches_advantage_equation_and_order() -> None:
    feature_bank = _feature_bank()
    incumbent = {
        1: torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
        2: torch.tensor([2.0, 3.0, 4.0], requires_grad=True),
    }
    experts = {
        1: torch.tensor(
            [[0.5, 1.5], [2.5, 1.0], [2.0, 3.5]],
            requires_grad=True,
        ),
        2: torch.tensor(
            [[1.5, 2.5], [2.0, 3.5], [4.5, 3.0]],
            requires_grad=True,
        ),
    }
    expert_ids = {1: ["code:0", "math:0"], 2: ["code:0", "math:0"]}
    bank = UtilityLabelBank.from_losses(
        feature_bank,
        incumbent_losses_by_layer=incumbent,
        expert_losses_by_layer=experts,
        expert_ids_by_layer=expert_ids,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )

    expected_layer_one = incumbent[1].detach().unsqueeze(-1) - experts[1].detach()
    assert bank.layer_indices == (1, 2)
    assert bank.expert_ids_for_layer(1) == ("code:0", "math:0")
    assert torch.equal(bank.utilities_for_layer(1), expected_layer_one)
    assert not bank.utilities_for_layer(1).requires_grad
    assert incumbent[1].requires_grad
    assert experts[1].requires_grad
    assert bank.loss_name == "cross_entropy"
    assert bank.incumbent_route == "anchor"
    assert bank.dtype == torch.float32
    bank.assert_aligned(feature_bank)


def test_utility_accessors_are_defensive() -> None:
    bank = _utility_bank()
    utilities = bank.utilities_for_layer(1)
    row_ids = bank.row_ids
    utilities.zero_()
    row_ids.zero_()

    assert torch.count_nonzero(bank.utilities_for_layer(1)).item() > 0
    assert torch.equal(bank.row_ids, torch.tensor([10, 20, 30]))
    with pytest.raises(KeyError, match="no layer 9"):
        bank.utilities_for_layer(9)
    with pytest.raises(KeyError, match="no layer 9"):
        bank.expert_ids_for_layer(9)


@pytest.mark.parametrize("bank_factory", [_domain_bank, _utility_bank])
def test_label_bank_safetensors_round_trip_is_exact_and_deterministic(
    bank_factory: object,
    tmp_path: Path,
) -> None:
    bank = bank_factory()  # type: ignore[operator]
    first_path = tmp_path / "labels-a.safetensors"
    second_path = tmp_path / "labels-b.safetensors"
    bank.save_safetensors(first_path)
    bank.save_safetensors(second_path)

    if isinstance(bank, DomainLabelBank):
        restored = DomainLabelBank.from_safetensors(first_path)
        assert torch.equal(restored.labels, bank.labels)
        assert restored.class_names == bank.class_names
    else:
        restored = UtilityLabelBank.from_safetensors(first_path)
        assert restored.layer_indices == bank.layer_indices
        for layer_index in bank.layer_indices:
            assert torch.equal(
                restored.utilities_for_layer(layer_index),
                bank.utilities_for_layer(layer_index),
            )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert restored.metadata == bank.metadata
    assert restored.metadata_sha256 == bank.metadata_sha256
    assert torch.equal(restored.row_ids, bank.row_ids)


def test_label_serialization_protects_destinations(tmp_path: Path) -> None:
    bank = _domain_bank()
    payload_path = tmp_path / "labels.safetensors"
    bank.save_safetensors(payload_path)
    original_bytes = payload_path.read_bytes()

    with pytest.raises(FileExistsError):
        bank.save_safetensors(payload_path)
    with pytest.raises(LabelBankSerializationError, match="end in"):
        bank.save_safetensors(tmp_path / "labels.bin")
    with pytest.raises(LabelBankSerializationError, match="parent"):
        bank.save_safetensors(tmp_path / "missing" / "labels.safetensors")

    bank.save_safetensors(payload_path, overwrite=True)
    assert payload_path.read_bytes() == original_bytes


@pytest.mark.parametrize("bank_factory", [_domain_bank, _utility_bank])
def test_label_banks_reject_missing_metadata_and_finite_tensor_changes(
    bank_factory: object,
    tmp_path: Path,
) -> None:
    missing_metadata = tmp_path / "missing-metadata.safetensors"
    save_file({"row_ids": torch.arange(3)}, missing_metadata)
    loader = (
        DomainLabelBank.from_safetensors
        if bank_factory is _domain_bank
        else UtilityLabelBank.from_safetensors
    )
    with pytest.raises(LabelBankSerializationError, match="metadata"):
        loader(missing_metadata)

    bank = bank_factory()  # type: ignore[operator]
    valid_path = tmp_path / "valid.safetensors"
    changed_path = tmp_path / "changed.safetensors"
    bank.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()

    changed_tensors = dict(tensors)
    changed_name = "labels" if isinstance(bank, DomainLabelBank) else (
        "layers.1.utilities"
    )
    if isinstance(bank, DomainLabelBank):
        changed_tensors[changed_name] = (
            changed_tensors[changed_name] + 1
        ) % bank.class_count
    else:
        changed_tensors[changed_name] = changed_tensors[changed_name] + 1
    save_file(changed_tensors, changed_path, metadata=metadata)
    with pytest.raises(LabelBankSerializationError, match="does not match"):
        loader(changed_path)


def test_domain_loader_rejects_unknown_metadata_fields(tmp_path: Path) -> None:
    bank = _domain_bank()
    valid_path = tmp_path / "valid.safetensors"
    invalid_path = tmp_path / "invalid.safetensors"
    bank.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["anchored_moe"])
    metadata["unknown"] = True
    save_file(
        tensors,
        invalid_path,
        metadata={"anchored_moe": json.dumps(metadata)},
    )

    with pytest.raises(LabelBankSerializationError, match="fields"):
        DomainLabelBank.from_safetensors(invalid_path)


def test_domain_and_utility_payload_formats_are_not_interchangeable(
    tmp_path: Path,
) -> None:
    domain_path = tmp_path / "domains.safetensors"
    utility_path = tmp_path / "utilities.safetensors"
    _domain_bank().save_safetensors(domain_path)
    _utility_bank().save_safetensors(utility_path)

    with pytest.raises(LabelBankSerializationError, match="format"):
        DomainLabelBank.from_safetensors(utility_path)
    with pytest.raises(LabelBankSerializationError, match="format"):
        UtilityLabelBank.from_safetensors(domain_path)


@pytest.mark.parametrize(
    ("labels", "class_names", "match"),
    [
        (torch.tensor([0.0, 1.0]), ["code", "math"], "torch.long"),
        (torch.tensor([0, 2]), ["code", "math"], "outside"),
        (torch.tensor([0, 1]), ["code", "code"], "duplicates"),
    ],
)
def test_domain_bank_rejects_invalid_targets(
    labels: torch.Tensor,
    class_names: list[str],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DomainLabelBank(
            labels,
            class_names=class_names,
            row_ids=torch.arange(labels.shape[0]),
            feature_bank_metadata_sha256="0" * 64,
            label_source="test",
        )


def test_utility_bank_rejects_invalid_losses_and_expert_columns() -> None:
    feature_bank = _feature_bank()
    incumbents = {
        1: torch.ones(3),
        2: torch.ones(3),
    }
    experts = {
        1: torch.ones(3, 2),
        2: torch.ones(3, 2),
    }
    expert_ids = {1: ["code:0", "math:0"], 2: ["code:0", "math:0"]}

    with pytest.raises(ValueError, match="exactly match"):
        UtilityLabelBank.from_losses(
            feature_bank,
            incumbent_losses_by_layer={1: torch.ones(3)},
            expert_losses_by_layer=experts,
            expert_ids_by_layer=expert_ids,
            loss_name="cross_entropy",
            incumbent_route="anchor",
        )

    wrong_columns = dict(expert_ids)
    wrong_columns[1] = ["code:0"]
    with pytest.raises(ValueError, match="do not match"):
        UtilityLabelBank.from_losses(
            feature_bank,
            incumbent_losses_by_layer=incumbents,
            expert_losses_by_layer=experts,
            expert_ids_by_layer=wrong_columns,
            loss_name="cross_entropy",
            incumbent_route="anchor",
        )

    nonfinite_experts = dict(experts)
    nonfinite_experts[1] = experts[1].clone()
    nonfinite_experts[1][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        UtilityLabelBank.from_losses(
            feature_bank,
            incumbent_losses_by_layer=incumbents,
            expert_losses_by_layer=nonfinite_experts,
            expert_ids_by_layer=expert_ids,
            loss_name="cross_entropy",
            incumbent_route="anchor",
        )
