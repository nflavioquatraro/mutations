"""Acceptance tests for CAL-001 router feature reference banks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from anchored_moe import (
    FeatureBankSerializationError,
    IdentityRouterFeatureExtractor,
    RouterFeatureBank,
)


def _feature_specification(feature_size: int = 4) -> dict[str, object]:
    return {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": feature_size,
        "version": 1,
    }


def _dataset_provenance() -> dict[str, object]:
    return {
        "dataset_id": "synthetic-router-reference",
        "revision": "revision-1",
        "split": "train",
        "filters": [{"language": "python"}],
    }


def _bank() -> RouterFeatureBank:
    return RouterFeatureBank(
        {
            3: torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
            1: torch.tensor(
                [[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5]]
            ),
        },
        feature_specification=_feature_specification(),
        dataset_provenance=_dataset_provenance(),
        seed=13,
        row_ids=torch.tensor([10, 20]),
    )


def test_feature_batches_flatten_deterministically_and_detach_sources() -> None:
    extractor = IdentityRouterFeatureExtractor(4)
    first = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    first.requires_grad_()
    second = torch.arange(8, 12, dtype=torch.float32).reshape(1, 1, 4)
    bank = RouterFeatureBank.from_feature_batches(
        {
            3: [extractor(first), extractor(second)],
            1: [extractor(first + 1), extractor(second + 1)],
        },
        feature_specification=extractor.feature_specification,
        dataset_provenance=_dataset_provenance(),
        seed=17,
    )
    expected_layer_three = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    assert bank.layer_indices == (1, 3)
    assert bank.row_count == 3
    assert len(bank) == 3
    assert bank.feature_size == 4
    assert bank.dtype == torch.float32
    assert bank.seed == 17
    assert torch.equal(bank.row_ids, torch.arange(3))
    assert torch.equal(bank.features_for_layer(3), expected_layer_three)
    assert not bank.features_for_layer(3).requires_grad
    assert first.requires_grad

    first.detach().zero_()
    assert torch.equal(bank.features_for_layer(3), expected_layer_three)


def test_accessors_are_defensive_and_compatibility_is_exact() -> None:
    bank = _bank()
    features = bank.features_for_layer(1)
    row_ids = bank.row_ids
    specification = bank.feature_specification
    provenance = bank.dataset_provenance

    features.zero_()
    row_ids.zero_()
    specification["output_size"] = 99
    provenance["dataset_id"] = "changed"

    assert torch.count_nonzero(bank.features_for_layer(1)).item() > 0
    assert torch.equal(bank.row_ids, torch.tensor([10, 20]))
    assert bank.feature_specification["output_size"] == 4
    assert bank.dataset_provenance["dataset_id"] == (
        "synthetic-router-reference"
    )
    bank.assert_compatible(_feature_specification())
    with pytest.raises(ValueError, match="incompatible"):
        bank.assert_compatible(_feature_specification(feature_size=5))
    with pytest.raises(KeyError, match="no layer 9"):
        bank.features_for_layer(9)


def test_feature_bank_safetensors_round_trip_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    bank = _bank()
    first_path = tmp_path / "bank-a.safetensors"
    second_path = tmp_path / "bank-b.safetensors"

    bank.save_safetensors(first_path)
    bank.save_safetensors(second_path)
    restored = RouterFeatureBank.from_safetensors(first_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert restored.metadata == bank.metadata
    assert restored.metadata_sha256 == bank.metadata_sha256
    assert restored.layer_indices == (1, 3)
    assert torch.equal(restored.row_ids, bank.row_ids)
    for layer_index in bank.layer_indices:
        assert torch.equal(
            restored.features_for_layer(layer_index),
            bank.features_for_layer(layer_index),
        )


def test_feature_bank_serialization_protects_destinations(tmp_path: Path) -> None:
    bank = _bank()
    payload_path = tmp_path / "bank.safetensors"
    bank.save_safetensors(payload_path)
    original_bytes = payload_path.read_bytes()

    with pytest.raises(FileExistsError):
        bank.save_safetensors(payload_path)
    with pytest.raises(FeatureBankSerializationError, match="end in"):
        bank.save_safetensors(tmp_path / "bank.bin")
    with pytest.raises(FeatureBankSerializationError, match="parent"):
        bank.save_safetensors(tmp_path / "missing" / "bank.safetensors")

    bank.save_safetensors(payload_path, overwrite=True)
    assert payload_path.read_bytes() == original_bytes


def test_feature_bank_rejects_missing_or_malformed_metadata(
    tmp_path: Path,
) -> None:
    missing_metadata = tmp_path / "missing-metadata.safetensors"
    save_file({"row_ids": torch.arange(2)}, missing_metadata)

    with pytest.raises(FeatureBankSerializationError, match="metadata"):
        RouterFeatureBank.from_safetensors(missing_metadata)

    bank = _bank()
    valid_path = tmp_path / "valid.safetensors"
    malformed_path = tmp_path / "malformed.safetensors"
    bank.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["anchored_moe"])
    metadata["unknown"] = True
    save_file(
        tensors,
        malformed_path,
        metadata={"anchored_moe": json.dumps(metadata)},
    )

    with pytest.raises(FeatureBankSerializationError, match="fields"):
        RouterFeatureBank.from_safetensors(malformed_path)


def test_feature_bank_rejects_tensor_schema_and_digest_changes(
    tmp_path: Path,
) -> None:
    bank = _bank()
    valid_path = tmp_path / "valid.safetensors"
    missing_path = tmp_path / "missing.safetensors"
    changed_path = tmp_path / "changed.safetensors"
    bank.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    with safe_open(valid_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()

    missing_tensors = dict(tensors)
    missing_tensors.pop("layers.1.features")
    save_file(missing_tensors, missing_path, metadata=metadata)
    with pytest.raises(FeatureBankSerializationError, match="tensor keys"):
        RouterFeatureBank.from_safetensors(missing_path)

    changed_tensors = dict(tensors)
    changed_tensors["layers.1.features"] = (
        changed_tensors["layers.1.features"] + 1.0
    )
    save_file(changed_tensors, changed_path, metadata=metadata)
    with pytest.raises(FeatureBankSerializationError, match="does not match"):
        RouterFeatureBank.from_safetensors(changed_path)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        ({}, "must not be empty"),
        ({1: torch.ones(2, 5)}, "output_size"),
        ({1: torch.ones(2, 4), 2: torch.ones(3, 4)}, "row count"),
        (
            {
                1: torch.ones(2, 4, dtype=torch.float32),
                2: torch.ones(2, 4, dtype=torch.float64),
            },
            "same dtype",
        ),
        ({1: torch.tensor([[1.0, float("nan"), 2.0, 3.0]])}, "finite"),
    ],
)
def test_feature_bank_rejects_invalid_feature_collections(
    features: dict[int, torch.Tensor],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        RouterFeatureBank(
            features,
            feature_specification=_feature_specification(),
            dataset_provenance=_dataset_provenance(),
        )


def test_feature_bank_rejects_invalid_provenance_and_row_ids() -> None:
    features = {1: torch.ones(2, 4)}

    with pytest.raises(ValueError, match="missing required"):
        RouterFeatureBank(
            features,
            feature_specification=_feature_specification(),
            dataset_provenance={"dataset_id": "missing-fields"},
        )
    provenance_with_content = _dataset_provenance()
    provenance_with_content["prompt"] = "raw example"
    with pytest.raises(ValueError, match="example content"):
        RouterFeatureBank(
            features,
            feature_specification=_feature_specification(),
            dataset_provenance=provenance_with_content,
        )
    with pytest.raises(ValueError, match="unique"):
        RouterFeatureBank(
            features,
            feature_specification=_feature_specification(),
            dataset_provenance=_dataset_provenance(),
            row_ids=torch.tensor([1, 1]),
        )
    with pytest.raises(TypeError, match="torch.long"):
        RouterFeatureBank(
            features,
            feature_specification=_feature_specification(),
            dataset_provenance=_dataset_provenance(),
            row_ids=torch.tensor([1.0, 2.0]),
        )
