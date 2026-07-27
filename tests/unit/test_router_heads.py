"""Acceptance tests for independent append-only affine router heads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    RouterSerializationError,
)


def _fitted_head(
    *,
    layer_index: int = 2,
    expert_id: str = "code",
) -> AffineRouterHead:
    head = AffineRouterHead(
        4,
        layer_index=layer_index,
        expert_id=expert_id,
        temperature=2.0,
        threshold=0.25,
    )
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[0.5, -1.0, 1.5, 0.25]]))
        head.linear.bias.fill_(0.75)
    return head


def test_affine_head_matches_calibrated_score_equation() -> None:
    head = _fitted_head()
    features = torch.tensor(
        [[[1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 2.0, -2.0]]]
    )

    scores = head(features)
    raw_scores = torch.nn.functional.linear(
        features,
        head.linear.weight,
        head.linear.bias,
    ).squeeze(-1)

    assert scores.shape == (1, 2)
    assert torch.equal(scores, raw_scores / 2.0 - 0.25)
    assert head.key == (2, "code")


def test_new_affine_head_is_neutral_and_trainability_is_explicit() -> None:
    trainable = AffineRouterHead(4, layer_index=0, expert_id="code")
    frozen = AffineRouterHead(
        4,
        layer_index=0,
        expert_id="math",
        trainable=False,
    )
    features = torch.randn(2, 3, 4)

    assert torch.equal(trainable(features), torch.zeros(2, 3))
    assert all(parameter.requires_grad for parameter in trainable.parameters())
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_append_preserves_old_heads_scores_and_layer_order() -> None:
    features = torch.randn(2, 3, 4)
    code = _fitted_head(expert_id="code")
    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(code)
    old_parameters = {
        name: (parameter.detach().clone(), parameter.data_ptr())
        for name, parameter in code.named_parameters()
    }
    old_scores = bank(features, layer_index=2)

    math = _fitted_head(expert_id="math")
    other_layer = _fitted_head(layer_index=3, expert_id="code")
    bank.append(math)
    bank.append(other_layer)
    new_scores = bank(features, layer_index=2)

    assert bank.head_keys == ((2, "code"), (2, "math"), (3, "code"))
    assert bank.expert_ids_for_layer(2) == ("code", "math")
    assert new_scores.shape == (2, 3, 2)
    assert torch.equal(new_scores[..., :1], old_scores)
    for name, parameter in code.named_parameters():
        old_value, old_data_ptr = old_parameters[name]
        assert torch.equal(parameter, old_value)
        assert parameter.data_ptr() == old_data_ptr


def test_bank_rejects_duplicate_incompatible_or_missing_heads() -> None:
    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(_fitted_head())

    with pytest.raises(ValueError, match="already installed"):
        bank.append(_fitted_head())
    with pytest.raises(ValueError, match="feature size"):
        bank.append(AffineRouterHead(5, layer_index=2, expert_id="math"))
    with pytest.raises(ValueError, match="one dtype"):
        bank.append(
            AffineRouterHead(4, layer_index=2, expert_id="math").to(
                dtype=torch.float64
            )
        )
    with pytest.raises(TypeError, match="AffineRouterHead"):
        bank.append(torch.nn.Linear(4, 1))  # type: ignore[arg-type]
    with pytest.raises(LookupError, match="layer 3"):
        bank(torch.randn(2, 4), layer_index=3)


def test_affine_router_safetensors_round_trip_is_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(_fitted_head(expert_id="code"))
    bank.append(_fitted_head(expert_id="math"))
    features = torch.randn(2, 3, 4)
    first_path = tmp_path / "router-a.safetensors"
    second_path = tmp_path / "router-b.safetensors"

    bank.save_safetensors(first_path)
    bank.save_safetensors(second_path)
    restored = AppendOnlyRouterHeads.from_safetensors(first_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert restored.head_keys == bank.head_keys
    assert restored.expert_ids_for_layer(2) == ("code", "math")
    assert torch.equal(
        restored(features, layer_index=2),
        bank(features, layer_index=2),
    )
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_affine_router_load_can_restore_trainable_heads(tmp_path: Path) -> None:
    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(_fitted_head())
    payload_path = tmp_path / "router.safetensors"
    bank.save_safetensors(payload_path)

    restored = AppendOnlyRouterHeads.from_safetensors(
        payload_path,
        trainable=True,
    )

    assert all(parameter.requires_grad for parameter in restored.parameters())


def test_affine_router_serialization_protects_destinations(tmp_path: Path) -> None:
    empty = AppendOnlyRouterHeads(feature_size=4)
    with pytest.raises(RouterSerializationError, match="empty"):
        empty.save_safetensors(tmp_path / "empty.safetensors")

    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(_fitted_head())
    payload_path = tmp_path / "router.safetensors"
    bank.save_safetensors(payload_path)
    original_bytes = payload_path.read_bytes()

    with pytest.raises(FileExistsError):
        bank.save_safetensors(payload_path)
    with pytest.raises(RouterSerializationError, match="end in"):
        bank.save_safetensors(tmp_path / "router.bin")
    with pytest.raises(RouterSerializationError, match="parent"):
        bank.save_safetensors(tmp_path / "missing" / "router.safetensors")

    bank.save_safetensors(payload_path, overwrite=True)
    assert payload_path.read_bytes() == original_bytes


def test_affine_router_rejects_missing_metadata_and_tensor_schema(
    tmp_path: Path,
) -> None:
    missing_metadata = tmp_path / "missing-metadata.safetensors"
    save_file({"heads.0.linear.weight": torch.zeros(1, 4)}, missing_metadata)
    with pytest.raises(RouterSerializationError, match="metadata"):
        AppendOnlyRouterHeads.from_safetensors(missing_metadata)

    bank = AppendOnlyRouterHeads(feature_size=4)
    bank.append(_fitted_head())
    valid_path = tmp_path / "valid.safetensors"
    invalid_path = tmp_path / "invalid.safetensors"
    bank.save_safetensors(valid_path)
    tensors = load_file(valid_path)
    tensors.pop("heads.0.linear.bias")
    with safe_open(valid_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    save_file(tensors, invalid_path, metadata=metadata)

    with pytest.raises(RouterSerializationError, match="state keys"):
        AppendOnlyRouterHeads.from_safetensors(invalid_path)


def test_affine_router_rejects_nonfinite_state(tmp_path: Path) -> None:
    bank = AppendOnlyRouterHeads(feature_size=4)
    head = _fitted_head()
    with torch.no_grad():
        head.linear.weight[0, 0] = torch.nan
    bank.append(head)

    with pytest.raises(RouterSerializationError, match="non-finite"):
        bank.save_safetensors(tmp_path / "router.safetensors")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_size": 0, "layer_index": 0, "expert_id": "code"},
        {"feature_size": 4, "layer_index": -1, "expert_id": "code"},
        {"feature_size": 4, "layer_index": 0, "expert_id": ""},
        {
            "feature_size": 4,
            "layer_index": 0,
            "expert_id": "code",
            "temperature": 0.0,
        },
        {
            "feature_size": 4,
            "layer_index": 0,
            "expert_id": "code",
            "threshold": float("nan"),
        },
    ],
)
def test_affine_head_rejects_invalid_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AffineRouterHead(**kwargs)  # type: ignore[arg-type]


def test_affine_head_rejects_incompatible_features() -> None:
    head = _fitted_head()

    with pytest.raises(TypeError, match="floating-point"):
        head(torch.ones(2, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="final feature"):
        head(torch.tensor(1.0))
    with pytest.raises(ValueError, match="does not match"):
        head(torch.randn(2, 5))
    with pytest.raises(ValueError, match="dtype does not match"):
        head(torch.randn(2, 4, dtype=torch.float64))
