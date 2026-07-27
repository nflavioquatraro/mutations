"""Acceptance tests for frozen router feature extraction."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import IdentityRouterFeatureExtractor


def test_identity_features_preserve_tensor_storage_and_gradients() -> None:
    extractor = IdentityRouterFeatureExtractor(feature_size=4)
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    features = extractor(hidden_states)
    features.square().sum().backward()

    assert features is hidden_states
    assert features.data_ptr() == hidden_states.data_ptr()
    assert features.dtype == hidden_states.dtype
    assert features.device == hidden_states.device
    assert hidden_states.grad is not None
    assert list(extractor.parameters()) == []
    assert extractor.feature_specification == {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": 4,
        "version": 1,
    }


@pytest.mark.parametrize("feature_size", [0, -1, True, 1.5])
def test_identity_features_require_a_positive_integer_size(
    feature_size: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        IdentityRouterFeatureExtractor(feature_size)  # type: ignore[arg-type]


def test_identity_features_reject_incompatible_inputs() -> None:
    extractor = IdentityRouterFeatureExtractor(feature_size=4)

    with pytest.raises(TypeError, match="torch.Tensor"):
        extractor("not-a-tensor")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        extractor(torch.ones(2, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="final feature"):
        extractor(torch.tensor(1.0))
    with pytest.raises(ValueError, match="does not match"):
        extractor(torch.randn(2, 3, 5))
