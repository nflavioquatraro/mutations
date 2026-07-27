"""Frozen router feature transformations."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from torch import Tensor

from anchored_moe.routers.base import RouterFeatureExtractor


class IdentityRouterFeatureExtractor(RouterFeatureExtractor):
    """Use pre-MLP hidden states directly as frozen router features.

    For hidden states ``h`` with shape ``[..., feature_size]``, this extractor
    implements ``phi(h) = h``. It has no parameters, performs no allocation or
    detachment, and therefore preserves dtype, device, storage, and any
    caller-controlled gradient flow. "Frozen" describes the feature
    transformation; gradient stopping and feature caching remain separate
    calibration concerns.

    Args:
        feature_size: Required size of the final hidden-state dimension.
    """

    feature_type = "identity"
    input_name = "pre_mlp_hidden_state"

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        if (
            isinstance(feature_size, bool)
            or not isinstance(feature_size, int)
            or feature_size <= 0
        ):
            raise ValueError("feature_size must be a positive integer.")
        self.output_size = feature_size
        self._feature_specification: Mapping[str, Any] = MappingProxyType(
            {
                "feature_type": self.feature_type,
                "input": self.input_name,
                "output_size": feature_size,
                "version": 1,
            }
        )

    @property
    def feature_specification(self) -> Mapping[str, Any]:
        """Return the immutable identity-feature compatibility metadata."""

        return self._feature_specification

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Validate and return ``hidden_states`` without changing its storage.

        Args:
            hidden_states:
                Floating-point pre-MLP tensor with shape
                ``[..., feature_size]``.

        Raises:
            TypeError: If the input is not a floating-point tensor.
            ValueError: If it has no feature axis or the feature size differs.
        """

        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must use a floating-point dtype.")
        if hidden_states.ndim < 1:
            raise ValueError("hidden_states must have a final feature dimension.")
        if hidden_states.shape[-1] != self.output_size:
            raise ValueError(
                "hidden_states feature dimension does not match the extractor: "
                f"{hidden_states.shape[-1]} != {self.output_size}."
            )
        return hidden_states
