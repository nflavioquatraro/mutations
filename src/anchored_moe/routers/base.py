"""Model-family-neutral interfaces for router features and score heads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn


class RouterFeatureExtractor(nn.Module, ABC):
    """Transform pre-MLP hidden states into frozen router features.

    Implementations map hidden states
    ``h[..., hidden_size]`` to features ``phi(h)[..., output_size]``. A feature
    extractor owns neither score heads nor route selection. Any parameters it
    introduces must remain frozen after construction so installing a package
    cannot change the shared router feature space.
    """

    output_size: int

    @property
    @abstractmethod
    def feature_specification(self) -> Mapping[str, Any]:
        """Return JSON-compatible metadata that identifies the feature space."""

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return router features for one pre-MLP hidden-state tensor."""


class RouterHead(nn.Module, ABC):
    """Score one expert from frozen router features at one decoder layer.

    A head is independently owned by the expert identified by ``expert_id``.
    It does not select a route, compare against the anchor, or combine expert
    outputs.
    """

    feature_size: int
    layer_index: int
    expert_id: str

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Return one scalar score per feature vector."""
