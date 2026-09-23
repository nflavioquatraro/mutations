"""Execution contracts for shared-prefix models; no tensor framework imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixState:
    """Backend-owned request snapshot at the branch boundary.

    backend_key binds the base, topology and numerical execution settings.
    request_key identifies exact tokens and masks. payload retains full sequence
    state; its arrays must be immutable. pooled features alone cannot resume a
    suffix. States cannot be reused across incompatible backend instances.
    """

    backend_key: str
    request_key: str
    payload: Any


class BranchBackend(ABC):
    """Model-specific split/resume contract used by routing and training.

    The backend owns a pinned immutable base and declares its trainable suffix
    subset. Methods must not mutate PrefixState or supplied parameter mappings.
    Numeric arrays returned during training retain their native autodiff graph.
    """

    @property
    def forbidden_generation_ids(self) -> tuple[int, ...]:
        """Return token IDs that cannot be fed back into this backend's decoder."""
        return ()

    @property
    def max_sequence_length(self) -> int | None:
        """Return a fixed context limit, or None when the backend has no fixed limit."""
        return None

    @property
    @abstractmethod
    def identity(self) -> dict:
        """Return JSON-compatible base, runtime and split identity metadata."""

    @property
    @abstractmethod
    def suffix_parameters(self) -> Mapping[str, Any]:
        """Return immutable original tensors eligible for branch specialization."""

    @property
    @abstractmethod
    def feature_size(self) -> int:
        """Return the pooled feature width for prefix and suffix scorers."""

    @abstractmethod
    def prefill(self, input_ids: Any) -> PrefixState:
        """Encode a nonempty unpadded [1, sequence] request to the split boundary."""

    @abstractmethod
    def features(self, state: PrefixState, suffix: Mapping | None = None) -> Any:
        """Return [feature_size] pooled prefix or complete suffix features."""

    @abstractmethod
    def logits(self, state: PrefixState, suffix: Mapping) -> Any:
        """Resume a suffix and return [1, sequence, vocabulary] logits."""

    @abstractmethod
    def native_parameters(self, parameters: Mapping) -> Mapping:
        """Convert parameter leaves to the backend's differentiable array type."""
