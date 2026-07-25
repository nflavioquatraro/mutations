"""Stable interface for model-family-specific MLP conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from torch import nn

from anchored_moe.configuration import AnchoredMoEConfig


@dataclass(frozen=True)
class MLPReference:
    """A replaceable MLP and the decoder layer that owns it.

    Attributes:
        layer_index: Zero-based decoder-layer index.
        parent: Module on which ``attribute_name`` is installed.
        attribute_name: Attribute containing the MLP.
        module: Current MLP module at discovery time.
    """

    layer_index: int
    parent: nn.Module
    attribute_name: str
    module: nn.Module


@dataclass(frozen=True)
class ConversionReport:
    """Deterministic summary returned after a backend conversion."""

    backend: str
    total_layers: int
    converted_layers: tuple[int, ...]


class ModelBackend(ABC):
    """Interface separating model-family structure from routing behavior."""

    model_type: str

    @abstractmethod
    def supports_model(self, model: nn.Module) -> bool:
        """Return whether ``model`` exposes the exact supported family contract."""

    @abstractmethod
    def iter_mlp_references(self, model: nn.Module) -> Iterator[MLPReference]:
        """Yield every replaceable decoder MLP in stable layer order."""

    @abstractmethod
    def replace_mlp(
        self,
        reference: MLPReference,
        replacement: nn.Module,
    ) -> None:
        """Replace one discovered MLP while preserving its owning layer."""

    @abstractmethod
    def convert(
        self,
        model: nn.Module,
        config: AnchoredMoEConfig,
    ) -> ConversionReport:
        """Wrap selected MLPs and return the resolved conversion summary."""
