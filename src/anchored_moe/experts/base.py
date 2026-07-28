"""Model-family-neutral expert interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from torch import Tensor, nn


class ExpertSerializationError(ValueError):
    """Raised when an expert tensor payload is unsafe or incompatible."""


class Expert(nn.Module, ABC):
    """Base interface for independently trainable expert transformations.

    An expert maps pre-MLP hidden states ``h`` to an MLP output with the same
    calling convention as its anchor module. The interface owns execution,
    parameter accounting, and tensor serialization. It deliberately does not
    own routing, package installation, capacity decisions, or training loops.
    """

    parameterization: ClassVar[str]

    @property
    def parameter_count(self) -> int:
        """Return the total number of expert parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of parameters currently eligible for gradients."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_trainable(self, trainable: bool) -> None:
        """Set the parameter subset owned by this expert's optimizer.

        Full-capacity experts train every owned parameter, so the default
        implementation delegates to :meth:`torch.nn.Module.requires_grad_`.
        Parameter-efficient experts override this method to keep their private
        frozen base parameters disabled while toggling only adapter tensors.

        Args:
            trainable: Whether the expert's optimizable parameter subset should
                require gradients.
        """

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        self.requires_grad_(trainable)

    @abstractmethod
    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Transform pre-MLP hidden states using this expert."""

    @abstractmethod
    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Serialize only expert tensors to a non-executable safetensors file."""
