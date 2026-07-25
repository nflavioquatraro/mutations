"""Model-family-neutral routed components."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class RoutedMLP(nn.Module):
    """Wrap an immutable dense MLP with explicit forced-anchor execution.

    The wrapper holds the original MLP object rather than copying it. Its
    foundation behavior is therefore exactly one call to the same anchor
    module; it performs no arithmetic, routing, or tensor reshaping around the
    call. Future routing modes will remain opt-in and must preserve this exact
    fallback path.

    Args:
        anchor_mlp: Original dense MLP module owned by the converted layer.
    """

    routing_mode = "forced_anchor"

    def __init__(self, anchor_mlp: nn.Module) -> None:
        super().__init__()
        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if isinstance(anchor_mlp, RoutedMLP):
            raise ValueError("A RoutedMLP cannot wrap another RoutedMLP.")

        self.anchor_mlp = anchor_mlp
        # Freezing is mechanical and local: conversion changes no tensor value,
        # while every anchor parameter is made ineligible for optimization.
        self.anchor_mlp.requires_grad_(False)

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute the anchor MLP with coefficient one and no extra operations."""

        return self.anchor_mlp(hidden_states, *args, **kwargs)

    def extra_repr(self) -> str:
        """Describe the only routing mode currently supported."""

        return f"routing_mode={self.routing_mode!r}"
