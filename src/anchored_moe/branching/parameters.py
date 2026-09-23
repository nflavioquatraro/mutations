"""Independent suffix parameter storage and differentiable materialization.

References:
    Hu et al., LoRA: https://arxiv.org/abs/2106.09685

Differences:
    Updates apply only to explicitly resolved suffix matrices. Frozen shared
    tensors are never optimizer arguments. The algebra works with NumPy or JAX.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from .configuration import BranchConfig


class BranchParameterization(Protocol):
    """Initialize owned tensors and compose them with immutable suffix weights.

    Keys are backend-resolved parameter paths; arrays preserve native gradients.
    Implementations must not mutate base or learned mappings or tensor storage.
    """

    def initialize(self, base: Mapping[str, Any], config: BranchConfig) -> dict:
        """Return independently owned initial tensors, with deterministic seeds."""
        ...

    def materialize(
        self, base: Mapping[str, Any], learned: Mapping[str, Any], config: BranchConfig
    ) -> dict:
        """Return effective suffix tensors without breaking native gradients."""
        ...


class FrozenSuffix:
    """Original suffix with no trainable or serialized branch tensors."""

    def initialize(self, base, config) -> dict:
        """Return an empty optimizer subset."""
        return {}

    def materialize(self, base, learned, config) -> dict:
        """Return the original tensor references; reject unexpected updates."""
        if learned:
            raise ValueError("A frozen branch cannot carry learned tensors.")
        return dict(base)


class CopiedSuffix:
    """Independent full copies of the backend's declared suffix tensor subset."""

    def initialize(self, base, config) -> dict:
        """Copy every suffix tensor, preserving dtype and original values."""
        return {key: np.array(value, copy=True) for key, value in base.items()}

    def materialize(self, base, learned, config) -> dict:
        """Use the complete owned suffix; topology is checked at publication."""
        return dict(learned)


class LoRASuffix:
    """Low-rank updates to resolved [input, output] attention matrices.

    A is [input, rank], B is [rank, output]; zero B gives exact initial fallback.
    Shared base weights remain immutable; effective updates use alpha / rank.
    """

    def initialize(self, base, config) -> dict:
        """Resolve every requested target and create seeded A, zero B tensors."""
        rng = np.random.default_rng(config.seed)
        result, found = {}, set()
        for key, value in sorted(base.items()):
            for target in config.targets:
                if key.endswith(f"/self_attn/{target}/kernel"):
                    if value.ndim != 2:
                        raise ValueError(f"LoRA target is not a matrix: {key}")
                    found.add(target)
                    result[f"{key}/A"] = (
                        rng.standard_normal((value.shape[0], config.rank)) * 0.02
                    ).astype(np.float32)
                    result[f"{key}/B"] = np.zeros(
                        (config.rank, value.shape[1]), np.float32
                    )
        if found != set(config.targets):
            raise ValueError(
                f"Unresolved LoRA targets: {sorted(set(config.targets) - found)}"
            )
        return result

    def materialize(self, base, learned, config) -> dict:
        """Compose updates using native array operations for autodiff."""
        result = dict(base)
        for key in learned:
            if key.endswith("/A"):
                path = key[:-2]
                delta = (config.alpha / config.rank) * (
                    learned[key] @ learned[f"{path}/B"]
                )
                result[path] = base[path] + delta.astype(base[path].dtype)
        return result


def parameterization(config: BranchConfig) -> BranchParameterization:
    """Resolve the three builtin parameter strategies without importing JAX."""
    return {"frozen": FrozenSuffix, "copy": CopiedSuffix, "lora": LoRASuffix}[
        config.parameterization
    ]()


def checked_parameters(candidate: Mapping, template: Mapping) -> dict[str, np.ndarray]:
    """Validate keys/shapes/dtypes/finiteness and return private read-only copies."""
    if set(candidate) != set(template):
        raise ValueError("Branch parameter keys do not match the declared topology.")
    result = {}
    for key, reference in template.items():
        value = np.array(candidate[key], copy=True)
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Branch parameter shape/dtype mismatch: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite branch parameter: {key}")
        value.setflags(write=False)
        result[key] = value
    return result
