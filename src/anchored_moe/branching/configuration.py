"""Immutable topology and parameterization configuration for response branches."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass


def _identifier(value: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value) is None
    ):
        raise ValueError(
            "Branch IDs must start with a letter and contain letters, digits, _ or -."
        )


@dataclass(frozen=True)
class BranchConfig:
    """Declare a suffix; original embeddings, final norm and output head stay shared.

    branch_id: Stable unique identifier, also used in package metadata.
    parameterization: ``frozen``, ``lora`` or ``copy``. Copy trains backend-declared
        suffix tensors; it does not implicitly unfreeze shared auxiliary memory.
    rank, alpha: Positive LoRA rank and scale numerator, used only by LoRA.
    targets: Unique projection names resolved by the backend for LoRA.
    seed: Nonnegative initialization seed; never modifies global RNG state.
    """

    branch_id: str
    parameterization: str = "lora"
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "gate_proj", "out_proj")
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject unsupported strategies and invalid or ambiguous declarations."""
        _identifier(self.branch_id)
        if self.parameterization not in {"frozen", "lora", "copy"}:
            raise ValueError("Unknown branch parameterization.")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ValueError("rank must be a positive integer.")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("alpha must be finite and positive.")
        if (
            not isinstance(self.targets, tuple)
            or not self.targets
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError(
                "targets must be a nonempty tuple of unique projection names."
            )
        for target in self.targets:
            _identifier(target)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")


@dataclass(frozen=True)
class BranchedModelConfig:
    """Select a contiguous frozen prefix and independently owned suffixes.

    shared_prefix_layers: Count, not zero-based index; must leave a nonempty suffix.
    response_branches: Ordered candidates; exactly one named fallback is frozen.
    router_branch: Optional separate suffix used only to extract routing features.
        None selects features directly from the shared boundary.
    fallback_branch: Original frozen response path. Low-evidence routes use it.
    """

    shared_prefix_layers: int
    response_branches: tuple[BranchConfig, ...]
    router_branch: BranchConfig | None = None
    fallback_branch: str = "base"

    def __post_init__(self) -> None:
        """Validate independent IDs, immutable topology and a frozen fallback."""
        if (
            isinstance(self.shared_prefix_layers, bool)
            or not isinstance(self.shared_prefix_layers, int)
            or self.shared_prefix_layers <= 0
        ):
            raise ValueError("shared_prefix_layers must be a positive integer.")
        if (
            not isinstance(self.response_branches, tuple)
            or not self.response_branches
            or any(not isinstance(b, BranchConfig) for b in self.response_branches)
        ):
            raise ValueError(
                "response_branches must be a nonempty tuple of BranchConfig."
            )
        if self.router_branch is not None and not isinstance(
            self.router_branch, BranchConfig
        ):
            raise TypeError("router_branch must be a BranchConfig or None.")
        branches = self.response_branches + (
            (self.router_branch,) if self.router_branch else ()
        )
        ids = [b.branch_id for b in branches]
        if len(ids) != len(set(ids)):
            raise ValueError("Response and router branch IDs must be unique.")
        fallback = next(
            (b for b in self.response_branches if b.branch_id == self.fallback_branch),
            None,
        )
        if fallback is None or fallback.parameterization != "frozen":
            raise ValueError("fallback_branch must name a frozen response branch.")

    def to_dict(self) -> dict:
        """Return JSON-compatible configuration without model or tensor state."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> BranchedModelConfig:
        """Rebuild strict configuration; unknown fields raise TypeError."""

        def branch(raw):
            raw = dict(raw)
            if "targets" in raw:
                raw["targets"] = tuple(raw["targets"])
            return BranchConfig(**raw)

        raw = dict(value)
        raw["response_branches"] = tuple(branch(b) for b in raw["response_branches"])
        if raw.get("router_branch") is not None:
            raw["router_branch"] = branch(raw["router_branch"])
        return cls(**raw)
