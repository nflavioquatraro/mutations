"""Immutable expert-mixture compatibility metadata for active registries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from anchored_moe.registry.manifest import PackageValidationError
from anchored_moe.registry.package import ExpertPackage


class MixtureCompatibilityError(ValueError):
    """Raised when package metadata cannot define one compatibility index."""


@dataclass(frozen=True)
class MixtureCompatibilityIndex:
    """Align stable expert identifiers with explicit mixture groups.

    ``group_ids[i]`` owns ``expert_ids[i]``. A ``None`` group is deliberately
    different from a shared default group: it means the expert is not eligible
    for any mixture. Two distinct experts are compatible exactly when both
    declare the same non-empty group identifier.

    The index contains no tensors or local package paths. It can therefore be
    passed independently to a selection policy while package loading and
    expert execution remain separate responsibilities.

    Attributes:
        expert_ids: Unique stable expert identifiers.
        group_ids: Aligned compatibility groups or ``None`` for non-mixable
            experts.
    """

    expert_ids: tuple[str, ...]
    group_ids: tuple[str | None, ...]

    def __post_init__(self) -> None:
        """Validate aligned identifiers and defensively normalize sequences."""

        if isinstance(self.expert_ids, (str, bytes)) or not isinstance(
            self.expert_ids,
            Sequence,
        ):
            raise TypeError("expert_ids must be a sequence of strings.")
        if isinstance(self.group_ids, (str, bytes)) or not isinstance(
            self.group_ids,
            Sequence,
        ):
            raise TypeError("group_ids must be a sequence of strings or None.")
        if len(self.expert_ids) != len(self.group_ids):
            raise ValueError("expert_ids and group_ids must have equal length.")

        normalized_experts = _normalize_expert_sequence(self.expert_ids)
        normalized_groups = tuple(
            None
            if group_id is None
            else _normalize_identifier(group_id, field_name="group_ids")
            for group_id in self.group_ids
        )
        object.__setattr__(self, "expert_ids", normalized_experts)
        object.__setattr__(self, "group_ids", normalized_groups)

    @classmethod
    def from_packages(
        cls,
        packages: Sequence[ExpertPackage],
        *,
        layer_index: int,
    ) -> MixtureCompatibilityIndex:
        """Build one deterministic layer index from ordered package manifests.

        Package order is retained, and expert identifiers within each package
        follow the canonical payload-binding order. Explicit payload bindings
        are required because filenames never imply expert ownership.

        Args:
            packages: Active package descriptors in registry order.
            layer_index: Converted decoder layer whose expert IDs are indexed.

        Returns:
            An immutable expert-to-group index for the requested layer.

        Raises:
            TypeError: If inputs do not implement their declared interfaces.
            ValueError: If ``layer_index`` is invalid.
            MixtureCompatibilityError:
                If bindings are absent, layer coverage is incomplete, or
                active packages repeat a stable expert identifier.
        """

        if isinstance(packages, (str, bytes)) or not isinstance(
            packages,
            Sequence,
        ):
            raise TypeError("packages must be a sequence of ExpertPackage.")
        if not all(isinstance(package, ExpertPackage) for package in packages):
            raise TypeError("packages must contain only ExpertPackage values.")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")

        expert_ids: list[str] = []
        group_ids: list[str | None] = []
        for package in packages:
            if layer_index not in package.manifest.converted_layers:
                raise MixtureCompatibilityError(
                    f"Package {package.identity.version_key!r} does not cover "
                    f"converted layer {layer_index}."
                )
            try:
                bindings = package.payload_bindings
            except PackageValidationError as error:
                raise MixtureCompatibilityError(
                    f"Package {package.identity.version_key!r} cannot provide "
                    f"mixture compatibility: {error}"
                ) from error

            layer_expert_ids = tuple(
                binding.expert_id
                for binding in bindings
                if binding.layer_index == layer_index
            )
            if len(layer_expert_ids) != package.manifest.expert_count:
                raise MixtureCompatibilityError(
                    f"Package {package.identity.version_key!r} has incomplete "
                    f"expert bindings at layer {layer_index}."
                )
            expert_ids.extend(layer_expert_ids)
            group_ids.extend(
                [package.manifest.mixture_compatibility_group]
                * len(layer_expert_ids)
            )

        duplicates = sorted(
            expert_id
            for expert_id in set(expert_ids)
            if expert_ids.count(expert_id) > 1
        )
        if duplicates:
            raise MixtureCompatibilityError(
                "Active packages repeat expert identifiers at layer "
                f"{layer_index}: {duplicates}."
            )
        return cls(tuple(expert_ids), tuple(group_ids))

    def groups_for(self, expert_ids: Sequence[str]) -> tuple[str | None, ...]:
        """Return groups aligned to an exact candidate expert set.

        Candidate order may differ from index construction order, but no
        expert may be omitted or introduced. Requiring exact set equality
        prevents a stale compatibility index from silently governing another
        registry.
        """

        normalized = _normalize_expert_sequence(expert_ids)
        if set(normalized) != set(self.expert_ids) or len(normalized) != len(
            self.expert_ids
        ):
            raise ValueError(
                "expert_ids must exactly match the compatibility index."
            )
        group_by_expert = dict(zip(self.expert_ids, self.group_ids, strict=True))
        return tuple(group_by_expert[expert_id] for expert_id in normalized)

    def group_for(self, expert_id: str) -> str | None:
        """Return one expert's group, raising ``KeyError`` when it is unknown."""

        normalized = _normalize_identifier(expert_id, field_name="expert_id")
        try:
            index = self.expert_ids.index(normalized)
        except ValueError as error:
            raise KeyError(f"Unknown expert_id: {normalized!r}.") from error
        return self.group_ids[index]

    def are_compatible(self, first_expert_id: str, second_expert_id: str) -> bool:
        """Return whether two distinct experts share one explicit group."""

        if first_expert_id == second_expert_id:
            return False
        first_group = self.group_for(first_expert_id)
        second_group = self.group_for(second_expert_id)
        return first_group is not None and first_group == second_group


def _normalize_expert_sequence(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate one candidate expert sequence without importing policies."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(
        expert_ids,
        Sequence,
    ):
        raise TypeError("expert_ids must be a sequence of strings.")
    normalized = tuple(
        _normalize_identifier(expert_id, field_name="expert_ids")
        for expert_id in expert_ids
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("expert_ids must not contain duplicates.")
    return normalized


def _normalize_identifier(value: object, *, field_name: str) -> str:
    """Validate a stable non-empty identifier without changing its spelling."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise ValueError(
            f"{field_name} must contain non-empty strings without surrounding "
            "whitespace."
        )
    return value
