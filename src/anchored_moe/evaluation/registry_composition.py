"""Deterministic installation-order and registry-subset evaluation.

EVAL-006 enumerates bounded package subsets and installation orders without
changing package content, synchronizing a model, or recalibrating a router.
An optional callback may observe each fully composed temporary registry and
return a SHA-256 behavior fingerprint. The evaluator verifies that observation
does not mutate the registry it received.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from anchored_moe.registry.package import ExpertPackage, PackageIdentity
from anchored_moe.registry.registry import ExpertRegistry, RegistrySnapshot


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_MAXIMUM_TRIALS = 2_048


class RegistryCompositionEvaluationError(ValueError):
    """Raised when an EVAL-006 experiment or observation is invalid."""


@dataclass(frozen=True)
class RegistryCompositionTrial:
    """One exact package installation order and its observed result.

    Attributes:
        installation_order:
            Portable package identities in the order installed into a fresh
            registry.
        snapshot:
            Immutable registry state captured after every installation.
        behavior_sha256:
            Optional caller-supplied fingerprint of routes, logits, metrics, or
            another deterministic behavior observed for this registry.
    """

    installation_order: tuple[PackageIdentity, ...]
    snapshot: RegistrySnapshot
    behavior_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate exact trial ordering, snapshot identity, and fingerprint."""

        if not isinstance(self.installation_order, tuple) or not all(
            isinstance(identity, PackageIdentity)
            for identity in self.installation_order
        ):
            raise TypeError(
                "installation_order must be a tuple of PackageIdentity."
            )
        if len(set(self.installation_order)) != len(self.installation_order):
            raise ValueError("installation_order must not repeat a package.")
        if not isinstance(self.snapshot, RegistrySnapshot):
            raise TypeError("snapshot must be a RegistrySnapshot.")
        if self.snapshot.active_packages != self.installation_order:
            raise ValueError(
                "snapshot active_packages must match installation_order."
            )
        if self.behavior_sha256 is not None and (
            not isinstance(self.behavior_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.behavior_sha256) is None
        ):
            raise ValueError(
                "behavior_sha256 must be a lowercase 64-character digest or "
                "None."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible trial mapping."""

        return {
            "behavior_sha256": self.behavior_sha256,
            "compatibility_sha256": self.snapshot.compatibility_sha256,
            "installation_order": [
                identity.to_dict() for identity in self.installation_order
            ],
            "registry_sha256": self.snapshot.registry_sha256,
        }


@dataclass(frozen=True)
class RegistrySubsetEvaluation:
    """All installation orders for one canonical package subset.

    ``package_set`` is sorted by portable package identity and therefore does
    not imply an installation order. ``trials`` contains every permutation in
    deterministic lexicographic order, including one empty-order trial for the
    empty subset.
    """

    package_set: tuple[PackageIdentity, ...]
    trials: tuple[RegistryCompositionTrial, ...]

    def __post_init__(self) -> None:
        """Validate canonical membership and exhaustive permutation coverage."""

        if not isinstance(self.package_set, tuple) or not all(
            isinstance(identity, PackageIdentity) for identity in self.package_set
        ):
            raise TypeError("package_set must be a tuple of PackageIdentity.")
        if self.package_set != tuple(sorted(self.package_set)):
            raise ValueError("package_set must use canonical identity order.")
        if len(set(self.package_set)) != len(self.package_set):
            raise ValueError("package_set must not repeat a package.")
        if not isinstance(self.trials, tuple) or not all(
            isinstance(trial, RegistryCompositionTrial) for trial in self.trials
        ):
            raise TypeError(
                "trials must be a tuple of RegistryCompositionTrial."
            )
        expected_orders = tuple(permutations(self.package_set))
        actual_orders = tuple(trial.installation_order for trial in self.trials)
        if actual_orders != expected_orders:
            raise ValueError(
                "trials must exhaust package-set permutations in deterministic "
                "order."
            )
        observed_modes = {
            trial.behavior_sha256 is not None for trial in self.trials
        }
        if len(observed_modes) != 1:
            raise ValueError(
                "Every trial in a subset must use the same behavior-observation "
                "mode."
            )

    @property
    def package_count(self) -> int:
        """Return the number of active packages in this subset."""

        return len(self.package_set)

    @property
    def installation_order_count(self) -> int:
        """Return the factorial number of evaluated installation orders."""

        return len(self.trials)

    @property
    def distinct_registry_hash_count(self) -> int:
        """Return the number of distinct ordered-registry identities."""

        return len({trial.snapshot.registry_sha256 for trial in self.trials})

    @property
    def registry_order_sensitive(self) -> bool:
        """Return whether installation order changed the registry identity."""

        return self.distinct_registry_hash_count > 1

    @property
    def distinct_behavior_count(self) -> int | None:
        """Return distinct observed behaviors, or ``None`` without a callback."""

        if self.trials[0].behavior_sha256 is None:
            return None
        return len({trial.behavior_sha256 for trial in self.trials})

    @property
    def behavior_order_invariant(self) -> bool | None:
        """Return order invariance, or ``None`` when behavior was not observed."""

        count = self.distinct_behavior_count
        return None if count is None else count == 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible subset report."""

        return {
            "behavior_order_invariant": self.behavior_order_invariant,
            "distinct_behavior_count": self.distinct_behavior_count,
            "distinct_registry_hash_count": self.distinct_registry_hash_count,
            "installation_order_count": self.installation_order_count,
            "package_set": [
                identity.to_dict() for identity in self.package_set
            ],
            "registry_order_sensitive": self.registry_order_sensitive,
            "trials": [trial.to_dict() for trial in self.trials],
        }


@dataclass(frozen=True)
class RegistryCompositionEvaluation:
    """Complete bounded EVAL-006 report for one package collection.

    Attributes:
        package_ids:
            Every supplied package identity in canonical order.
        subset_evaluations:
            Empty and non-empty subsets ordered first by size and then by
            canonical combination order.
        maximum_trials:
            Caller-declared upper bound checked before enumeration.
    """

    package_ids: tuple[PackageIdentity, ...]
    subset_evaluations: tuple[RegistrySubsetEvaluation, ...]
    maximum_trials: int

    def __post_init__(self) -> None:
        """Validate canonical packages, complete subsets, and trial budget."""

        if not isinstance(self.package_ids, tuple) or not all(
            isinstance(identity, PackageIdentity) for identity in self.package_ids
        ):
            raise TypeError("package_ids must be a tuple of PackageIdentity.")
        if self.package_ids != tuple(sorted(self.package_ids)):
            raise ValueError("package_ids must use canonical identity order.")
        if len(set(self.package_ids)) != len(self.package_ids):
            raise ValueError("package_ids must not repeat a package.")
        if not isinstance(self.subset_evaluations, tuple) or not all(
            isinstance(subset, RegistrySubsetEvaluation)
            for subset in self.subset_evaluations
        ):
            raise TypeError(
                "subset_evaluations must be a tuple of "
                "RegistrySubsetEvaluation."
            )
        expected_sets = tuple(
            package_set
            for size in range(len(self.package_ids) + 1)
            for package_set in combinations(self.package_ids, size)
        )
        actual_sets = tuple(
            subset.package_set for subset in self.subset_evaluations
        )
        if actual_sets != expected_sets:
            raise ValueError(
                "subset_evaluations must cover every subset in deterministic "
                "order."
            )
        _validate_positive_integer(
            self.maximum_trials,
            field_name="maximum_trials",
        )
        if self.trial_count > self.maximum_trials:
            raise ValueError("Report trial_count exceeds maximum_trials.")
        observed_modes = {
            subset.behavior_order_invariant is not None
            for subset in self.subset_evaluations
        }
        if len(observed_modes) != 1:
            raise ValueError(
                "Every subset must use the same behavior-observation mode."
            )

    @property
    def subset_count(self) -> int:
        """Return the exact number of evaluated package subsets."""

        return len(self.subset_evaluations)

    @property
    def trial_count(self) -> int:
        """Return the total number of evaluated partial permutations."""

        return sum(
            subset.installation_order_count
            for subset in self.subset_evaluations
        )

    @property
    def registry_order_sensitive_subset_count(self) -> int:
        """Return how many subsets have order-dependent registry hashes."""

        return sum(
            subset.registry_order_sensitive
            for subset in self.subset_evaluations
        )

    @property
    def behavior_order_sensitive_subset_count(self) -> int | None:
        """Return order-sensitive behavior subsets, or ``None`` if unobserved."""

        if self.subset_evaluations[0].behavior_order_invariant is None:
            return None
        return sum(
            subset.behavior_order_invariant is False
            for subset in self.subset_evaluations
        )

    @property
    def behavior_order_invariant(self) -> bool | None:
        """Return global order invariance, or ``None`` if unobserved."""

        count = self.behavior_order_sensitive_subset_count
        return None if count is None else count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible evaluation mapping."""

        return {
            "behavior_order_invariant": self.behavior_order_invariant,
            "behavior_order_sensitive_subset_count": (
                self.behavior_order_sensitive_subset_count
            ),
            "maximum_trials": self.maximum_trials,
            "package_ids": [
                identity.to_dict() for identity in self.package_ids
            ],
            "registry_order_sensitive_subset_count": (
                self.registry_order_sensitive_subset_count
            ),
            "subset_count": self.subset_count,
            "subset_evaluations": [
                subset.to_dict() for subset in self.subset_evaluations
            ],
            "trial_count": self.trial_count,
        }


def evaluate_registry_compositions(
    packages: Sequence[ExpertPackage],
    *,
    behavior_fingerprint: Callable[[ExpertRegistry], str] | None = None,
    maximum_trials: int = _DEFAULT_MAXIMUM_TRIALS,
) -> RegistryCompositionEvaluation:
    """Evaluate every subset and installation order of a small registry.

    The required trial count for ``N`` packages is the number of partial
    permutations

    ``sum_(k=0)^N P(N, k)``.

    The evaluator canonicalizes packages by portable identity, preflights the
    complete package set in one temporary registry, checks the trial budget,
    and only then begins enumeration. Each trial uses a fresh
    :class:`~anchored_moe.registry.ExpertRegistry`, so historical slots from
    one order cannot affect another.

    ``behavior_fingerprint`` is optional and explicitly caller-controlled. It
    receives the fully composed temporary registry and must return a lowercase
    SHA-256 digest. The callback may synchronize or inspect external state, but
    it must not mutate the supplied registry; before/after snapshots enforce
    that boundary. Callback exceptions propagate unchanged.

    Args:
        packages:
            Strictly validated local package descriptors. The empty sequence is
            valid and evaluates the exact empty registry once.
        behavior_fingerprint:
            Optional deterministic observer for route, logit, metric, or other
            behavioral output.
        maximum_trials:
            Positive upper bound checked before subset/permutation execution.
            The default 2,048 permits at most six packages.

    Returns:
        An immutable report with deterministic subset and trial ordering.

    Raises:
        TypeError:
            If packages, callback, or trial-bound types are invalid.
        ValueError:
            If package identities repeat or the trial budget is invalid.
        RegistryCompositionEvaluationError:
            If the experiment exceeds its trial budget, a behavior fingerprint
            is invalid, or the behavior callback mutates its registry.
        RegistryLifecycleError:
            If complete-set preflight finds incompatible package content.
        PackageValidationError:
            If live package files fail validation.
    """

    if isinstance(packages, (str, bytes)) or not isinstance(packages, Sequence):
        raise TypeError("packages must be a sequence of ExpertPackage.")
    package_values = tuple(packages)
    if not all(isinstance(package, ExpertPackage) for package in package_values):
        raise TypeError("packages must contain only ExpertPackage values.")
    if behavior_fingerprint is not None and not callable(behavior_fingerprint):
        raise TypeError("behavior_fingerprint must be callable or None.")
    trial_limit = _validate_positive_integer(
        maximum_trials,
        field_name="maximum_trials",
    )

    packages_by_identity: dict[PackageIdentity, ExpertPackage] = {}
    version_keys: set[tuple[str, str]] = set()
    for package in package_values:
        identity = package.identity
        if identity in packages_by_identity:
            raise ValueError(
                "packages must not repeat an immutable package identity."
            )
        if identity.version_key in version_keys:
            raise ValueError(
                "packages must not contain conflicting content for one "
                "specialization version."
            )
        packages_by_identity[identity] = package
        version_keys.add(identity.version_key)

    package_ids = tuple(sorted(packages_by_identity))
    ordered_packages = tuple(
        packages_by_identity[identity] for identity in package_ids
    )
    required_trials = sum(
        math.perm(len(ordered_packages), size)
        for size in range(len(ordered_packages) + 1)
    )
    if required_trials > trial_limit:
        raise RegistryCompositionEvaluationError(
            "Registry composition requires "
            f"{required_trials} trials, exceeding maximum_trials={trial_limit}."
        )

    # Complete-set preflight ensures duplicate versions, strict validation,
    # and anchor/router-feature compatibility fail before any observer runs.
    preflight_registry = ExpertRegistry()
    for package in ordered_packages:
        preflight_registry.install(package)

    subset_results: list[RegistrySubsetEvaluation] = []
    for size in range(len(ordered_packages) + 1):
        for package_subset in combinations(ordered_packages, size):
            trials = tuple(
                _evaluate_installation_order(
                    installation_order,
                    behavior_fingerprint=behavior_fingerprint,
                )
                for installation_order in permutations(package_subset)
            )
            subset_results.append(
                RegistrySubsetEvaluation(
                    package_set=tuple(
                        package.identity for package in package_subset
                    ),
                    trials=trials,
                )
            )

    return RegistryCompositionEvaluation(
        package_ids=package_ids,
        subset_evaluations=tuple(subset_results),
        maximum_trials=trial_limit,
    )


def _evaluate_installation_order(
    installation_order: tuple[ExpertPackage, ...],
    *,
    behavior_fingerprint: Callable[[ExpertRegistry], str] | None,
) -> RegistryCompositionTrial:
    """Compose one fresh registry and make optional observation read-only."""

    registry = ExpertRegistry()
    for package in installation_order:
        registry.install(package)
    snapshot = registry.snapshot()

    behavior_sha256: str | None = None
    if behavior_fingerprint is not None:
        behavior_sha256 = behavior_fingerprint(registry)
        if registry.snapshot() != snapshot:
            raise RegistryCompositionEvaluationError(
                "behavior_fingerprint mutated the evaluated registry."
            )
        if (
            not isinstance(behavior_sha256, str)
            or _SHA256_PATTERN.fullmatch(behavior_sha256) is None
        ):
            raise RegistryCompositionEvaluationError(
                "behavior_fingerprint must return a lowercase 64-character "
                "SHA-256 digest."
            )

    return RegistryCompositionTrial(
        installation_order=tuple(
            package.identity for package in installation_order
        ),
        snapshot=snapshot,
        behavior_sha256=behavior_sha256,
    )


def _validate_positive_integer(value: Any, *, field_name: str) -> int:
    """Return one positive non-boolean integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value
