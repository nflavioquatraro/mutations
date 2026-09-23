"""Request scoring, deterministic selection and held-out utility calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RequestScores:
    """Ordered finite branch scores with explicit utility/probability semantics.

    Utility is predicted improvement over base. Probability is a normalized
    categorical distribution, including base. Neither implies calibration.
    """

    branch_ids: tuple[str, ...]
    values: tuple[float, ...]
    kind: str = "utility"

    def __post_init__(self) -> None:
        """Reject malformed, nonfinite or incorrectly normalized scores."""
        if (
            not isinstance(self.branch_ids, tuple)
            or not self.branch_ids
            or any(not isinstance(i, str) or not i for i in self.branch_ids)
            or len(set(self.branch_ids)) != len(self.branch_ids)
        ):
            raise ValueError("Scores need ordered unique branch IDs.")
        if (
            not isinstance(self.values, tuple)
            or len(self.values) != len(self.branch_ids)
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("Scores need one finite value per branch.")
        if self.kind not in {"utility", "probability"}:
            raise ValueError("Unknown score kind.")
        if self.kind == "probability" and (
            min(self.values) < 0
            or max(self.values) > 1
            or not np.isclose(sum(self.values), 1, atol=1e-6)
        ):
            raise ValueError("Probability scores must be nonnegative and sum to one.")


@dataclass(frozen=True)
class BranchRoute:
    """Auditable whole-response decision; score is not necessarily confidence."""

    branch_id: str
    score: float | None
    score_kind: str | None
    reason: str


class RequestScorer(Protocol):
    """Score branch candidates using immutable pre-response features only."""

    def score(self, features) -> RequestScores:
        """Map a one-dimensional feature vector to typed candidate scores."""
        ...


@dataclass(frozen=True, init=False, eq=False)
class LinearRequestScorer:
    """Frozen affine head over prefix or router-suffix features.

    weights: [feature, candidates], bias: [candidates]. Inputs are privately
    copied. Utility returns affine values; probability applies temperature
    softmax. The runtime independently chooses which feature path to execute.
    """

    branch_ids: tuple[str, ...]
    kind: str
    temperature: float
    _weights: np.ndarray
    _bias: np.ndarray

    def __init__(self, branch_ids, weights, bias, *, kind="utility", temperature=1.0):
        """Validate and snapshot finite floating-point parameters."""
        object.__setattr__(self, "branch_ids", tuple(branch_ids))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(
            self, "_weights", np.array(weights, dtype=np.float32, copy=True)
        )
        object.__setattr__(self, "_bias", np.array(bias, dtype=np.float32, copy=True))
        if (
            self._weights.ndim != 2
            or self._weights.shape[0] == 0
            or self._weights.shape[1] != len(self.branch_ids)
            or self._bias.shape != (len(self.branch_ids),)
        ):
            raise ValueError("Invalid affine scorer dimensions.")
        if (
            not np.isfinite(self._weights).all()
            or not np.isfinite(self._bias).all()
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError(
                "Scorer parameters must be finite; temperature must be positive."
            )
        RequestScores(
            self.branch_ids,
            tuple([1 / len(self.branch_ids)] * len(self.branch_ids))
            if self.branch_ids
            else (),
            kind,
        )
        self._weights.setflags(write=False)
        self._bias.setflags(write=False)

    @property
    def parameters(self) -> dict:
        """Return private copies of the two affine tensors for training/export."""
        return {"weight": self._weights.copy(), "bias": self._bias.copy()}

    @property
    def feature_size(self) -> int:
        """Return the expected one-dimensional feature width."""
        return self._weights.shape[0]

    def score(self, features) -> RequestScores:
        """Evaluate affine utilities or temperature-scaled categorical scores."""
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (self.feature_size,) or not np.isfinite(features).all():
            raise ValueError("Invalid routing feature vector.")
        values = features @ self._weights + self._bias
        if self.kind == "probability":
            values = values / self.temperature
            values = np.exp(values - values.max())
            values /= values.sum()
        return RequestScores(
            self.branch_ids, tuple(float(v) for v in values), self.kind
        )

    @classmethod
    def fit_ridge(cls, features, utilities, branch_ids, *, regularization=1e-3):
        """Fit independent utility heads with an unpenalized intercept.

        features: [requests, feature], utilities: [requests, candidates]. Solves
        in float64, publishes float32. This does not fit routing thresholds.
        """
        x, y = (
            np.asarray(features, dtype=np.float64),
            np.asarray(utilities, dtype=np.float64),
        )
        if (
            x.ndim != 2
            or not x.shape[0]
            or y.shape != (x.shape[0], len(branch_ids))
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Expected finite aligned feature and utility matrices.")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be positive.")
        xc, yc = x - x.mean(0), y - y.mean(0)
        weights = np.linalg.solve(
            xc.T @ xc + regularization * np.eye(x.shape[1]), xc.T @ yc
        )
        return cls(branch_ids, weights, y.mean(0) - x.mean(0) @ weights)


class RequestSelectionPolicy(Protocol):
    """Choose one response branch independently of feature extraction/training."""

    def select(self, scores: RequestScores, fallback: str) -> BranchRoute:
        """Return a deterministic selected branch or the explicit fallback."""
        ...


@dataclass(frozen=True)
class FixedBranchSelection:
    """Bypass scoring for forced-path evaluation and equivalence checks."""

    branch_id: str

    def select(self, scores: RequestScores | None, fallback: str) -> BranchRoute:
        """Return the configured branch; the owner validates its existence."""
        return BranchRoute(self.branch_id, None, None, "fixed")


@dataclass(frozen=True)
class ArgmaxSelection:
    """Stable argmax; utility compares against fixed-zero base evidence."""

    def select(self, scores: RequestScores, fallback: str) -> BranchRoute:
        """Prefer base on a tie; otherwise preserve declared candidate order."""
        best = max(scores.values)
        if scores.kind == "utility" and best <= 0:
            return BranchRoute(fallback, 0.0, scores.kind, "base_evidence")
        if (
            fallback in scores.branch_ids
            and scores.values[scores.branch_ids.index(fallback)] == best
        ):
            return BranchRoute(fallback, best, scores.kind, "base_tie")
        index = scores.values.index(best)
        return BranchRoute(scores.branch_ids[index], best, scores.kind, "argmax")


@dataclass(frozen=True)
class ThresholdSelection:
    """Thresholded argmax with explicit score semantics and optional calibration.

    thresholds: One finite threshold for each specialist; fallback needs none.
    composition_key: If fitted, binds base, branches and scorer. The runtime
        rejects it after any of those change. None means a manual policy.
    calibration_data_id: Caller-supplied identity of held-out calibration data.
    """

    thresholds: tuple[tuple[str, float], ...]
    kind: str = "utility"
    composition_key: str | None = None
    calibration_data_id: str | None = None

    def __post_init__(self) -> None:
        """Reject duplicate thresholds and utility/probability confusion."""
        if self.kind not in {"utility", "probability"} or not isinstance(
            self.thresholds, tuple
        ):
            raise ValueError("Invalid threshold policy.")
        ids = []
        for pair in self.thresholds:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "Threshold entries must be immutable (ID, value) pairs."
                )
            key, value = pair
            if not isinstance(key, str) or not key:
                raise ValueError("Threshold IDs must be nonempty strings.")
            ids.append(key)
            if not np.isfinite(value) or (
                self.kind == "probability" and not 0 <= value <= 1
            ):
                raise ValueError("Invalid threshold value.")
        if len(ids) != len(set(ids)):
            raise ValueError("Threshold IDs must be unique.")
        if (self.composition_key is None) != (self.calibration_data_id is None):
            raise ValueError(
                "Calibration identity and composition binding are required together."
            )
        if self.composition_key is not None and any(
            not isinstance(v, str) or not v.strip()
            for v in (self.composition_key, self.calibration_data_id)
        ):
            raise ValueError("Calibration identities must be nonempty strings.")

    def select(self, scores: RequestScores, fallback: str) -> BranchRoute:
        """Test the best candidate; failed thresholds abstain without reranking."""
        if scores.kind != self.kind:
            raise ValueError("Selection policy and scorer kinds differ.")
        thresholds = dict(self.thresholds)
        if set(thresholds) != set(scores.branch_ids) - {fallback}:
            raise ValueError("Thresholds must cover exactly the specialist candidates.")
        route = ArgmaxSelection().select(scores, fallback)
        if route.branch_id != fallback and route.score <= thresholds[route.branch_id]:
            return BranchRoute(fallback, route.score, scores.kind, "below_threshold")
        return route


def calibrate_utility_selection(
    scores,
    utilities,
    branch_ids,
    *,
    composition_key: str,
    data_id: str,
    max_false_activation_rate: float = 0.01,
) -> ThresholdSelection:
    """Fit per-head thresholds on held-out requests, using strict > activation.

    Both matrices are [requests, specialists]. False activation is the fraction
    of nonbeneficial rows (utility <= 0) exceeding that head's threshold. This is
    a per-head empirical constraint, not a global or out-of-distribution bound.
    Caller must keep this data disjoint from fitting and final evaluation.
    """
    s, u = np.asarray(scores, dtype=np.float64), np.asarray(utilities, dtype=np.float64)
    if (
        s.ndim != 2
        or not s.shape[0]
        or s.shape != u.shape
        or s.shape[1] != len(branch_ids)
        or not np.isfinite(s).all()
        or not np.isfinite(u).all()
    ):
        raise ValueError("Expected finite aligned score and utility matrices.")
    if not composition_key or not data_id or not 0 <= max_false_activation_rate <= 1:
        raise ValueError(
            "Calibration needs identities and a false-activation budget in [0, 1]."
        )
    thresholds = []
    for index, branch_id in enumerate(branch_ids):
        negative = u[:, index] <= 0
        if not negative.any():
            raise ValueError(
                "Each specialist needs nonbeneficial calibration examples."
            )
        candidates = np.unique(np.concatenate(([0.0], s[:, index][s[:, index] >= 0])))
        threshold = next(
            float(t)
            for t in candidates
            if np.mean(s[negative, index] > t) <= max_false_activation_rate
        )
        thresholds.append((branch_id, threshold))
    return ThresholdSelection(tuple(thresholds), "utility", composition_key, data_id)
