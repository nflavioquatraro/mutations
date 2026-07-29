"""Gradient-free expert-selection bias updates for balanced sparse routing.

BAL-004 adds one persistent bias per specialist to the router scores used for
selection. The original calibrated scores remain available separately for
activation weighting, and candidate zero remains the immutable anchor with no
learned or updated bias. After a completed batch, overloaded specialists have
their biases decreased and underloaded specialists have theirs increased.

References:
    Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for
    Mixture-of-Experts." https://arxiv.org/abs/2408.15664

    DeepSeek-AI et al., "DeepSeek-V3 Technical Report."
    https://arxiv.org/abs/2412.19437

Differences from the papers:
    This framework primitive is independent of one MoE training loop and
    accepts explicit completed-batch assignment indices. Candidate zero is
    reserved for the Anchored MoE anchor or inactive padding and is excluded
    from balancing. Optional non-negative assignment weights support padding
    masks. Distributed load aggregation and automatic Trainer-step hooks are
    deliberately outside this initial implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F


LOSS_FREE_BIAS_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = (
    f"anchored_moe.loss_free_selection_bias.v{LOSS_FREE_BIAS_FORMAT_VERSION}"
)
_METADATA_KEY: Final = "anchored_moe"
_SAFETENSORS_SUFFIX: Final = ".safetensors"


class LossFreeBiasBalancingError(ValueError):
    """Raised when BAL-004 scores, assignments, or serialized state are invalid."""


@dataclass(frozen=True)
class LossFreeBiasConfig:
    """Configuration for additive, sign-updated specialist selection biases.

    Args:
        update_rate:
            Finite positive magnitude ``u`` used in
            ``b_i <- b_i + u * sign(mean_load - load_i)`` after each observed
            batch. The paper reports ``1e-3`` as its main sign-update setting.
    """

    update_rate: float = 1e-3

    def __post_init__(self) -> None:
        """Normalize the public update rate to one finite positive float."""

        if (
            isinstance(self.update_rate, bool)
            or not isinstance(self.update_rate, (int, float))
            or not isfinite(self.update_rate)
            or float(self.update_rate) <= 0
        ):
            raise LossFreeBiasBalancingError(
                "update_rate must be a finite positive real number."
            )
        object.__setattr__(self, "update_rate", float(self.update_rate))

    def to_dict(self) -> dict[str, float]:
        """Return the complete JSON-compatible configuration."""

        return {"update_rate": self.update_rate}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LossFreeBiasConfig:
        """Strictly reconstruct a serialized BAL-004 configuration."""

        if not isinstance(value, dict):
            raise LossFreeBiasBalancingError(
                "Loss-free bias config must be a dictionary."
            )
        if set(value) != {"update_rate"}:
            raise LossFreeBiasBalancingError(
                "Loss-free bias config fields are incompatible."
            )
        return cls(update_rate=value["update_rate"])


@dataclass(frozen=True)
class LossFreeBiasedScores:
    """Raw router scores beside their selection-only biased counterparts.

    ``raw_scores`` and ``selection_scores`` have shape
    ``token_shape + (K,)`` in ``expert_ids`` order. ``biases`` has shape
    ``(K,)`` and is a detached snapshot. The anchor is intentionally absent:
    selection policies continue to prepend their exact fixed-zero anchor.

    Activation-combination code must use ``raw_scores``. Only top-k selection
    should consume ``selection_scores``.
    """

    raw_scores: Tensor
    selection_scores: Tensor
    biases: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shape, dtype, device, identifiers, and additive semantics."""

        expert_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", expert_ids)
        for field_name in ("raw_scores", "selection_scores", "biases"):
            value = getattr(self, field_name)
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise LossFreeBiasBalancingError(
                    f"{field_name} must be a floating-point torch.Tensor."
                )
            if not torch.isfinite(value).all().item():
                raise LossFreeBiasBalancingError(
                    f"{field_name} must contain only finite values."
                )
        if self.raw_scores.shape != self.selection_scores.shape:
            raise LossFreeBiasBalancingError(
                "raw_scores and selection_scores must share one shape."
            )
        if self.raw_scores.ndim < 1:
            raise LossFreeBiasBalancingError(
                "Router scores must have a final specialist dimension."
            )
        if self.raw_scores.shape[-1] != len(expert_ids):
            raise LossFreeBiasBalancingError(
                "Router score width must match expert_ids."
            )
        if self.biases.shape != (len(expert_ids),):
            raise LossFreeBiasBalancingError(
                "biases must have one value per specialist."
            )
        if not (
            self.raw_scores.dtype
            == self.selection_scores.dtype
            == self.biases.dtype
        ):
            raise LossFreeBiasBalancingError(
                "Raw scores, selection scores, and biases must share one dtype."
            )
        if not (
            self.raw_scores.device
            == self.selection_scores.device
            == self.biases.device
        ):
            raise LossFreeBiasBalancingError(
                "Raw scores, selection scores, and biases must share one device."
            )
        if not torch.equal(
            self.selection_scores,
            self.raw_scores + self.biases,
        ):
            raise LossFreeBiasBalancingError(
                "selection_scores must equal raw_scores plus specialist biases."
            )
        if self.biases.requires_grad:
            raise LossFreeBiasBalancingError(
                "Selection-bias snapshots must not require gradients."
            )


@dataclass(frozen=True)
class LossFreeBiasUpdateReport:
    """Immutable diagnostics for one completed BAL-004 load update.

    Assignment loads may be fractional when callers supply weights. Candidate
    zero assignments do not contribute to any specialist load.
    """

    expert_ids: tuple[str, ...]
    assignment_loads: tuple[float, ...]
    mean_load: float
    load_errors: tuple[float, ...]
    previous_biases: tuple[float, ...]
    updated_biases: tuple[float, ...]
    maximum_load_violation: float
    update_step: int

    def __post_init__(self) -> None:
        """Validate deterministic report widths and finite diagnostics."""

        expert_ids = _normalize_expert_ids(self.expert_ids)
        object.__setattr__(self, "expert_ids", expert_ids)
        width = len(expert_ids)
        for field_name in (
            "assignment_loads",
            "load_errors",
            "previous_biases",
            "updated_biases",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or len(values) != width:
                raise LossFreeBiasBalancingError(
                    f"{field_name} must contain one value per specialist."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in values
            ):
                raise LossFreeBiasBalancingError(
                    f"{field_name} must contain only finite real values."
                )
        if any(load < 0 for load in self.assignment_loads):
            raise LossFreeBiasBalancingError(
                "assignment_loads must be non-negative."
            )
        for field_name in ("mean_load", "maximum_load_violation"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise LossFreeBiasBalancingError(
                    f"{field_name} must be a finite non-negative real number."
                )
        if (
            isinstance(self.update_step, bool)
            or not isinstance(self.update_step, int)
            or self.update_step <= 0
        ):
            raise LossFreeBiasBalancingError(
                "update_step must be a positive integer."
            )


class LossFreeBiasBalancer(nn.Module):
    r"""Maintain gradient-free specialist biases for top-k selection.

    For raw calibrated specialist scores \(s_{t,i}\) and persistent biases
    \(b_i\), selection uses

    \[
    \tilde{s}_{t,i}=s_{t,i}+b_i.
    \]

    Output weighting continues to use \(s_{t,i}\), never
    \(\tilde{s}_{t,i}\). Given completed-batch specialist loads \(c_i\), their
    mean \(\bar c\), and update rate \(u\), the next batch uses

    \[
    b_i \leftarrow b_i + u\,\operatorname{sign}(\bar c-c_i).
    \]

    The module owns buffers only, has no parameters, and introduces no
    auxiliary-loss gradient. Candidate zero denotes the immutable anchor or an
    inactive slot and never receives a bias.

    References:
        Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for
        Mixture-of-Experts." https://arxiv.org/abs/2408.15664

    Differences from the paper:
        Assignment indices and optional padding weights are supplied
        explicitly so the primitive composes with anchored top-1, top-2, and
        dynamic-k policies. Distributed aggregation and automatic optimizer
        hooks are not performed.
    """

    strategy_name: Final = "loss_free_selection_bias"

    def __init__(
        self,
        expert_ids: Sequence[str],
        *,
        config: LossFreeBiasConfig | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        """Create zero biases in stable specialist order."""

        super().__init__()
        self.expert_ids = _normalize_expert_ids(expert_ids)
        self.config = config or LossFreeBiasConfig()
        if not isinstance(self.config, LossFreeBiasConfig):
            raise TypeError("config must be LossFreeBiasConfig or None.")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype.")
        probe = torch.empty((), dtype=dtype)
        if not probe.is_floating_point():
            raise LossFreeBiasBalancingError(
                "Loss-free selection biases require a floating-point dtype."
            )
        self.register_buffer(
            "biases",
            torch.zeros(len(self.expert_ids), dtype=dtype, device=device),
        )
        self.register_buffer(
            "_update_step",
            torch.zeros((), dtype=torch.int64, device=device),
        )

    @property
    def update_step(self) -> int:
        """Return the number of completed load observations."""

        return int(self._update_step.item())

    def apply_for_selection(self, expert_scores: Tensor) -> LossFreeBiasedScores:
        """Return raw scores and a separately biased selection view.

        Args:
            expert_scores:
                Finite floating tensor with shape ``token_shape + (K,)`` in
                this balancer's stable ``expert_ids`` order.

        The input tensor is retained as ``raw_scores`` without mutation.
        Gradients can still flow to it through ``selection_scores``, but the
        persistent biases are buffers and never receive gradients.
        """

        _validate_scores(
            expert_scores,
            expert_count=len(self.expert_ids),
            biases=self.biases,
        )
        return LossFreeBiasedScores(
            raw_scores=expert_scores,
            selection_scores=expert_scores + self.biases,
            biases=self.biases.detach().clone(),
            expert_ids=self.expert_ids,
        )

    @torch.no_grad()
    def update(
        self,
        assignment_indices: Tensor,
        *,
        assignment_weights: Tensor | None = None,
    ) -> LossFreeBiasUpdateReport:
        """Update biases from completed-batch active assignment occurrences.

        Args:
            assignment_indices:
                Long tensor of any non-empty shape. Zero represents the anchor
                or an inactive route slot; values ``1..K`` refer to
                ``expert_ids``. For top-2 or dynamic-k routing, callers should
                pass one occurrence per active specialist and zero padding for
                inactive slots.
            assignment_weights:
                Optional finite non-negative tensor with exactly the assignment
                shape. Boolean masks are accepted.

        The update must occur after the assignments have already been used for
        the completed batch. This historical ordering prevents current-batch
        or future-token load information from influencing earlier routes.
        """

        if not torch.isfinite(self.biases).all().item():
            raise LossFreeBiasBalancingError(
                "Persistent selection biases must contain only finite values."
            )
        indices, weights = _validate_assignments(
            assignment_indices,
            assignment_weights=assignment_weights,
            expert_count=len(self.expert_ids),
            device=self.biases.device,
            dtype=self.biases.dtype,
        )
        occurrences = F.one_hot(
            indices,
            num_classes=len(self.expert_ids) + 1,
        )[..., 1:].to(dtype=self.biases.dtype)
        loads = (
            occurrences * weights.unsqueeze(-1)
        ).reshape(-1, len(self.expert_ids)).sum(dim=0)
        mean_load = loads.mean()
        errors = mean_load - loads
        previous_biases = self.biases.detach().clone()
        self.biases.add_(
            self.config.update_rate * torch.sign(errors)
        )
        if not torch.isfinite(self.biases).all().item():
            self.biases.copy_(previous_biases)
            raise LossFreeBiasBalancingError(
                "Loss-free bias update produced non-finite state."
            )
        self._update_step.add_(1)

        if mean_load.item() == 0:
            maximum_violation = 0.0
        else:
            maximum_violation = float(
                ((loads.max() - mean_load) / mean_load).item()
            )
        return LossFreeBiasUpdateReport(
            expert_ids=self.expert_ids,
            assignment_loads=_tensor_to_float_tuple(loads),
            mean_load=float(mean_load.item()),
            load_errors=_tensor_to_float_tuple(errors),
            previous_biases=_tensor_to_float_tuple(previous_biases),
            updated_biases=_tensor_to_float_tuple(self.biases),
            maximum_load_violation=maximum_violation,
            update_step=self.update_step,
        )

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save buffer-only BAL-004 state and strict metadata."""

        destination = _validate_save_path(path, overwrite=overwrite)
        metadata_payload = {
            "config": self.config.to_dict(),
            "expert_ids": list(self.expert_ids),
            "format": _SERIALIZATION_FORMAT,
            "format_version": LOSS_FREE_BIAS_FORMAT_VERSION,
            "strategy": self.strategy_name,
        }
        metadata = {
            _METADATA_KEY: json.dumps(
                metadata_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
        state = {
            "biases": self.biases.detach().cpu().contiguous(),
            "update_step": self._update_step.detach().cpu().contiguous(),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=_SAFETENSORS_SUFFIX,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            save_file(state, temporary_path, metadata=metadata)
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError, ValueError) as error:
            raise LossFreeBiasBalancingError(
                f"Cannot save loss-free bias state to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> LossFreeBiasBalancer:
        """Strictly load buffer-only BAL-004 state from safetensors."""

        source = _validate_load_path(path)
        try:
            with safe_open(source, framework="pt", device="cpu") as payload:
                serialized_metadata = (payload.metadata() or {}).get(
                    _METADATA_KEY
                )
            metadata = json.loads(serialized_metadata)
        except (OSError, RuntimeError, TypeError, json.JSONDecodeError) as error:
            raise LossFreeBiasBalancingError(
                f"Cannot read loss-free bias metadata from {source}."
            ) from error
        if not isinstance(metadata, dict):
            raise LossFreeBiasBalancingError(
                f"Invalid loss-free bias metadata in {source}."
            )
        expected_fields = {
            "config",
            "expert_ids",
            "format",
            "format_version",
            "strategy",
        }
        if set(metadata) != expected_fields:
            raise LossFreeBiasBalancingError(
                f"Incompatible loss-free bias metadata fields in {source}."
            )
        if (
            metadata["format"] != _SERIALIZATION_FORMAT
            or metadata["format_version"] != LOSS_FREE_BIAS_FORMAT_VERSION
            or metadata["strategy"] != cls.strategy_name
        ):
            raise LossFreeBiasBalancingError(
                f"Incompatible loss-free bias format in {source}."
            )
        expert_ids = _normalize_expert_ids(metadata["expert_ids"])
        config = LossFreeBiasConfig.from_dict(metadata["config"])
        try:
            state = load_file(
                source,
                device=str(device) if device is not None else "cpu",
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise LossFreeBiasBalancingError(
                f"Cannot load loss-free bias tensors from {source}: {error}"
            ) from error
        if set(state) != {"biases", "update_step"}:
            raise LossFreeBiasBalancingError(
                f"Incompatible loss-free bias tensor keys in {source}."
            )
        biases = state["biases"]
        update_step = state["update_step"]
        if (
            not biases.is_floating_point()
            or biases.shape != (len(expert_ids),)
            or not torch.isfinite(biases).all().item()
        ):
            raise LossFreeBiasBalancingError(
                f"Invalid specialist bias tensor in {source}."
            )
        if (
            update_step.dtype != torch.int64
            or update_step.shape != ()
            or int(update_step.item()) < 0
        ):
            raise LossFreeBiasBalancingError(
                f"Invalid loss-free bias update step in {source}."
            )
        balancer = cls(
            expert_ids,
            config=config,
            dtype=biases.dtype,
            device=biases.device,
        )
        balancer.biases.copy_(biases)
        balancer._update_step.copy_(update_step)
        return balancer


def _normalize_expert_ids(expert_ids: Sequence[str]) -> tuple[str, ...]:
    """Return two or more unique stable specialist identifiers."""

    if isinstance(expert_ids, (str, bytes)) or not isinstance(
        expert_ids,
        Sequence,
    ):
        raise LossFreeBiasBalancingError(
            "expert_ids must be a sequence of identifiers."
        )
    normalized = tuple(expert_ids)
    if len(normalized) < 2:
        raise LossFreeBiasBalancingError(
            "Loss-free balancing requires at least two specialists."
        )
    if any(
        not isinstance(expert_id, str)
        or not expert_id
        or expert_id != expert_id.strip()
        for expert_id in normalized
    ):
        raise LossFreeBiasBalancingError(
            "expert_ids must be non-empty strings without surrounding whitespace."
        )
    if len(set(normalized)) != len(normalized):
        raise LossFreeBiasBalancingError("expert_ids must be unique.")
    return normalized


def _validate_scores(
    expert_scores: Tensor,
    *,
    expert_count: int,
    biases: Tensor,
) -> None:
    """Validate specialist scores without including an anchor column."""

    if not isinstance(expert_scores, Tensor) or not expert_scores.is_floating_point():
        raise LossFreeBiasBalancingError(
            "expert_scores must be a floating-point torch.Tensor."
        )
    if expert_scores.ndim < 1 or expert_scores.shape[-1] != expert_count:
        raise LossFreeBiasBalancingError(
            "expert_scores must have one final column per specialist."
        )
    if expert_scores.dtype != biases.dtype:
        raise LossFreeBiasBalancingError(
            "expert_scores and persistent biases must share one dtype."
        )
    if expert_scores.device != biases.device:
        raise LossFreeBiasBalancingError(
            "expert_scores and persistent biases must share one device."
        )
    if not torch.isfinite(biases).all().item():
        raise LossFreeBiasBalancingError(
            "Persistent selection biases must contain only finite values."
        )
    if not torch.isfinite(expert_scores).all().item():
        raise LossFreeBiasBalancingError(
            "expert_scores must contain only finite values."
        )


def _validate_assignments(
    assignment_indices: Tensor,
    *,
    assignment_weights: Tensor | None,
    expert_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Return validated assignments and floating occurrence weights."""

    if not isinstance(assignment_indices, Tensor):
        raise LossFreeBiasBalancingError(
            "assignment_indices must be a torch.Tensor."
        )
    if assignment_indices.dtype != torch.long:
        raise LossFreeBiasBalancingError(
            "assignment_indices must use torch.long."
        )
    if assignment_indices.ndim < 1 or assignment_indices.numel() == 0:
        raise LossFreeBiasBalancingError(
            "assignment_indices must have a non-empty shape."
        )
    if assignment_indices.device != device:
        raise LossFreeBiasBalancingError(
            "assignment_indices and persistent biases must share one device."
        )
    minimum = int(assignment_indices.min().item())
    maximum = int(assignment_indices.max().item())
    if minimum < 0 or maximum > expert_count:
        raise LossFreeBiasBalancingError(
            "assignment_indices contains a candidate outside the anchor and "
            f"{expert_count} specialists."
        )

    if assignment_weights is None:
        weights = torch.ones(
            assignment_indices.shape,
            dtype=dtype,
            device=device,
        )
    else:
        if not isinstance(assignment_weights, Tensor):
            raise LossFreeBiasBalancingError(
                "assignment_weights must be a torch.Tensor or None."
            )
        if assignment_weights.shape != assignment_indices.shape:
            raise LossFreeBiasBalancingError(
                "assignment_weights must match the assignment shape."
            )
        if assignment_weights.device != device:
            raise LossFreeBiasBalancingError(
                "assignment_weights and persistent biases must share one device."
            )
        if assignment_weights.is_complex():
            raise LossFreeBiasBalancingError(
                "assignment_weights must not be complex."
            )
        if not torch.isfinite(assignment_weights).all().item():
            raise LossFreeBiasBalancingError(
                "assignment_weights must contain only finite values."
            )
        if torch.any(assignment_weights < 0).item():
            raise LossFreeBiasBalancingError(
                "assignment_weights must be non-negative."
            )
        weights = assignment_weights.to(dtype=dtype)
    return assignment_indices, weights


def _tensor_to_float_tuple(value: Tensor) -> tuple[float, ...]:
    """Detach a one-dimensional diagnostic tensor into immutable floats."""

    return tuple(float(item) for item in value.detach().cpu().tolist())


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    """Validate a non-symlink safetensors destination and existing parent."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise LossFreeBiasBalancingError(
            f"Loss-free bias state must end in {_SAFETENSORS_SUFFIX}."
        )
    if destination.is_symlink():
        raise LossFreeBiasBalancingError(
            "Loss-free bias destination must not be a symbolic link."
        )
    if not destination.parent.is_dir():
        raise LossFreeBiasBalancingError(
            "Loss-free bias destination parent must already exist."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if destination.exists() and not destination.is_file():
        raise LossFreeBiasBalancingError(
            "Loss-free bias destination must be a regular file."
        )
    return destination


def _validate_load_path(path: str | Path) -> Path:
    """Validate an existing non-symlink safetensors source."""

    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise LossFreeBiasBalancingError(
            f"Loss-free bias state must end in {_SAFETENSORS_SUFFIX}."
        )
    if source.is_symlink():
        raise LossFreeBiasBalancingError(
            "Loss-free bias source must not be a symbolic link."
        )
    if not source.is_file():
        raise LossFreeBiasBalancingError(
            "Loss-free bias source must be a regular file."
        )
    return source
