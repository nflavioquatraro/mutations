"""Frozen learned mixture-weight heads for already selected expert pairs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from anchored_moe.policies.combination import (
    ActivationCombinationResult,
    CombinationPolicy,
    _combine_with_weights,
    _validate_combination_inputs,
)


LEARNED_MIXTURE_HEAD_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = "anchored_moe.learned_mixture_weight_head.v1"
_METADATA_KEY: Final = "anchored_moe"
_SAFETENSORS_SUFFIX: Final = ".safetensors"


class LearnedMixtureSerializationError(ValueError):
    """Raised when a learned mixture head payload is unsafe or incompatible."""


class LearnedMixtureWeightHead(nn.Module):
    """Produce mixing weights independently from the pair-selection decision.

    For an already selected and ordered specialist pair with calibrated
    selection scores ``[g_1, g_2]``, COM-003 computes one learned logit

    ``z = w_1 g_1 + w_2 g_2 + b``

    and returns ``softmax([z, 0])``. The first weight applies to the
    first-ranked specialist and the second weight applies to the second-ranked
    specialist. Selection remains external: this head cannot approve a pair,
    change its order, interpret anchor candidates, or execute experts.

    The coefficients are immutable buffers rather than parameters. Fitting is
    performed by :func:`anchored_moe.calibration.fit_learned_mixture_head`,
    after which deployment is frozen and storage-independent from fitting
    state.

    Args:
        weight:
            Finite floating-point tensor with shape ``(2,)``.
        bias:
            Finite floating-point scalar tensor.
    """

    architecture: ClassVar[str] = "learned_pair_weight_affine"
    score_feature_size: ClassVar[int] = 2

    def __init__(self, *, weight: Tensor, bias: Tensor) -> None:
        super().__init__()
        normalized_weight, normalized_bias = _normalize_coefficients(
            weight,
            bias=bias,
        )
        self.register_buffer("weight", normalized_weight)
        self.register_buffer("bias", normalized_bias)

    def mixing_logits(self, selected_scores: Tensor) -> Tensor:
        """Return learned pair logits with shape ``selected_scores.shape``.

        The second logit is fixed at zero, making the two-class model
        identifiable while retaining a conventional softmax deployment path.
        """

        _validate_selected_scores(
            selected_scores,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        first_logit = torch.sum(
            selected_scores * self.weight,
            dim=-1,
        ) + self.bias
        return torch.stack((first_logit, torch.zeros_like(first_logit)), dim=-1)

    def forward(self, selected_scores: Tensor) -> Tensor:
        """Return finite normalized pair weights for ordered selected scores."""

        return torch.softmax(self.mixing_logits(selected_scores), dim=-1)

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save frozen coefficients without executable content."""

        destination = _validate_save_path(path, overwrite=overwrite)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in sorted(self.state_dict().items())
        }
        if not all(
            value.is_floating_point() and torch.isfinite(value).all().item()
            for value in state.values()
        ):
            raise LearnedMixtureSerializationError(
                "Learned mixture head state must be finite and floating-point."
            )
        metadata_value = {
            "architecture": self.architecture,
            "format": _SERIALIZATION_FORMAT,
            "format_version": LEARNED_MIXTURE_HEAD_FORMAT_VERSION,
            "score_feature_size": self.score_feature_size,
        }
        metadata = {
            _METADATA_KEY: json.dumps(
                metadata_value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
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
            raise LearnedMixtureSerializationError(
                f"Cannot save learned mixture head to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
    ) -> LearnedMixtureWeightHead:
        """Strictly load a frozen learned mixture head on CPU."""

        source = _validate_load_path(path)
        metadata = _load_metadata(source)
        expected_metadata = {
            "architecture": cls.architecture,
            "format": _SERIALIZATION_FORMAT,
            "format_version": LEARNED_MIXTURE_HEAD_FORMAT_VERSION,
            "score_feature_size": cls.score_feature_size,
        }
        if metadata != expected_metadata:
            raise LearnedMixtureSerializationError(
                "Learned mixture head metadata is incompatible."
            )
        try:
            state = load_file(source, device="cpu")
        except (OSError, RuntimeError, ValueError) as error:
            raise LearnedMixtureSerializationError(
                f"Cannot load learned mixture head from {source}: {error}"
            ) from error
        if set(state) != {"bias", "weight"}:
            raise LearnedMixtureSerializationError(
                "Learned mixture head tensor keys are incompatible."
            )
        try:
            return cls(weight=state["weight"], bias=state["bias"]).eval()
        except (TypeError, ValueError) as error:
            raise LearnedMixtureSerializationError(
                f"Invalid learned mixture head state in {source}: {error}"
            ) from error


class LearnedMixtureWeightCombination(CombinationPolicy):
    """Combine selected outputs with weights from one frozen COM-003 head."""

    policy_name = "learned_mixture_weight"

    def __init__(self, head: LearnedMixtureWeightHead) -> None:
        """Bind an independently fitted, buffer-only mixture head."""

        if not isinstance(head, LearnedMixtureWeightHead):
            raise TypeError("head must be a LearnedMixtureWeightHead.")
        if tuple(head.parameters()):
            raise ValueError("Learned mixture deployment head must be frozen.")
        self.head = head

    def combine(
        self,
        selected_outputs: Tensor,
        *,
        selected_scores: Tensor,
    ) -> ActivationCombinationResult:
        """Return learned-weight activation combination and exact weights."""

        _validate_combination_inputs(
            selected_outputs,
            selected_scores=selected_scores,
        )
        weights = self.head(selected_scores)
        return _combine_with_weights(
            selected_outputs,
            selected_scores=selected_scores,
            weights=weights,
        )


def _normalize_coefficients(
    weight: Tensor,
    *,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate, detach, and copy learned coefficients."""

    if not isinstance(weight, Tensor) or not weight.is_floating_point():
        raise TypeError("weight must be a floating-point torch.Tensor.")
    if weight.shape != (2,):
        raise ValueError("weight must have shape (2,).")
    if not isinstance(bias, Tensor) or not bias.is_floating_point():
        raise TypeError("bias must be a floating-point torch.Tensor.")
    if bias.shape != ():
        raise ValueError("bias must be a scalar tensor.")
    if weight.dtype != bias.dtype:
        raise ValueError("weight and bias must share one dtype.")
    if weight.device != bias.device:
        raise ValueError("weight and bias must share one device.")
    if not torch.isfinite(weight).all().item() or not torch.isfinite(bias).item():
        raise ValueError("weight and bias must contain only finite values.")
    return (
        weight.detach().clone().contiguous(),
        bias.detach().clone().contiguous(),
    )


def _validate_selected_scores(
    selected_scores: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate the ordered score-pair deployment contract."""

    if not isinstance(selected_scores, Tensor):
        raise TypeError("selected_scores must be a torch.Tensor.")
    if not selected_scores.is_floating_point():
        raise TypeError("selected_scores must use a floating-point dtype.")
    if selected_scores.ndim < 1 or selected_scores.shape[-1] != 2:
        raise ValueError(
            "selected_scores must have a final dimension of size two."
        )
    if selected_scores.dtype != dtype:
        raise ValueError(
            "selected_scores dtype does not match the learned mixture head."
        )
    if selected_scores.device != device:
        raise ValueError(
            "selected_scores device does not match the learned mixture head."
        )
    if not torch.isfinite(selected_scores).all().item():
        raise ValueError("selected_scores must contain only finite values.")


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    """Return a safe immutable destination for one head artifact."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise LearnedMixtureSerializationError(
            f"Learned mixture payload must end in {_SAFETENSORS_SUFFIX}."
        )
    if destination.is_symlink():
        raise LearnedMixtureSerializationError(
            "Learned mixture payload must not be a symbolic link."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Learned mixture payload already exists: {destination}"
        )
    if destination.exists() and not destination.is_file():
        raise LearnedMixtureSerializationError(
            "Learned mixture payload destination is not a regular file."
        )
    if not destination.parent.is_dir():
        raise LearnedMixtureSerializationError(
            "Learned mixture payload parent directory does not exist."
        )
    return destination


def _validate_load_path(path: str | Path) -> Path:
    """Return a safe existing learned-head payload path."""

    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise LearnedMixtureSerializationError(
            f"Learned mixture payload must end in {_SAFETENSORS_SUFFIX}."
        )
    if source.is_symlink():
        raise LearnedMixtureSerializationError(
            "Learned mixture payload must not be a symbolic link."
        )
    if not source.is_file():
        raise LearnedMixtureSerializationError(
            f"Learned mixture payload is not a regular file: {source}"
        )
    return source


def _load_metadata(source: Path) -> dict[str, Any]:
    """Load the strict JSON metadata envelope from safetensors."""

    try:
        with safe_open(source, framework="pt", device="cpu") as payload:
            serialized = (payload.metadata() or {}).get(_METADATA_KEY)
        metadata = json.loads(serialized)
    except Exception as error:
        raise LearnedMixtureSerializationError(
            f"Invalid learned mixture head metadata in {source}."
        ) from error
    if not isinstance(metadata, dict):
        raise LearnedMixtureSerializationError(
            f"Invalid learned mixture head metadata in {source}."
        )
    return metadata
