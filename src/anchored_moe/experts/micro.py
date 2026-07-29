"""Fine-grained gated-MLP micro-experts inspired by DeepSeekMoE.

References:
    Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in
    Mixture-of-Experts Language Models."
    https://arxiv.org/abs/2401.06066

Differences from the paper:
    DeepSeekMoE trains router-weighted fine-grained experts jointly from
    scratch. EXP-004 instead partitions one immutable dense gated MLP into an
    atomic set of contiguous intermediate-channel segments. Executing every
    segment with unit contribution reconstructs the pinned anchor at
    initialization. Routing a subset of segments is deliberately left to
    later features.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from anchored_moe.experts.base import Expert, ExpertSerializationError


MICRO_EXPERT_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = (
    f"anchored_moe.fine_grained_micro.v{MICRO_EXPERT_FORMAT_VERSION}"
)
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"
_CONFIG_FIELDS: Final = {
    "segmentation_factor",
    "gate_projection",
    "up_projection",
    "down_projection",
    "activation",
    "partition",
}


@dataclass(frozen=True)
class FineGrainedMicroExpertConfig:
    """Configuration for deterministic intermediate-channel segmentation.

    Args:
        segmentation_factor:
            Number ``m >= 2`` of equal-width micro-experts.
        gate_projection:
            Dotted path to the gated MLP's gate projection.
        up_projection:
            Dotted path to the gated MLP's value/up projection.
        down_projection:
            Dotted path to the gated MLP's output/down projection.
        activation:
            Dotted path to the pointwise activation callable.
        partition:
            Version-1 partition strategy. Only ``"contiguous"`` is supported.
    """

    segmentation_factor: int
    gate_projection: str = "gate_proj"
    up_projection: str = "up_proj"
    down_projection: str = "down_proj"
    activation: str = "act_fn"
    partition: str = "contiguous"

    def __post_init__(self) -> None:
        if (
            isinstance(self.segmentation_factor, bool)
            or not isinstance(self.segmentation_factor, int)
            or self.segmentation_factor < 2
        ):
            raise ValueError("segmentation_factor must be an integer >= 2.")
        paths = {
            "gate_projection": self.gate_projection,
            "up_projection": self.up_projection,
            "down_projection": self.down_projection,
            "activation": self.activation,
        }
        for field_name, path in paths.items():
            _validate_attribute_path(path, field_name=field_name)
        projection_paths = (
            self.gate_projection,
            self.up_projection,
            self.down_projection,
        )
        if len(set(projection_paths)) != len(projection_paths):
            raise ValueError("Projection paths must be distinct.")
        if self.partition != "contiguous":
            raise ValueError("partition must be 'contiguous' in format version 1.")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible configuration."""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, object],
    ) -> FineGrainedMicroExpertConfig:
        """Strictly reconstruct a configuration from serialized metadata."""

        if not isinstance(values, Mapping):
            raise ExpertSerializationError(
                "Fine-grained micro-expert config must be a mapping."
            )
        if set(values) != _CONFIG_FIELDS:
            raise ExpertSerializationError(
                "Fine-grained micro-expert config fields do not match "
                f"the version-1 schema: {sorted(values)}"
            )
        try:
            return cls(
                segmentation_factor=values["segmentation_factor"],  # type: ignore[arg-type]
                gate_projection=values["gate_projection"],  # type: ignore[arg-type]
                up_projection=values["up_projection"],  # type: ignore[arg-type]
                down_projection=values["down_projection"],  # type: ignore[arg-type]
                activation=values["activation"],  # type: ignore[arg-type]
                partition=values["partition"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Invalid fine-grained micro-expert config: {error}"
            ) from error


@dataclass(frozen=True)
class FineGrainedMicroExpertTopology:
    """Resolved immutable shape and source contract for one micro-expert set."""

    source_module_class: str
    activation_class: str
    hidden_size: int
    intermediate_size: int
    segmentation_factor: int
    segment_width: int
    weight_dtype: str

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible topology."""

        return asdict(self)


class FineGrainedMicroExpert(nn.Module):
    r"""One narrow intermediate-channel contribution to a gated MLP.

    For segment ``i`` and hidden states ``h``, the module computes

    \[
    E_i(h)=W_{d,i}\left(\sigma(W_{g,i}h)\odot W_{u,i}h\right).
    \]

    The module is owned and serialized by
    :class:`FineGrainedMicroExpertSet`; it is not independently attachable as
    a complete replacement expert in EXP-004.
    """

    def __init__(
        self,
        *,
        index: int,
        channel_start: int,
        channel_end: int,
        gate_projection: nn.Linear,
        up_projection: nn.Linear,
        down_projection: nn.Linear,
        activation: Any,
    ) -> None:
        super().__init__()
        self.index = index
        self.channel_start = channel_start
        self.channel_end = channel_end
        self.gate_proj = gate_projection
        self.up_proj = up_projection
        self.down_proj = down_projection
        self.activation = activation

    @property
    def width(self) -> int:
        """Return the number of intermediate channels owned by this segment."""

        return self.channel_end - self.channel_start

    def forward_intermediate(self, hidden_states: Tensor) -> Tensor:
        """Return this segment's gated intermediate activation."""

        return self.activation(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return this segment's full-hidden-size partial MLP contribution."""

        return self.down_proj(self.forward_intermediate(hidden_states))


class FineGrainedMicroExpertSet(Expert):
    r"""Atomic fine-grained partition of one compatible bias-free gated MLP.

    The source anchor must implement the structural equation

    \[
    A(h)=W_d\left(\sigma(W_gh)\odot W_uh\right)
    \]

    using three bias-free floating-point ``nn.Linear`` projections and a
    pointwise activation. The intermediate dimension is split into ``m``
    deterministic contiguous ranges. All source parameters are copied into
    disjoint narrow modules before the live anchor is frozen.

    ``forward()`` executes every segment and reconstructs the original
    intermediate and down-projection tensors in channel order. This preserves
    the complete :class:`Expert` replacement contract. Individual children
    remain executable through ``micro_experts`` for future retrieval research,
    but EXP-004 defines no selection or combination policy for subsets.

    References:
        Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in
        Mixture-of-Experts Language Models."
        https://arxiv.org/abs/2401.06066

    Differences from the paper:
        The paper trains router-weighted micro-experts jointly from scratch.
        This anchored representation starts from one dense gated MLP and uses
        all segments with unit contribution for exact initialization. It does
        not implement shared experts, routing, load balancing, or kernels.
    """

    parameterization: ClassVar[str] = "fine_grained_micro"

    def __init__(
        self,
        anchor_mlp: nn.Module,
        *,
        config: FineGrainedMicroExpertConfig,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(config, FineGrainedMicroExpertConfig):
            raise TypeError(
                "config must be a FineGrainedMicroExpertConfig instance."
            )
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")

        inspected = _inspect_compatible_anchor(anchor_mlp, config)
        micro_experts = _copy_micro_experts(
            inspected=inspected,
        )
        # Mutate caller-visible gradient flags only after the complete
        # topology and every independent tensor copy have succeeded.
        anchor_mlp.requires_grad_(False)
        self.config = config
        self.topology = inspected.topology
        self.micro_experts = nn.ModuleList(micro_experts)
        self.requires_grad_(trainable)
        self.train(anchor_mlp.training)

    @classmethod
    def from_anchor(
        cls,
        anchor_mlp: nn.Module,
        *,
        config: FineGrainedMicroExpertConfig,
        trainable: bool = True,
    ) -> FineGrainedMicroExpertSet:
        """Partition a compatible anchor and freeze it after a successful copy."""

        return cls(anchor_mlp, config=config, trainable=trainable)

    @classmethod
    def from_safetensors(
        cls,
        anchor_mlp: nn.Module,
        path: str | Path,
        *,
        trainable: bool = True,
    ) -> FineGrainedMicroExpertSet:
        """Strictly load an atomic micro-expert set without anchor payload data.

        Metadata, source structure, tensor keys, shapes, dtypes, and finite
        values are validated before the live anchor is frozen.
        """

        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        source = _validate_load_path(path)
        try:
            with safe_open(source, framework="pt", device="cpu") as payload:
                file_metadata = payload.metadata() or {}
        except Exception as error:
            raise ExpertSerializationError(
                f"Cannot read safetensors payload {source}: {error}"
            ) from error

        serialized_metadata = file_metadata.get(_METADATA_KEY)
        try:
            metadata = json.loads(serialized_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ExpertSerializationError(
                f"Invalid fine-grained micro-expert metadata in {source}."
            ) from error
        if not isinstance(metadata, dict):
            raise ExpertSerializationError(
                f"Invalid fine-grained micro-expert metadata in {source}."
            )
        if set(metadata) != {
            "config",
            "format",
            "parameterization",
            "topology",
        }:
            raise ExpertSerializationError(
                "Fine-grained micro-expert metadata fields do not match "
                "the version-1 schema."
            )
        if metadata.get("format") != _SERIALIZATION_FORMAT:
            raise ExpertSerializationError(
                f"Unsupported fine-grained micro-expert format in {source}."
            )
        if metadata.get("parameterization") != cls.parameterization:
            raise ExpertSerializationError(
                f"Incompatible expert parameterization in {source}."
            )

        config = FineGrainedMicroExpertConfig.from_dict(metadata["config"])
        inspected = _inspect_compatible_anchor(anchor_mlp, config)
        if metadata.get("topology") != inspected.topology.to_dict():
            raise ExpertSerializationError(
                f"Incompatible fine-grained micro-expert topology in {source}."
            )
        try:
            saved_state = load_file(source, device="cpu")
        except (OSError, RuntimeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot load fine-grained micro-expert tensors from {source}: "
                f"{error}"
            ) from error
        _validate_saved_state(
            saved_state,
            topology=inspected.topology,
        )

        try:
            expert = cls(anchor_mlp, config=config, trainable=trainable)
            expert.load_state_dict(saved_state, strict=True)
        except ExpertSerializationError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot restore fine-grained micro-expert tensors from "
                f"{source}: {error}"
            ) from error
        expert.requires_grad_(trainable)
        return expert

    @property
    def expert_count(self) -> int:
        """Return the number of atomic micro-experts."""

        return len(self.micro_experts)

    @property
    def segment_width(self) -> int:
        """Return the common intermediate width of each micro-expert."""

        return self.topology.segment_width

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute every segment as one complete replacement MLP.

        Concatenating the gated intermediates and down-projection columns
        restores the anchor's original channel order. One full-width
        ``linear`` reduction avoids introducing a different floating-point
        summation tree solely because the representation is segmented.
        """

        if args or kwargs:
            raise TypeError(
                "FineGrainedMicroExpertSet version 1 accepts only hidden_states."
            )
        if hidden_states.shape[-1] != self.topology.hidden_size:
            raise ValueError(
                "hidden_states last dimension must equal the topology "
                f"hidden_size {self.topology.hidden_size}."
            )
        gate_weight = torch.cat(
            tuple(expert.gate_proj.weight for expert in self.micro_experts),
            dim=0,
        )
        up_weight = torch.cat(
            tuple(expert.up_proj.weight for expert in self.micro_experts),
            dim=0,
        )
        # Re-forming the two full-width input projections as well as the down
        # projection keeps the initialization path numerically identical to
        # the dense structural equation. Executing smaller GEMMs separately
        # can select different kernels and introduce rounding before concat.
        intermediate = self.micro_experts[0].activation(
            F.linear(hidden_states, gate_weight)
        ) * F.linear(
            hidden_states,
            up_weight,
        )
        down_weight = torch.cat(
            tuple(expert.down_proj.weight for expert in self.micro_experts),
            dim=1,
        )
        return F.linear(intermediate, down_weight)

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically serialize only segmented expert tensors and metadata."""

        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state: dict[str, Tensor] = {}
        for name, value in sorted(self.state_dict().items()):
            if not isinstance(value, Tensor):
                raise ExpertSerializationError(
                    f"Fine-grained micro-expert state {name!r} is not a tensor."
                )
            tensor_state[name] = value.detach().cpu().contiguous()
        metadata_values = {
            "config": self.config.to_dict(),
            "format": _SERIALIZATION_FORMAT,
            "parameterization": self.parameterization,
            "topology": self.topology.to_dict(),
        }
        metadata = {
            _METADATA_KEY: json.dumps(
                metadata_values,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
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
            save_file(tensor_state, temporary_path, metadata=metadata)
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot save fine-grained micro-expert tensors to "
                f"{destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _InspectedAnchor:
    """Validated source components used only during failure-atomic copying."""

    gate_projection: nn.Linear
    up_projection: nn.Linear
    down_projection: nn.Linear
    activation: Any
    topology: FineGrainedMicroExpertTopology


def _inspect_compatible_anchor(
    anchor_mlp: nn.Module,
    config: FineGrainedMicroExpertConfig,
) -> _InspectedAnchor:
    """Validate the narrow version-1 gated-MLP structural contract."""

    projections: dict[str, nn.Linear] = {}
    for field_name, path in (
        ("gate_projection", config.gate_projection),
        ("up_projection", config.up_projection),
        ("down_projection", config.down_projection),
    ):
        candidate = _resolve_attribute(anchor_mlp, path)
        if not isinstance(candidate, nn.Linear):
            raise ValueError(f"{field_name} path {path!r} must resolve to nn.Linear.")
        if candidate.bias is not None:
            raise ValueError(
                f"{field_name} path {path!r} must be bias-free in format version 1."
            )
        if not candidate.weight.is_floating_point():
            raise ValueError(
                f"{field_name} path {path!r} must use floating-point weights."
            )
        projections[field_name] = candidate
    if len({id(module) for module in projections.values()}) != 3:
        raise ValueError("Projection paths must resolve to distinct modules.")

    gate = projections["gate_projection"]
    up = projections["up_projection"]
    down = projections["down_projection"]
    if gate.weight.shape != up.weight.shape:
        raise ValueError("Gate and up projection weight shapes must match.")
    intermediate_size, hidden_size = gate.weight.shape
    if tuple(down.weight.shape) != (hidden_size, intermediate_size):
        raise ValueError(
            "Down projection shape must be "
            "[hidden_size, intermediate_size]."
        )
    devices = {gate.weight.device, up.weight.device, down.weight.device}
    dtypes = {gate.weight.dtype, up.weight.dtype, down.weight.dtype}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("All gated-MLP projections must share device and dtype.")
    if intermediate_size % config.segmentation_factor != 0:
        raise ValueError(
            "intermediate_size must be exactly divisible by "
            "segmentation_factor."
        )

    activation = _resolve_attribute(anchor_mlp, config.activation)
    if not callable(activation):
        raise ValueError(
            f"activation path {config.activation!r} must resolve to a callable."
        )
    if isinstance(activation, nn.Module):
        if any(True for _ in activation.parameters(recurse=True)):
            raise ValueError("The activation module must not own parameters.")
        if any(True for _ in activation.buffers(recurse=True)):
            raise ValueError("The activation module must not own buffers.")

    allowed_parameter_names = {
        f"{config.gate_projection}.weight",
        f"{config.up_projection}.weight",
        f"{config.down_projection}.weight",
    }
    actual_parameter_names = {
        name for name, _ in anchor_mlp.named_parameters(recurse=True)
    }
    if actual_parameter_names != allowed_parameter_names:
        raise ValueError(
            "Compatible gated MLPs must own exactly the three configured "
            "projection weights."
        )
    if any(True for _ in anchor_mlp.named_buffers(recurse=True)):
        raise ValueError(
            "Compatible gated MLPs must not own persistent buffers in "
            "format version 1."
        )

    topology = FineGrainedMicroExpertTopology(
        source_module_class=_qualified_name(anchor_mlp),
        activation_class=_qualified_name(activation),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        segmentation_factor=config.segmentation_factor,
        segment_width=intermediate_size // config.segmentation_factor,
        weight_dtype=str(gate.weight.dtype).removeprefix("torch."),
    )
    return _InspectedAnchor(
        gate_projection=gate,
        up_projection=up,
        down_projection=down,
        activation=activation,
        topology=topology,
    )


def _copy_micro_experts(
    *,
    inspected: _InspectedAnchor,
) -> list[FineGrainedMicroExpert]:
    """Copy all contiguous segments before any live-anchor mutation."""

    topology = inspected.topology
    experts: list[FineGrainedMicroExpert] = []
    try:
        for index in range(topology.segmentation_factor):
            start = index * topology.segment_width
            end = start + topology.segment_width
            experts.append(
                FineGrainedMicroExpert(
                    index=index,
                    channel_start=start,
                    channel_end=end,
                    gate_projection=_copy_linear_rows(
                        inspected.gate_projection,
                        start=start,
                        end=end,
                    ),
                    up_projection=_copy_linear_rows(
                        inspected.up_projection,
                        start=start,
                        end=end,
                    ),
                    down_projection=_copy_linear_columns(
                        inspected.down_projection,
                        start=start,
                        end=end,
                    ),
                    activation=deepcopy(inspected.activation),
                )
            )
    except Exception as error:
        raise RuntimeError(
            "Cannot copy the complete fine-grained micro-expert set."
        ) from error
    return experts


def _copy_linear_rows(source: nn.Linear, *, start: int, end: int) -> nn.Linear:
    """Copy a contiguous output-row slice into an independent linear module."""

    copied = nn.Linear(
        source.in_features,
        end - start,
        bias=False,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        copied.weight.copy_(source.weight[start:end])
    return copied


def _copy_linear_columns(
    source: nn.Linear,
    *,
    start: int,
    end: int,
) -> nn.Linear:
    """Copy a contiguous input-column slice into an independent linear module."""

    copied = nn.Linear(
        end - start,
        source.out_features,
        bias=False,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        copied.weight.copy_(source.weight[:, start:end])
    return copied


def _validate_saved_state(
    saved_state: Mapping[str, Tensor],
    *,
    topology: FineGrainedMicroExpertTopology,
) -> None:
    """Require exact tensor closure, shapes, dtypes, and finite values."""

    expected_dtype = getattr(torch, topology.weight_dtype, None)
    if not isinstance(expected_dtype, torch.dtype):
        raise ExpertSerializationError(
            f"Unsupported serialized dtype {topology.weight_dtype!r}."
        )
    expected: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for index in range(topology.segmentation_factor):
        prefix = f"micro_experts.{index}"
        expected[f"{prefix}.gate_proj.weight"] = (
            (topology.segment_width, topology.hidden_size),
            expected_dtype,
        )
        expected[f"{prefix}.up_proj.weight"] = (
            (topology.segment_width, topology.hidden_size),
            expected_dtype,
        )
        expected[f"{prefix}.down_proj.weight"] = (
            (topology.hidden_size, topology.segment_width),
            expected_dtype,
        )
    if set(saved_state) != set(expected):
        raise ExpertSerializationError(
            "Fine-grained micro-expert tensor keys do not match the "
            "version-1 schema."
        )
    for name, (shape, dtype) in expected.items():
        tensor = saved_state[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ExpertSerializationError(
                f"Fine-grained micro-expert tensor schema mismatch for {name!r}."
            )
        if not torch.isfinite(tensor).all().item():
            raise ExpertSerializationError(
                f"Fine-grained micro-expert tensor {name!r} is not finite."
            )


def _resolve_attribute(root: object, path: str) -> object:
    """Resolve a validated dotted attribute path without executing code."""

    current = root
    for component in path.split("."):
        try:
            current = getattr(current, component)
        except AttributeError as error:
            raise ValueError(f"Attribute path {path!r} does not exist.") from error
    return current


def _validate_attribute_path(path: object, *, field_name: str) -> None:
    """Require a stable public dotted attribute path."""

    if not isinstance(path, str) or not path or path.strip() != path:
        raise ValueError(f"{field_name} must be a non-empty attribute path.")
    components = path.split(".")
    if any(
        not component.isidentifier() or component.startswith("_")
        for component in components
    ):
        raise ValueError(
            f"{field_name} must contain only public Python identifiers."
        )


def _qualified_name(value: object) -> str:
    """Return a stable fully qualified class or callable name."""

    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    value_type = type(value)
    if value_type.__module__ != "builtins":
        return f"{value_type.__module__}.{value_type.__qualname__}"
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if isinstance(module, str) and isinstance(qualname, str):
        return f"{module}.{qualname}"
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _validate_load_path(path: str | Path) -> Path:
    """Resolve the non-symlink safetensors path accepted by the loader."""

    source = Path(path)
    if source.suffix != _SAFETENSORS_SUFFIX:
        raise ExpertSerializationError(
            f"Expert payload must end in {_SAFETENSORS_SUFFIX}: {source}"
        )
    if source.is_symlink():
        raise ExpertSerializationError(
            f"Expert payload must not be a symbolic link: {source}"
        )
    if not source.is_file():
        raise ExpertSerializationError(
            f"Expert payload is not a regular file: {source}"
        )
    return source


def _validate_save_path(path: str | Path, *, overwrite: bool) -> Path:
    """Validate a destination without creating directories implicitly."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path)
    if destination.suffix != _SAFETENSORS_SUFFIX:
        raise ExpertSerializationError(
            f"Expert payload must end in {_SAFETENSORS_SUFFIX}: {destination}"
        )
    if destination.is_symlink():
        raise ExpertSerializationError(
            f"Expert payload must not be a symbolic link: {destination}"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Expert payload already exists: {destination}")
    if destination.exists() and not destination.is_file():
        raise ExpertSerializationError(
            f"Expert payload destination is not a regular file: {destination}"
        )
    if not destination.parent.is_dir():
        raise ExpertSerializationError(
            f"Expert payload parent directory does not exist: "
            f"{destination.parent}"
        )
    return destination
