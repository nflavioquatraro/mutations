"""Fixed block-sparse additive experts over private frozen anchor copies.

EXP-003 targets explicitly named linear projections inside one anchor MLP.
Each target receives a deterministic set of non-overlapping trainable blocks;
all other projection entries and the complete private base remain frozen.
The implementation executes compact block multiplications directly and never
materializes a dense delta weight, but it does not claim a fused sparse kernel.

The block mask is project-specific framework infrastructure rather than an
implementation derived from a research paper. SHA-256 is used only to turn a
public seed and canonical module coordinates into a stable mask.

References:
    NIST, "FIPS PUB 180-4: Secure Hash Standard (SHS)."
    https://doi.org/10.6028/NIST.FIPS.180-4

Differences from the reference:
    SHA-256 provides deterministic coordinate ranking, not a security or
    cryptographic authenticity guarantee for expert payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional as F

from anchored_moe.experts.base import (
    AnchorRelativeExpert,
    Expert,
    ExpertSerializationError,
)


BLOCK_SPARSE_DELTA_FORMAT_VERSION: Final = 1
_SERIALIZATION_FORMAT: Final = "anchored_moe.block_sparse_delta.v1"
_METADATA_KEY: Final = "anchored_moe"
_SAFETENSORS_SUFFIX: Final = ".safetensors"


@dataclass(frozen=True)
class BlockSparseDeltaConfig:
    """Configure one deterministic fixed block-sparse expert.

    Args:
        block_shape:
            Positive ``(output_rows, input_columns)`` stored for every active
            block. Target projection dimensions must be exactly divisible by
            these values; ragged boundary blocks are deliberately unsupported
            in format version 1.
        blocks_per_module:
            Positive number of active blocks selected independently in each
            targeted projection.
        target_modules:
            Unique module-name suffixes. Every suffix must resolve to exactly
            one ``torch.nn.Linear`` inside the supplied anchor MLP.
        seed:
            Non-negative integer used with the canonical module path and block
            coordinate to rank candidate blocks deterministically.

    The mask is static after construction. Learning masks, pruning dense
    updates, unstructured sparsity, and model-family-specific kernels remain
    outside this configuration.
    """

    block_shape: tuple[int, int] = (8, 8)
    blocks_per_module: int = 1
    target_modules: tuple[str, ...] = (
        "down_proj",
        "gate_proj",
        "up_proj",
    )
    seed: int = 0

    def __post_init__(self) -> None:
        """Normalize target ordering and reject ambiguous sparse topology."""

        if (
            not isinstance(self.block_shape, tuple)
            or len(self.block_shape) != 2
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                for size in self.block_shape
            )
        ):
            raise ValueError(
                "block_shape must contain two positive integer dimensions."
            )
        if (
            isinstance(self.blocks_per_module, bool)
            or not isinstance(self.blocks_per_module, int)
            or self.blocks_per_module <= 0
        ):
            raise ValueError("blocks_per_module must be a positive integer.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be a non-negative 63-bit integer.")
        if not isinstance(self.target_modules, tuple) or not self.target_modules:
            raise TypeError("target_modules must be a non-empty tuple.")

        normalized_targets: list[str] = []
        for module_name in self.target_modules:
            if (
                not isinstance(module_name, str)
                or not module_name
                or module_name != module_name.strip()
            ):
                raise ValueError(
                    "target_modules must contain non-empty names without "
                    "surrounding whitespace."
                )
            normalized_targets.append(module_name)
        if len(set(normalized_targets)) != len(normalized_targets):
            raise ValueError("target_modules must not contain duplicates.")
        object.__setattr__(
            self,
            "target_modules",
            tuple(sorted(normalized_targets)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible public configuration."""

        return {
            "block_shape": list(self.block_shape),
            "blocks_per_module": self.blocks_per_module,
            "seed": self.seed,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BlockSparseDeltaConfig:
        """Strictly reconstruct one serialized EXP-003 configuration."""

        if not isinstance(value, dict):
            raise TypeError("Block-sparse expert config must be a dictionary.")
        expected = {
            "block_shape",
            "blocks_per_module",
            "seed",
            "target_modules",
        }
        if set(value) != expected:
            raise ValueError(
                "Block-sparse expert config fields are incompatible."
            )
        block_shape = value["block_shape"]
        target_modules = value["target_modules"]
        if not isinstance(block_shape, list):
            raise TypeError("block_shape must be serialized as a list.")
        if not isinstance(target_modules, list):
            raise TypeError("target_modules must be serialized as a list.")
        return cls(
            block_shape=tuple(block_shape),
            blocks_per_module=value["blocks_per_module"],
            seed=value["seed"],
            target_modules=tuple(target_modules),
        )


@dataclass(frozen=True)
class BlockSparseModuleTopology:
    """Resolved immutable block mask for one targeted linear projection.

    Attributes:
        module_path:
            Canonical dotted path inside the private MLP.
        in_features:
            Dense input width of the compatible anchor projection.
        out_features:
            Dense output width of the compatible anchor projection.
        block_shape:
            Fixed ``(output_rows, input_columns)`` of every active block.
        coordinates:
            Canonically sorted ``(output_block, input_block)`` coordinates.
    """

    module_path: str
    in_features: int
    out_features: int
    block_shape: tuple[int, int]
    coordinates: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Validate dimensions, divisibility, uniqueness, and coordinate bounds."""

        if (
            not isinstance(self.module_path, str)
            or not self.module_path
            or self.module_path != self.module_path.strip()
        ):
            raise ValueError("module_path must be a non-empty canonical path.")
        for field_name in ("in_features", "out_features"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            not isinstance(self.block_shape, tuple)
            or len(self.block_shape) != 2
            or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                for size in self.block_shape
            )
        ):
            raise ValueError("block_shape must contain two positive integers.")
        block_out, block_in = self.block_shape
        if self.out_features % block_out or self.in_features % block_in:
            raise ValueError(
                "Projection dimensions must be divisible by block_shape."
            )
        if not isinstance(self.coordinates, tuple) or not self.coordinates:
            raise ValueError("coordinates must be a non-empty tuple.")
        if tuple(sorted(self.coordinates)) != self.coordinates:
            raise ValueError("coordinates must use canonical sorted order.")
        if len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("coordinates must not contain duplicates.")
        output_blocks = self.out_features // block_out
        input_blocks = self.in_features // block_in
        for coordinate in self.coordinates:
            if (
                not isinstance(coordinate, tuple)
                or len(coordinate) != 2
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    for index in coordinate
                )
            ):
                raise ValueError(
                    "coordinates must contain integer block-index pairs."
                )
            output_index, input_index = coordinate
            if not (
                0 <= output_index < output_blocks
                and 0 <= input_index < input_blocks
            ):
                raise ValueError("Block coordinate is outside the projection.")

    @property
    def active_parameter_count(self) -> int:
        """Return the exact trainable scalar count for this projection."""

        block_out, block_in = self.block_shape
        return len(self.coordinates) * block_out * block_in

    @property
    def dense_parameter_count(self) -> int:
        """Return the corresponding dense projection scalar count."""

        return self.out_features * self.in_features

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON-compatible topology metadata."""

        return {
            "block_shape": list(self.block_shape),
            "coordinates": [list(pair) for pair in self.coordinates],
            "in_features": self.in_features,
            "module_path": self.module_path,
            "out_features": self.out_features,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BlockSparseModuleTopology:
        """Strictly reconstruct serialized resolved topology metadata."""

        if not isinstance(value, dict):
            raise TypeError("Block-sparse topology must be a dictionary.")
        expected = {
            "block_shape",
            "coordinates",
            "in_features",
            "module_path",
            "out_features",
        }
        if set(value) != expected:
            raise ValueError("Block-sparse topology fields are incompatible.")
        block_shape = value["block_shape"]
        coordinates = value["coordinates"]
        if not isinstance(block_shape, list):
            raise TypeError("Topology block_shape must be a list.")
        if not isinstance(coordinates, list) or not all(
            isinstance(pair, list) for pair in coordinates
        ):
            raise TypeError("Topology coordinates must be a list of lists.")
        return cls(
            module_path=value["module_path"],
            in_features=value["in_features"],
            out_features=value["out_features"],
            block_shape=tuple(block_shape),
            coordinates=tuple(tuple(pair) for pair in coordinates),
        )


class _BlockSparseDeltaLinear(nn.Module):
    """Frozen linear projection plus compact trainable block updates."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        topology: BlockSparseModuleTopology,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear.")
        if (
            base_layer.in_features != topology.in_features
            or base_layer.out_features != topology.out_features
        ):
            raise ValueError("Base linear dimensions do not match topology.")
        if not base_layer.weight.is_floating_point():
            raise TypeError("Block-sparse projections must be floating point.")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.topology = topology
        self._coordinates = topology.coordinates
        self.register_buffer(
            "block_coordinates",
            torch.tensor(
                topology.coordinates,
                dtype=torch.int64,
                device=base_layer.weight.device,
            ),
        )
        block_out, block_in = topology.block_shape
        self.block_values = nn.Parameter(
            torch.zeros(
                len(topology.coordinates),
                block_out,
                block_in,
                dtype=base_layer.weight.dtype,
                device=base_layer.weight.device,
            )
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the frozen base and active blocks without a dense delta."""

        base_output = self.base_layer(hidden_states)
        block_out, block_in = self.topology.block_shape
        output_block_count = self.topology.out_features // block_out
        by_output_block: dict[int, list[tuple[int, int]]] = {}
        for parameter_index, (output_index, input_index) in enumerate(
            self._coordinates
        ):
            by_output_block.setdefault(output_index, []).append(
                (parameter_index, input_index)
            )

        output_chunks: list[Tensor] = []
        for output_index in range(output_block_count):
            contributions: list[Tensor] = []
            for parameter_index, input_index in by_output_block.get(
                output_index,
                (),
            ):
                input_start = input_index * block_in
                input_chunk = hidden_states[
                    ..., input_start : input_start + block_in
                ]
                contributions.append(
                    F.linear(
                        input_chunk,
                        self.block_values[parameter_index],
                    )
                )
            if contributions:
                chunk = contributions[0]
                for contribution in contributions[1:]:
                    chunk = chunk + contribution
            else:
                chunk = base_output.new_zeros(
                    *base_output.shape[:-1],
                    block_out,
                )
            output_chunks.append(chunk)
        delta_output = torch.cat(output_chunks, dim=-1)
        return base_output + delta_output


class BlockSparseDeltaExpert(AnchorRelativeExpert):
    r"""Complete MLP expert with fixed trainable block-sparse projection deltas.

    For a targeted frozen projection \(W\), fixed block coordinates
    \(\mathcal B\), and trainable block values \(\Delta W_b\), EXP-003 executes

    \[
    y = Wh + \sum_{b\in\mathcal B} S_b(\Delta W_b h_b),
    \]

    where \(h_b\) gathers the block's input columns and \(S_b\) places its
    result into the corresponding output rows. Blocks do not overlap because
    coordinates are unique. All block tensors start at zero, making complete
    expert output equal to the compatible anchor at construction.

    Args:
        anchor_mlp:
            MLP copied before sparse wrappers are installed. A successful
            construction freezes the supplied live anchor but never retains or
            mutates it.
        config:
            Static target, block-shape, count, and deterministic-mask settings.
            ``None`` uses :class:`BlockSparseDeltaConfig` defaults.
        trainable:
            Whether sparse block values require gradients. Private base
            parameters remain frozen in either mode.

    The expert returns a complete MLP output for existing forced and routed
    execution. ``forward_delta`` also implements the common anchor-relative
    contract. Package binding/export and fused sparse kernels are separate
    features.

    References:
        NIST, "FIPS PUB 180-4: Secure Hash Standard (SHS)."
        https://doi.org/10.6028/NIST.FIPS.180-4

    Differences from the reference:
        SHA-256 deterministically ranks block coordinates. It does not make
        the resulting mask random, secret, authenticated, or tamper-evident.
    """

    parameterization: ClassVar[str] = "block_sparse_delta"

    def __init__(
        self,
        anchor_mlp: nn.Module,
        *,
        config: BlockSparseDeltaConfig | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        _validate_constructor_inputs(
            anchor_mlp,
            config=config,
            trainable=trainable,
        )
        resolved_config = config or BlockSparseDeltaConfig()
        prepared, topology = _prepare_private_mlp(
            anchor_mlp,
            resolved_config,
        )
        # The caller's anchor changes only after copying, target resolution,
        # divisibility checks, mask selection, and wrapper construction pass.
        anchor_mlp.requires_grad_(False)
        self._initialize_prepared(
            prepared,
            config=resolved_config,
            topology=topology,
            trainable=trainable,
            training=anchor_mlp.training,
        )

    @classmethod
    def from_anchor(
        cls,
        anchor_mlp: nn.Module,
        *,
        config: BlockSparseDeltaConfig | None = None,
        trainable: bool = True,
    ) -> BlockSparseDeltaExpert:
        """Copy an anchor and install deterministic zero-initialized blocks."""

        return cls(anchor_mlp, config=config, trainable=trainable)

    @classmethod
    def from_safetensors(
        cls,
        anchor_mlp: nn.Module,
        path: str | Path,
        *,
        trainable: bool = False,
    ) -> BlockSparseDeltaExpert:
        """Strictly reconstruct sparse values over a compatible anchor MLP.

        Metadata, configuration, resolved topology, keys, shapes, dtypes, and
        exact coordinate tensors are validated before the supplied anchor is
        frozen. The private base is reconstructed and never loaded from the
        payload.
        """

        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        source = _validate_load_path(path)
        metadata = _load_metadata(source)
        expected_fields = {
            "config",
            "format",
            "format_version",
            "module_class",
            "parameterization",
            "topology",
        }
        if set(metadata) != expected_fields:
            raise ExpertSerializationError(
                f"Block-sparse metadata fields are incompatible in {source}."
            )
        expected_metadata = {
            "format": _SERIALIZATION_FORMAT,
            "format_version": BLOCK_SPARSE_DELTA_FORMAT_VERSION,
            "module_class": _qualified_class_name(anchor_mlp),
            "parameterization": cls.parameterization,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ExpertSerializationError(
                f"Incompatible block-sparse metadata in {source}: {mismatches}"
            )
        try:
            config = BlockSparseDeltaConfig.from_dict(metadata["config"])
            saved_topology_value = metadata["topology"]
            if not isinstance(saved_topology_value, list):
                raise TypeError("topology must be serialized as a list.")
            saved_topology = tuple(
                BlockSparseModuleTopology.from_dict(value)
                for value in saved_topology_value
            )
            saved_state = load_file(source, device="cpu")
            prepared, topology = _prepare_private_mlp(anchor_mlp, config)
            if saved_topology != topology:
                raise ExpertSerializationError(
                    "Serialized block topology does not match the compatible "
                    "anchor and deterministic configuration."
                )
            expected_state = _sparse_state(prepared, topology=topology)
            _validate_state_schema(expected_state, saved_state)
            _load_sparse_state(
                prepared,
                topology=topology,
                saved_state=saved_state,
            )
        except ExpertSerializationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot load block-sparse tensors from {source}: {error}"
            ) from error

        anchor_mlp.requires_grad_(False)
        expert = cls.__new__(cls)
        Expert.__init__(expert)
        expert._initialize_prepared(
            prepared,
            config=config,
            topology=topology,
            trainable=trainable,
            training=anchor_mlp.training,
        )
        return expert

    def _initialize_prepared(
        self,
        prepared: nn.Module,
        *,
        config: BlockSparseDeltaConfig,
        topology: tuple[BlockSparseModuleTopology, ...],
        trainable: bool,
        training: bool,
    ) -> None:
        """Register an already wrapped private MLP without another copy."""

        self.config = config
        self.expert_mlp = prepared
        self.topology = topology
        self._delta_parameter_names = tuple(
            name
            for name, _ in self.expert_mlp.named_parameters()
            if name.endswith(".block_values")
        )
        if len(self._delta_parameter_names) != len(self.topology):
            raise RuntimeError(
                "Every sparse topology entry must own one block tensor."
            )
        self.set_trainable(trainable)
        self.train(training)

    @property
    def delta_parameter_names(self) -> tuple[str, ...]:
        """Return exact private-module names owned by the sparse optimizer."""

        return self._delta_parameter_names

    @property
    def delta_parameter_count(self) -> int:
        """Return the exact trainable and serialized scalar count."""

        return sum(item.active_parameter_count for item in self.topology)

    @property
    def dense_delta_parameter_count(self) -> int:
        """Return the dense target capacity represented by the sparse mask."""

        return sum(item.dense_parameter_count for item in self.topology)

    @property
    def base_parameter_count(self) -> int:
        """Return the scalar count of reconstructed frozen base parameters."""

        return self.parameter_count - self.delta_parameter_count

    def set_trainable(self, trainable: bool) -> None:
        """Toggle only sparse block values while retaining a frozen base."""

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        self.expert_mlp.requires_grad_(False)
        if trainable:
            delta_names = set(self._delta_parameter_names)
            for name, parameter in self.expert_mlp.named_parameters():
                if name in delta_names:
                    parameter.requires_grad_(True)

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute the complete private MLP plus compact block updates."""

        return self.expert_mlp(hidden_states, *args, **kwargs)

    def forward_delta(
        self,
        hidden_states: Tensor,
        *args: Any,
        anchor_output: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Return complete sparse-expert output relative to a live anchor."""

        if not isinstance(anchor_output, Tensor):
            raise TypeError("anchor_output must be a torch.Tensor.")
        expert_output = self.expert_mlp(hidden_states, *args, **kwargs)
        if expert_output.shape != anchor_output.shape:
            raise ValueError(
                "anchor_output must have the complete expert output shape: "
                f"{tuple(anchor_output.shape)} != {tuple(expert_output.shape)}."
            )
        if expert_output.dtype != anchor_output.dtype:
            raise ValueError(
                "anchor_output and expert output must share one dtype."
            )
        if expert_output.device != anchor_output.device:
            raise ValueError(
                "anchor_output and expert output must share one device."
            )
        return expert_output - anchor_output

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save block values, coordinates, and canonical metadata."""

        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in sorted(
                _sparse_state(
                    self.expert_mlp,
                    topology=self.topology,
                ).items()
            )
        }
        for name, tensor in tensor_state.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise ExpertSerializationError(
                    f"Sparse tensor {name} must contain only finite values."
                )
        metadata_values = {
            "config": self.config.to_dict(),
            "format": _SERIALIZATION_FORMAT,
            "format_version": BLOCK_SPARSE_DELTA_FORMAT_VERSION,
            "module_class": _qualified_class_name(self.expert_mlp),
            "parameterization": self.parameterization,
            "topology": [item.to_dict() for item in self.topology],
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
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot save block-sparse tensors to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _validate_constructor_inputs(
    anchor_mlp: nn.Module,
    *,
    config: BlockSparseDeltaConfig | None,
    trainable: bool,
) -> None:
    """Validate public inputs without mutating the supplied live anchor."""

    if not isinstance(anchor_mlp, nn.Module):
        raise TypeError("anchor_mlp must be a torch.nn.Module.")
    if config is not None and not isinstance(config, BlockSparseDeltaConfig):
        raise TypeError("config must be a BlockSparseDeltaConfig or None.")
    if not isinstance(trainable, bool):
        raise TypeError("trainable must be a boolean.")


def _prepare_private_mlp(
    anchor_mlp: nn.Module,
    config: BlockSparseDeltaConfig,
) -> tuple[nn.Module, tuple[BlockSparseModuleTopology, ...]]:
    """Copy one MLP, resolve targets, and install zero sparse wrappers."""

    try:
        prepared = deepcopy(anchor_mlp)
    except Exception as error:
        raise RuntimeError(
            f"Cannot copy anchor module {type(anchor_mlp).__qualname__}."
        ) from error
    prepared.requires_grad_(False)
    targets = _resolve_target_linears(prepared, config.target_modules)
    topology: list[BlockSparseModuleTopology] = []
    for module_path, linear in targets:
        block_out, block_in = config.block_shape
        if linear.out_features % block_out or linear.in_features % block_in:
            raise ValueError(
                f"Target {module_path} dimensions "
                f"({linear.out_features}, {linear.in_features}) must be "
                f"divisible by block_shape={config.block_shape}."
            )
        output_blocks = linear.out_features // block_out
        input_blocks = linear.in_features // block_in
        total_blocks = output_blocks * input_blocks
        if config.blocks_per_module > total_blocks:
            raise ValueError(
                f"blocks_per_module={config.blocks_per_module} exceeds "
                f"{total_blocks} blocks available in {module_path}."
            )
        coordinates = _select_coordinates(
            module_path,
            output_blocks=output_blocks,
            input_blocks=input_blocks,
            block_count=config.blocks_per_module,
            seed=config.seed,
        )
        module_topology = BlockSparseModuleTopology(
            module_path=module_path,
            in_features=linear.in_features,
            out_features=linear.out_features,
            block_shape=config.block_shape,
            coordinates=coordinates,
        )
        _replace_submodule(
            prepared,
            module_path,
            _BlockSparseDeltaLinear(
                linear,
                topology=module_topology,
            ),
        )
        topology.append(module_topology)
    return prepared, tuple(topology)


def _resolve_target_linears(
    module: nn.Module,
    target_suffixes: tuple[str, ...],
) -> tuple[tuple[str, nn.Linear], ...]:
    """Resolve every suffix to exactly one distinct floating linear module."""

    named_modules = tuple(module.named_modules())
    resolved: list[tuple[str, nn.Linear]] = []
    seen_paths: set[str] = set()
    for suffix in target_suffixes:
        matches = tuple(
            (name, candidate)
            for name, candidate in named_modules
            if name and (name == suffix or name.endswith(f".{suffix}"))
        )
        if len(matches) != 1:
            raise ValueError(
                f"target_modules suffix {suffix!r} must resolve to exactly "
                f"one module, found {len(matches)}."
            )
        module_path, candidate = matches[0]
        if not isinstance(candidate, nn.Linear):
            raise TypeError(
                f"Target {module_path} must be torch.nn.Linear, found "
                f"{type(candidate).__qualname__}."
            )
        if module_path in seen_paths:
            raise ValueError(
                f"target_modules resolve repeatedly to {module_path}."
            )
        seen_paths.add(module_path)
        resolved.append((module_path, candidate))
    return tuple(sorted(resolved, key=lambda item: item[0]))


def _replace_submodule(
    root: nn.Module,
    module_path: str,
    replacement: nn.Module,
) -> None:
    """Replace one resolved descendant while preserving its canonical path."""

    parent_path, separator, child_name = module_path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    setattr(parent, child_name, replacement)


def _select_coordinates(
    module_path: str,
    *,
    output_blocks: int,
    input_blocks: int,
    block_count: int,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    """Rank the complete block grid by stable SHA-256 and select a prefix."""

    ranked: list[tuple[bytes, int, int]] = []
    for output_index in range(output_blocks):
        for input_index in range(input_blocks):
            identity = (
                f"anchored_moe.exp003.v1|{seed}|{module_path}|"
                f"{output_index}|{input_index}"
            ).encode("utf-8")
            ranked.append(
                (
                    hashlib.sha256(identity).digest(),
                    output_index,
                    input_index,
                )
            )
    ranked.sort()
    return tuple(
        sorted(
            (output_index, input_index)
            for _, output_index, input_index in ranked[:block_count]
        )
    )


def _sparse_state(
    module: nn.Module,
    *,
    topology: tuple[BlockSparseModuleTopology, ...],
) -> dict[str, Tensor]:
    """Return the complete sparse-only state in topology order."""

    state: dict[str, Tensor] = {}
    for index, item in enumerate(topology):
        sparse_linear = module.get_submodule(item.module_path)
        if not isinstance(sparse_linear, _BlockSparseDeltaLinear):
            raise ExpertSerializationError(
                f"Target {item.module_path} is not a sparse projection."
            )
        state[f"modules.{index}.block_coordinates"] = (
            sparse_linear.block_coordinates
        )
        state[f"modules.{index}.block_values"] = sparse_linear.block_values
    return state


def _load_sparse_state(
    module: nn.Module,
    *,
    topology: tuple[BlockSparseModuleTopology, ...],
    saved_state: dict[str, Tensor],
) -> None:
    """Copy validated values and verify exact fixed coordinate tensors."""

    with torch.no_grad():
        for index, item in enumerate(topology):
            sparse_linear = module.get_submodule(item.module_path)
            if not isinstance(sparse_linear, _BlockSparseDeltaLinear):
                raise ExpertSerializationError(
                    f"Target {item.module_path} is not a sparse projection."
                )
            coordinate_key = f"modules.{index}.block_coordinates"
            value_key = f"modules.{index}.block_values"
            expected_coordinates = (
                sparse_linear.block_coordinates.detach().cpu()
            )
            if not torch.equal(
                saved_state[coordinate_key],
                expected_coordinates,
            ):
                raise ExpertSerializationError(
                    f"Sparse coordinates changed for {item.module_path}."
                )
            values = saved_state[value_key]
            if not torch.isfinite(values).all().item():
                raise ExpertSerializationError(
                    f"Sparse values are non-finite for {item.module_path}."
                )
            sparse_linear.block_values.copy_(
                values.to(
                    device=sparse_linear.block_values.device,
                    dtype=sparse_linear.block_values.dtype,
                )
            )


def _validate_state_schema(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
) -> None:
    """Reject sparse key, shape, or dtype changes before activation."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ExpertSerializationError(
            "Block-sparse state keys are incompatible: "
            f"missing={missing}, unexpected={unexpected}."
        )
    mismatches: list[str] = []
    for name in sorted(expected):
        expected_tensor = expected[name]
        actual_tensor = actual[name]
        if expected_tensor.shape != actual_tensor.shape:
            mismatches.append(
                f"{name}: shape {tuple(actual_tensor.shape)} != "
                f"{tuple(expected_tensor.shape)}"
            )
        if expected_tensor.dtype != actual_tensor.dtype:
            mismatches.append(
                f"{name}: dtype {actual_tensor.dtype} != {expected_tensor.dtype}"
            )
    if mismatches:
        raise ExpertSerializationError(
            "Block-sparse tensor schema is incompatible: "
            + "; ".join(mismatches)
        )


def _load_metadata(source: Path) -> dict[str, Any]:
    """Parse the one canonical safetensors metadata envelope."""

    try:
        with safe_open(source, framework="pt", device="cpu") as payload:
            serialized = (payload.metadata() or {}).get(_METADATA_KEY)
    except Exception as error:
        raise ExpertSerializationError(
            f"Cannot read safetensors payload {source}: {error}"
        ) from error
    try:
        metadata = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ExpertSerializationError(
            f"Invalid block-sparse metadata in {source}."
        ) from error
    if not isinstance(metadata, dict):
        raise ExpertSerializationError(
            f"Invalid block-sparse metadata in {source}."
        )
    return metadata


def _qualified_class_name(module: nn.Module) -> str:
    """Return the stable original top-level MLP class identifier."""

    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _validate_load_path(path: str | Path) -> Path:
    """Resolve the one regular non-symlink safetensors source accepted."""

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
    """Validate an immutable destination without creating directories."""

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
            f"Expert payload parent directory does not exist: {destination.parent}"
        )
    return destination
