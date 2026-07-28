"""Standalone PEFT/LoRA experts over private frozen anchor copies.

EXP-002 preserves the existing complete-expert execution contract while
reducing trainable and serialized state. Each expert deep-copies one anchor
MLP, freezes that private base, and asks PEFT to inject LoRA adapters into the
configured linear projections. The original anchor is frozen only after the
copy and adapter topology have been constructed successfully.

References:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models."
    https://arxiv.org/abs/2106.09685

    Hugging Face PEFT LoRA documentation.
    https://huggingface.co/docs/peft/main/conceptual_guides/lora

Differences from the paper:
    LoRA is applied to a private frozen MLP copy so the object continues to
    return a complete specialist MLP output under the framework's existing
    top-1/top-2 expert interface. This avoids mutating the live anchor but
    duplicates frozen base storage at runtime. Shared-anchor delta execution
    is deliberately deferred to COM-004.
"""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import peft
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from anchored_moe.experts.base import Expert, ExpertSerializationError


_SERIALIZATION_FORMAT: Final = "anchored_moe.peft_lora.v1"
_SAFETENSORS_SUFFIX: Final = ".safetensors"
_METADATA_KEY: Final = "anchored_moe"
_ADAPTER_NAME: Final = "default"


@dataclass(frozen=True)
class LoRAExpertConfig:
    """Configuration for one standalone PEFT/LoRA MLP expert.

    Args:
        rank:
            Positive LoRA rank ``r`` used by every targeted projection.
        alpha:
            Positive PEFT scaling numerator. Ordinary LoRA scales the update by
            ``alpha / rank``; rank-stabilized LoRA uses ``alpha / sqrt(rank)``.
        dropout:
            LoRA-branch dropout probability in the half-open interval
            ``[0, 1)``. It is active only while the expert is in training mode.
        target_modules:
            Unique module-name suffixes targeted inside the private MLP copy.
            Qwen3 commonly uses ``gate_proj``, ``up_proj``, and ``down_proj``.
        use_rslora:
            Whether PEFT should use rank-stabilized LoRA scaling.

    Bias training, modules-to-save, non-zero adapter initialization, DoRA, and
    task-specific wrappers are intentionally excluded. Those options would
    weaken adapter-only ownership or change this version-1 payload contract.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "down_proj",
        "gate_proj",
        "up_proj",
    )
    use_rslora: bool = False

    def __post_init__(self) -> None:
        """Normalize target ordering and validate the supported LoRA subset."""

        for field_name in ("rank", "alpha"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be a finite number in [0, 1).")
        if not isinstance(self.use_rslora, bool):
            raise TypeError("use_rslora must be a boolean.")
        if (
            not isinstance(self.target_modules, tuple)
            or not self.target_modules
        ):
            raise TypeError(
                "target_modules must be a non-empty tuple of module names."
            )
        normalized: list[str] = []
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
            normalized.append(module_name)
        if len(set(normalized)) != len(normalized):
            raise ValueError("target_modules must not contain duplicates.")
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "target_modules", tuple(sorted(normalized)))

    @property
    def scaling(self) -> float:
        """Return the configured LoRA or rank-stabilized multiplier."""

        denominator = self.rank**0.5 if self.use_rslora else self.rank
        return self.alpha / denominator

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible public configuration."""

        return {
            "alpha": self.alpha,
            "dropout": self.dropout,
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "use_rslora": self.use_rslora,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoRAExpertConfig:
        """Strictly reconstruct a serialized EXP-002 configuration."""

        if not isinstance(value, dict):
            raise TypeError("LoRA expert config must be a dictionary.")
        expected = {
            "alpha",
            "dropout",
            "rank",
            "target_modules",
            "use_rslora",
        }
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "LoRA expert config fields are incompatible: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}."
            )
        target_modules = value["target_modules"]
        if not isinstance(target_modules, list):
            raise TypeError("LoRA expert config target_modules must be a list.")
        return cls(
            rank=value["rank"],
            alpha=value["alpha"],
            dropout=value["dropout"],
            target_modules=tuple(target_modules),
            use_rslora=value["use_rslora"],
        )

    def to_peft_config(self) -> LoraConfig:
        """Build the supported PEFT configuration without model mutation."""

        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias="none",
            init_lora_weights=True,
            use_rslora=self.use_rslora,
        )


class PEFTLoRAExpert(Expert):
    r"""Complete MLP expert backed by PEFT LoRA adapter parameters.

    For immutable anchor MLP parameters ``W`` and low-rank factors ``A_k`` and
    ``B_k``, each targeted projection executes

    \[
    W_k h = W h + s B_k A_k h,
    \]

    where ``s = alpha / rank`` for ordinary LoRA. PEFT initializes ``B_k`` to
    zero, so a newly constructed expert exactly matches its private anchor copy
    in evaluation mode. The live anchor is never passed to PEFT.

    Args:
        anchor_mlp:
            Dense MLP copied before PEFT injection. A successful construction
            freezes the supplied anchor but does not retain or mutate it.
        config:
            Stable supported LoRA configuration. ``None`` uses
            :class:`LoRAExpertConfig` defaults.
        trainable:
            Whether adapter parameters should require gradients. Private base
            parameters remain frozen in either mode.

    The expert returns a complete MLP output and therefore works with existing
    forced-path and routed replacement execution. It is not a COM-004
    delta-only expert.

    References:
        Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models."
        https://arxiv.org/abs/2106.09685

        Hugging Face PEFT low-level adapter injection and checkpoint APIs.
        https://huggingface.co/docs/peft/main/developer_guides/low_level_api

    Differences from the referenced methods:
        PEFT mutates the module supplied for injection. This implementation
        injects only into an independently owned deep copy and stores just one
        adapter in a framework-specific safetensors envelope. The private base
        is reconstructed from the compatible anchor instead of being saved.
    """

    parameterization = "peft_lora"

    def __init__(
        self,
        anchor_mlp: nn.Module,
        *,
        config: LoRAExpertConfig | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        _validate_constructor_inputs(
            anchor_mlp,
            config=config,
            trainable=trainable,
        )
        resolved_config = config or LoRAExpertConfig()
        prepared = _prepare_private_mlp(anchor_mlp, resolved_config)
        # No mutation reaches the caller until copying and PEFT injection have
        # both succeeded. This makes invalid target-module configurations
        # failure-atomic with respect to the immutable anchor.
        anchor_mlp.requires_grad_(False)
        self._initialize_prepared(
            prepared,
            config=resolved_config,
            trainable=trainable,
            training=anchor_mlp.training,
        )

    @classmethod
    def from_anchor(
        cls,
        anchor_mlp: nn.Module,
        *,
        config: LoRAExpertConfig | None = None,
        trainable: bool = True,
    ) -> PEFTLoRAExpert:
        """Copy an anchor MLP, inject PEFT LoRA, and freeze the anchor."""

        return cls(
            anchor_mlp,
            config=config,
            trainable=trainable,
        )

    @classmethod
    def from_safetensors(
        cls,
        anchor_mlp: nn.Module,
        path: str | Path,
        *,
        trainable: bool = False,
    ) -> PEFTLoRAExpert:
        """Strictly reconstruct one adapter over a compatible anchor MLP.

        Metadata, configuration, source module class, adapter keys, shapes, and
        dtypes are validated before the supplied anchor is frozen. The saved
        PEFT version is recorded for reproducibility but schema-compatible
        later PEFT versions may load the payload.
        """

        if not isinstance(anchor_mlp, nn.Module):
            raise TypeError("anchor_mlp must be a torch.nn.Module.")
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        source = _validate_load_path(path)
        metadata = _load_metadata(source)
        expected_metadata = {
            "format": _SERIALIZATION_FORMAT,
            "parameterization": cls.parameterization,
            "module_class": _qualified_class_name(anchor_mlp),
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ExpertSerializationError(
                f"Incompatible PEFT/LoRA metadata in {source}: {mismatches}"
            )
        saved_peft_version = metadata.get("peft_version")
        if (
            not isinstance(saved_peft_version, str)
            or not saved_peft_version
        ):
            raise ExpertSerializationError(
                f"Invalid PEFT version metadata in {source}."
            )
        try:
            config = LoRAExpertConfig.from_dict(metadata.get("config"))
        except (TypeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Invalid PEFT/LoRA configuration in {source}: {error}"
            ) from error

        try:
            saved_state = load_file(source, device="cpu")
            prepared = _prepare_private_mlp(anchor_mlp, config)
            expected_state = _adapter_state(prepared)
            _validate_state_schema(expected_state, saved_state)
            set_peft_model_state_dict(
                prepared,
                saved_state,
                adapter_name=_ADAPTER_NAME,
                ignore_mismatched_sizes=False,
            )
            _validate_loaded_state(prepared, saved_state)
        except ExpertSerializationError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ExpertSerializationError(
                f"Cannot load PEFT/LoRA tensors from {source}: {error}"
            ) from error

        anchor_mlp.requires_grad_(False)
        expert = cls.__new__(cls)
        Expert.__init__(expert)
        expert._initialize_prepared(
            prepared,
            config=config,
            trainable=trainable,
            training=anchor_mlp.training,
        )
        return expert

    def _initialize_prepared(
        self,
        prepared: nn.Module,
        *,
        config: LoRAExpertConfig,
        trainable: bool,
        training: bool,
    ) -> None:
        """Register one already-injected private module without copying it."""

        self.config = config
        self.expert_mlp = prepared
        self._adapter_parameter_names = _adapter_parameter_names(prepared)
        if not self._adapter_parameter_names:
            raise RuntimeError("PEFT did not create any LoRA parameters.")
        self.set_trainable(trainable)
        self.train(training)

    @property
    def adapter_parameter_names(self) -> tuple[str, ...]:
        """Return exact private-module names owned by the LoRA optimizer."""

        return self._adapter_parameter_names

    @property
    def adapter_parameter_count(self) -> int:
        """Return the scalar count of serialized LoRA adapter parameters."""

        adapter_names = set(self._adapter_parameter_names)
        return sum(
            parameter.numel()
            for name, parameter in self.expert_mlp.named_parameters()
            if name in adapter_names
        )

    @property
    def base_parameter_count(self) -> int:
        """Return the scalar count of reconstructed frozen base parameters."""

        return self.parameter_count - self.adapter_parameter_count

    def set_trainable(self, trainable: bool) -> None:
        """Toggle only LoRA tensors while keeping the private base frozen."""

        if not isinstance(trainable, bool):
            raise TypeError("trainable must be a boolean.")
        self.expert_mlp.requires_grad_(False)
        if trainable:
            adapter_names = set(self._adapter_parameter_names)
            for name, parameter in self.expert_mlp.named_parameters():
                if name in adapter_names:
                    parameter.requires_grad_(True)

    def forward(
        self,
        hidden_states: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Execute the complete private MLP plus its active LoRA update."""

        return self.expert_mlp(hidden_states, *args, **kwargs)

    def save_safetensors(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        """Atomically save only PEFT adapter tensors and canonical metadata."""

        destination = _validate_save_path(path, overwrite=overwrite)
        tensor_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in sorted(_adapter_state(self.expert_mlp).items())
        }
        metadata_values = {
            "config": self.config.to_dict(),
            "format": _SERIALIZATION_FORMAT,
            "module_class": _qualified_class_name(self.expert_mlp),
            "parameterization": self.parameterization,
            "peft_version": peft.__version__,
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
                f"Cannot save PEFT/LoRA tensors to {destination}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _validate_constructor_inputs(
    anchor_mlp: nn.Module,
    *,
    config: LoRAExpertConfig | None,
    trainable: bool,
) -> None:
    """Validate construction inputs without mutating the supplied anchor."""

    if not isinstance(anchor_mlp, nn.Module):
        raise TypeError("anchor_mlp must be a torch.nn.Module.")
    if config is not None and not isinstance(config, LoRAExpertConfig):
        raise TypeError("config must be a LoRAExpertConfig or None.")
    if not isinstance(trainable, bool):
        raise TypeError("trainable must be a boolean.")


def _prepare_private_mlp(
    anchor_mlp: nn.Module,
    config: LoRAExpertConfig,
) -> nn.Module:
    """Copy an MLP and inject one ordinary PEFT adapter into the copy."""

    try:
        prepared = deepcopy(anchor_mlp)
    except Exception as error:
        raise RuntimeError(
            f"Cannot copy anchor module {type(anchor_mlp).__qualname__}."
        ) from error
    prepared.requires_grad_(False)
    try:
        injected = inject_adapter_in_model(
            config.to_peft_config(),
            prepared,
            adapter_name=_ADAPTER_NAME,
        )
    except Exception as error:
        raise ValueError(
            "Cannot inject PEFT LoRA into the private anchor copy; verify "
            f"target_modules={config.target_modules}: {error}"
        ) from error
    if injected is not prepared:
        raise RuntimeError("PEFT adapter injection replaced the private MLP.")
    return prepared


def _adapter_state(module: nn.Module) -> dict[str, Tensor]:
    """Return PEFT's adapter-only, adapter-name-neutral state mapping."""

    state = get_peft_model_state_dict(
        module,
        adapter_name=_ADAPTER_NAME,
        save_embedding_layers=False,
    )
    if not state or not all(isinstance(value, Tensor) for value in state.values()):
        raise ExpertSerializationError(
            "PEFT adapter state must be a non-empty tensor mapping."
        )
    return dict(state)


def _adapter_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Map PEFT checkpoint keys back to exact registered parameter names."""

    state_keys = set(_adapter_state(module))
    matched: list[str] = []
    for name, _ in module.named_parameters():
        neutral_name = name.replace(f".{_ADAPTER_NAME}.", ".")
        if neutral_name in state_keys:
            matched.append(name)
    if len(matched) != len(state_keys):
        unmatched = sorted(
            state_keys
            - {
                name.replace(f".{_ADAPTER_NAME}.", ".")
                for name in matched
            }
        )
        raise ExpertSerializationError(
            f"Cannot map PEFT adapter tensors to parameters: {unmatched}."
        )
    return tuple(matched)


def _load_metadata(source: Path) -> dict[str, Any]:
    """Parse the canonical metadata envelope without executable content."""

    try:
        with safe_open(source, framework="pt", device="cpu") as payload:
            serialized_metadata = (payload.metadata() or {}).get(_METADATA_KEY)
    except Exception as error:
        raise ExpertSerializationError(
            f"Cannot read safetensors payload {source}: {error}"
        ) from error
    try:
        metadata = json.loads(serialized_metadata)
    except (TypeError, json.JSONDecodeError) as error:
        raise ExpertSerializationError(
            f"Invalid PEFT/LoRA metadata in {source}."
        ) from error
    if not isinstance(metadata, dict):
        raise ExpertSerializationError(
            f"Invalid PEFT/LoRA metadata in {source}."
        )
    return metadata


def _validate_loaded_state(
    module: nn.Module,
    expected: dict[str, Tensor],
) -> None:
    """Verify PEFT assigned every adapter tensor exactly after loading."""

    actual = _adapter_state(module)
    _validate_state_schema(expected, actual)
    mismatched = [
        name
        for name in sorted(expected)
        if not expected[name].equal(actual[name].detach().cpu())
    ]
    if mismatched:
        raise ExpertSerializationError(
            f"PEFT did not restore adapter tensors exactly: {mismatched}."
        )


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


def _validate_state_schema(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
) -> None:
    """Reject adapter key, shape, or dtype changes before activation."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ExpertSerializationError(
            "PEFT/LoRA state keys are incompatible: "
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
            "PEFT/LoRA tensor schema is incompatible: " + "; ".join(mismatches)
        )
