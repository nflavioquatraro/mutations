"""Declarative, backend-resolved attention and MLP adapter targeting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn


@dataclass(frozen=True)
class AdapterTargetConfig:
    """Select decoder projections without changing the frozen backbone.

    Args:
        layers: ``all``, ``upper_half``, ``last_n``, or explicit layer indices.
        last_n: Required only for the ``last_n`` selector; never silently capped.
        components: Semantic ``attention`` and/or ``mlp`` groups.
        module_paths: Exact layer-relative linear paths; replaces components
            when nonempty (set ``components=()`` to avoid ambiguity).
        rank: Positive low-rank dimension.
        alpha: Positive LoRA scaling numerator.
        dropout: Adapter dropout in [0, 1).
        rank_pattern: Exact resolved module paths and rank overrides.
        alpha_pattern: Exact resolved module paths and alpha overrides.

    Resolution rejects nonexistent layers, missing/nonlinear targets, tied
    projection storage, and unmatched overrides before PEFT mutates a model.
    Normalization, biases, embeddings, and the LM head are not trainable.
    """

    layers: str | tuple[int, ...] = "last_n"
    last_n: int | None = 8
    components: tuple[str, ...] = ("attention", "mlp")
    module_paths: tuple[str, ...] = ()
    rank: int = 8
    alpha: float = 16
    dropout: float = 0.0
    rank_pattern: tuple[tuple[str, int], ...] = ()
    alpha_pattern: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate immutable selectors and finite LoRA hyperparameters."""

        if isinstance(self.layers, str):
            if self.layers not in {"all", "upper_half", "last_n"}:
                raise ValueError("Unknown layer selector.")
        elif (
            not isinstance(self.layers, tuple)
            or not self.layers
            or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in self.layers
            )
            or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("layers must be a selector or unique nonnegative indices.")
        if self.layers == "last_n":
            _positive(self.last_n, "last_n", integer=True)
        elif self.last_n is not None:
            raise ValueError("last_n must be None for other layer selectors.")
        for values in (self.components, self.module_paths):
            if (
                not isinstance(values, tuple)
                or any(
                    not isinstance(v, str) or not v or v.strip() != v for v in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError("Target groups and paths must be unique tuples.")
        if bool(self.components) == bool(self.module_paths):
            raise ValueError("Specify components or module_paths, exclusively.")
        if set(self.components) - {"attention", "mlp"}:
            raise ValueError("Unknown semantic component.")
        _positive(self.rank, "rank", integer=True)
        _positive(self.alpha, "alpha")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a number.")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1).")
        for field, integer in ((self.rank_pattern, True), (self.alpha_pattern, False)):
            if not isinstance(field, tuple):
                raise TypeError("Override patterns must be tuples of path/value pairs.")
            names = []
            for entry in field:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError("Invalid override pair.")
                name, value = entry
                if not isinstance(name, str) or not name or name.strip() != name:
                    raise ValueError("Override paths must be nonempty strings.")
                names.append(name)
                _positive(value, "override", integer=integer)
            if len(set(names)) != len(names):
                raise ValueError("Duplicate override path.")

    def to_dict(self) -> dict:
        """Return JSON-compatible configuration with deterministic fields."""

        return asdict(self)


@dataclass(frozen=True)
class AdapterTargetReport:
    """Resolved layer indices and exact model-relative projection paths."""

    model_type: str
    layer_indices: tuple[int, ...]
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class AdapterTargetBackend:
    """Structural targeting extension point, defaulting to Qwen3 causal LMs.

    Args:
        model_type: Required Hugging Face model family identifier.
        layers_path: Exact model-relative path to the ordered decoder layers.
        attention_paths: Exact layer-relative attention projections.
        mlp_paths: Exact layer-relative MLP projections.

    Custom backends may describe another family with this contract. Only Qwen3
    is tested. This does not alter the existing MLP conversion backend.
    """

    model_type: str = "qwen3"
    layers_path: str = "model.layers"
    attention_paths: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    mlp_paths: tuple[str, ...] = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    def resolve(
        self, model: nn.Module, config: AdapterTargetConfig
    ) -> AdapterTargetReport:
        """Resolve targets without mutation; reject incompatible model structures."""

        if (
            getattr(getattr(model, "config", None), "model_type", None)
            != self.model_type
        ):
            raise ValueError("Model family does not match the adapter backend.")
        layers = model.get_submodule(self.layers_path)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise ValueError("Backend layers_path must identify a nonempty ModuleList.")
        count = len(layers)
        if config.layers == "last_n":
            if config.last_n > count:
                raise ValueError("last_n exceeds decoder depth.")
            indices = tuple(range(count - config.last_n, count))
        elif config.layers == "all":
            indices = tuple(range(count))
        elif config.layers == "upper_half":
            indices = tuple(range(count // 2, count))
        else:
            indices = tuple(sorted(config.layers))
        if indices[-1] >= count:
            raise ValueError("Adapter layer index exceeds decoder depth.")
        paths = config.module_paths or tuple(
            path
            for group, group_paths in (
                ("attention", self.attention_paths),
                ("mlp", self.mlp_paths),
            )
            if group in config.components
            for path in group_paths
        )
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("Backend must resolve distinct nonempty projection paths.")
        names = tuple(
            sorted(f"{self.layers_path}.{i}.{p}" for i in indices for p in paths)
        )
        storage = set()
        for name in names:
            module = model.get_submodule(name)
            if isinstance(module, BaseTunerLayer):
                module = module.get_base_layer()
            if type(module) is not nn.Linear:
                raise ValueError(f"Target must be a native unwrapped Linear: {name}")
            key = (
                str(module.weight.device),
                module.weight.untyped_storage().data_ptr(),
            )
            if module.weight.device.type == "meta" or key in storage:
                raise ValueError(
                    "Targets must be materialized with independent storage."
                )
            storage.add(key)
        for pattern in (config.rank_pattern, config.alpha_pattern):
            if set(dict(pattern)) - set(names):
                raise ValueError("Every override must match an exact resolved target.")
        return AdapterTargetReport(self.model_type, indices, names)


def _positive(value: object, name: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(
        value, int if integer else (int, float)
    ):
        raise TypeError(
            f"{name} must be a positive {'integer' if integer else 'number'}."
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
