"""Hugging Face configuration for anchored mixture-of-experts models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from transformers import PretrainedConfig


_DEFAULT_STRATEGIES: dict[str, dict[str, Any]] = {
    "router": {"architecture": "none"},
    "selection": {"policy": "forced_anchor"},
    "combination": {"policy": "hard"},
    "capacity": {"overflow": "anchor"},
    "registry": {"recalibration": "none"},
}

_FOUNDATION_STRATEGIES: dict[str, tuple[str, str]] = {
    "router": ("architecture", "none"),
    "selection": ("policy", "forced_anchor"),
    "combination": ("policy", "hard"),
    "capacity": ("overflow", "anchor"),
    "registry": ("recalibration", "none"),
}


@dataclass(frozen=True)
class ResolvedStrategyConfig:
    """Validated strategy blocks used by the current model instance.

    Each block is stored independently so later router, selection, combination,
    capacity, and registry implementations can evolve without coupling their
    configuration parsers. The foundation release deliberately accepts only
    forced-anchor execution.
    """

    router: Mapping[str, Any]
    selection: Mapping[str, Any]
    combination: Mapping[str, Any]
    capacity: Mapping[str, Any]
    registry: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        strategy_config: Mapping[str, Mapping[str, Any]] | None,
    ) -> ResolvedStrategyConfig:
        """Merge user strategy blocks with deterministic foundation defaults."""

        resolved = deepcopy(_DEFAULT_STRATEGIES)
        if strategy_config is not None:
            unknown_blocks = sorted(set(strategy_config) - set(resolved))
            if unknown_blocks:
                names = ", ".join(unknown_blocks)
                raise ValueError(f"Unknown strategy configuration block(s): {names}")

            for block_name, block_values in strategy_config.items():
                if not isinstance(block_values, Mapping):
                    raise TypeError(
                        f"Strategy block {block_name!r} must be a mapping, "
                        f"got {type(block_values).__name__}."
                    )
                resolved[block_name].update(deepcopy(dict(block_values)))

        # Reject strategies whose runtime does not exist yet. This makes a
        # configuration executable rather than silently falling back to anchor.
        for block_name, (field_name, supported_value) in _FOUNDATION_STRATEGIES.items():
            actual_value = resolved[block_name].get(field_name)
            if actual_value != supported_value:
                raise ValueError(
                    f"Unsupported {block_name}.{field_name}={actual_value!r}; "
                    f"the foundation release supports only {supported_value!r}."
                )

        return cls(**resolved)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-serializable defensive copy of all strategy blocks."""

        return {
            "router": deepcopy(dict(self.router)),
            "selection": deepcopy(dict(self.selection)),
            "combination": deepcopy(dict(self.combination)),
            "capacity": deepcopy(dict(self.capacity)),
            "registry": deepcopy(dict(self.registry)),
        }


class AnchoredMoEConfig(PretrainedConfig):
    """Configuration for an anchored model and its composable strategies.

    Args:
        base_model_name_or_path:
            Hugging Face repository identifier or local path for the immutable
            anchor. It may be omitted for randomly initialized test models.
        base_model_revision:
            Exact Hub revision of the anchor when one is available.
        base_config_sha256:
            Optional SHA-256 digest of the resolved base-model configuration.
        tokenizer_name_or_path:
            Tokenizer repository or local path. Defaults to the base model.
        tokenizer_revision:
            Exact tokenizer revision when one is available.
        backend:
            Model-family adapter used to locate and replace MLP modules.
        converted_layers:
            Either ``"upper_half"`` or an explicit sequence of zero-based
            decoder-layer indices.
        strategy_config:
            Independent router, selection, combination, capacity, and registry
            configuration blocks.

    The full checkpoint and tokenizer pins are optional only because unit tests
    construct models from local random configurations. Production package
    loading will require them when compatibility validation is implemented.
    """

    model_type = "anchored_moe"

    def __init__(
        self,
        *,
        base_model_name_or_path: str | None = None,
        base_model_revision: str | None = None,
        base_config_sha256: str | None = None,
        tokenizer_name_or_path: str | None = None,
        tokenizer_revision: str | None = None,
        backend: str = "qwen3",
        converted_layers: str | Sequence[int] = "upper_half",
        strategy_config: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_revision = base_model_revision
        self.base_config_sha256 = base_config_sha256
        self.tokenizer_name_or_path = (
            tokenizer_name_or_path or base_model_name_or_path
        )
        self.tokenizer_revision = tokenizer_revision
        self.backend = self._validate_backend(backend)
        self.converted_layers = self._normalize_converted_layers(converted_layers)

        resolved = ResolvedStrategyConfig.from_mapping(strategy_config)
        self.strategy_config = resolved.to_dict()

    @property
    def resolved_strategies(self) -> ResolvedStrategyConfig:
        """Return the validated, immutable view of the strategy configuration."""

        return ResolvedStrategyConfig.from_mapping(self.strategy_config)

    def resolve_converted_layers(self, num_hidden_layers: int) -> tuple[int, ...]:
        """Resolve the configured selector against a concrete decoder depth.

        ``upper_half`` starts at ``num_hidden_layers // 2``. For an odd depth,
        the upper partition therefore contains one more layer than the lower
        partition. Explicit indices are normalized into ascending order at
        construction time.
        """

        if isinstance(num_hidden_layers, bool) or not isinstance(num_hidden_layers, int):
            raise TypeError("num_hidden_layers must be an integer.")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be greater than zero.")

        if self.converted_layers == "upper_half":
            return tuple(range(num_hidden_layers // 2, num_hidden_layers))

        indices = tuple(self.converted_layers)
        out_of_range = [
            index for index in indices if index < 0 or index >= num_hidden_layers
        ]
        if out_of_range:
            raise ValueError(
                "Converted layer indices are outside the model depth "
                f"{num_hidden_layers}: {out_of_range}"
            )
        return indices

    @staticmethod
    def _validate_backend(backend: str) -> str:
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend must be a non-empty string.")
        return backend.strip()

    @staticmethod
    def _normalize_converted_layers(
        converted_layers: str | Sequence[int],
    ) -> str | list[int]:
        if isinstance(converted_layers, str):
            if converted_layers != "upper_half":
                raise ValueError(
                    "converted_layers must be 'upper_half' or an explicit "
                    "sequence of layer indices."
                )
            return converted_layers

        if not isinstance(converted_layers, Sequence):
            raise TypeError(
                "converted_layers must be 'upper_half' or a sequence of integers."
            )

        normalized: list[int] = []
        for index in converted_layers:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("Every converted layer index must be an integer.")
            normalized.append(index)

        if not normalized:
            raise ValueError("At least one converted layer must be selected.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("converted_layers must not contain duplicate indices.")
        return sorted(normalized)
