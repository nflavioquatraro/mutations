"""Qwen3 backend for discovering and wrapping decoder MLP modules."""

from __future__ import annotations

from typing import Iterator, Sequence

from torch import nn

from anchored_moe.backends.base import (
    ConversionReport,
    MLPReference,
    ModelBackend,
)
from anchored_moe.configuration import AnchoredMoEConfig
from anchored_moe.modeling import RoutedMLP


class Qwen3Backend(ModelBackend):
    """Convert Qwen3 decoder MLPs without changing dense computation.

    The adapter supports both ``Qwen3ForCausalLM`` (whose decoder layers live
    under ``model.model.layers``) and the bare ``Qwen3Model`` (whose layers live
    under ``model.layers``). Structural checks are deliberately explicit so a
    similarly shaped but incompatible model cannot be converted accidentally.
    """

    model_type = "qwen3"

    def supports_model(self, model: nn.Module) -> bool:
        """Return whether the model declares Qwen3 and exposes decoder layers."""

        model_config = getattr(model, "config", None)
        if getattr(model_config, "model_type", None) != self.model_type:
            return False
        try:
            layers = self._decoder_layers(model)
        except (AttributeError, TypeError):
            return False
        return bool(layers) and all(
            isinstance(getattr(layer, "mlp", None), nn.Module) for layer in layers
        )

    def iter_mlp_references(self, model: nn.Module) -> Iterator[MLPReference]:
        """Yield Qwen3 ``layer.mlp`` modules in decoder order."""

        if not self.supports_model(model):
            raise TypeError(
                "Qwen3Backend requires a model with config.model_type='qwen3' "
                "and decoder layers exposing an 'mlp' module."
            )

        for layer_index, layer in enumerate(self._decoder_layers(model)):
            yield MLPReference(
                layer_index=layer_index,
                parent=layer,
                attribute_name="mlp",
                module=layer.mlp,
            )

    def replace_mlp(
        self,
        reference: MLPReference,
        replacement: nn.Module,
    ) -> None:
        """Replace the exact MLP represented by ``reference``."""

        if not isinstance(replacement, nn.Module):
            raise TypeError("replacement must be a torch.nn.Module.")
        current = getattr(reference.parent, reference.attribute_name, None)
        if current is not reference.module:
            raise RuntimeError(
                f"Layer {reference.layer_index} changed after discovery; "
                "refusing a stale replacement."
            )
        setattr(reference.parent, reference.attribute_name, replacement)

    def convert(
        self,
        model: nn.Module,
        config: AnchoredMoEConfig,
    ) -> ConversionReport:
        """Freeze the full model and wrap the configured Qwen3 MLPs.

        Conversion is intentionally not idempotent. Encountering an existing
        ``RoutedMLP`` raises instead of creating nested wrappers or obscuring
        which operation owns the model state.
        """

        if config.backend != self.model_type:
            raise ValueError(
                f"Configuration selects backend {config.backend!r}, "
                f"but this backend is {self.model_type!r}."
            )

        references = tuple(self.iter_mlp_references(model))
        selected = config.resolve_converted_layers(len(references))
        selected_set = set(selected)

        for reference in references:
            if reference.layer_index in selected_set and isinstance(
                reference.module, RoutedMLP
            ):
                raise RuntimeError(
                    f"Qwen3 layer {reference.layer_index} is already routed."
                )

        # The shared backbone, embeddings, normalization, LM head, and every
        # anchor MLP are immutable by contract. New experts will be attached
        # after this operation and can independently opt into gradients.
        model.requires_grad_(False)
        for reference in references:
            if reference.layer_index in selected_set:
                self.replace_mlp(reference, RoutedMLP(reference.module))

        return ConversionReport(
            backend=self.model_type,
            total_layers=len(references),
            converted_layers=selected,
        )

    @staticmethod
    def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
        causal_lm_body = getattr(model, "model", None)
        if causal_lm_body is not None and hasattr(causal_lm_body, "layers"):
            layers = causal_lm_body.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Model has no supported Qwen3 decoder layer path.")

        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError("Qwen3 decoder layers must be an ordered module sequence.")
        return layers
