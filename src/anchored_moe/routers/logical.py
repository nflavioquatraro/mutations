"""Frozen logical router compiled from independently owned affine heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from anchored_moe.routers.base import RouterScoreBank
from anchored_moe.routers.heads import AffineRouterHead, AppendOnlyRouterHeads


_ROUTER_ARCHITECTURE = "logical_append_only_columns"


class _FrozenAffineColumns(nn.Module):
    """Store one layer's compiled affine columns as immutable buffers."""

    def __init__(self, head: AffineRouterHead) -> None:
        super().__init__()
        weight, bias, temperature, threshold = _copy_head_state(head)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("temperature", temperature)
        self.register_buffer("threshold", threshold)

    def append(self, head: AffineRouterHead) -> None:
        """Append one validated row without retaining the source head."""

        weight, bias, temperature, threshold = _copy_head_state(head)
        # Replacing buffers is deliberate: the logical router has no trainable
        # parameter identity, while concatenation leaves incumbent values
        # unchanged inside one registered layer-local matrix.
        self.weight = torch.cat((self.weight, weight), dim=0)
        self.bias = torch.cat((self.bias, bias), dim=0)
        self.temperature = torch.cat((self.temperature, temperature), dim=0)
        self.threshold = torch.cat((self.threshold, threshold), dim=0)

    def forward(self, features: Tensor) -> Tensor:
        """Score compiled rows with shape-stable affine reductions.

        A single wider GEMM can change the floating-point reduction selected by
        a backend when a column is appended. Evaluating each matrix row with a
        stable ``[1, D]`` shape keeps incumbent results bit-exact across append
        operations while retaining one logical matrix as registered state.
        """

        calibrated_columns = []
        for index in range(self.weight.shape[0]):
            raw_score = F.linear(
                features,
                self.weight[index : index + 1],
                self.bias[index : index + 1],
            ).squeeze(-1)
            calibrated_columns.append(
                raw_score / self.temperature[index] - self.threshold[index]
            )
        return torch.stack(calibrated_columns, dim=-1)


class LogicalAppendOnlyRouter(RouterScoreBank):
    """Compile independent affine heads into frozen layer-local score matrices.

    RTR-003 treats RTR-002 heads as the package-owned fitting and serialization
    format. This registry-level runtime copies each head into one row of a
    layer-local matrix:

    ``G_l(h) = (Phi_l(h) W_l^T + b_l) / T_l - tau_l``.

    For ``K_l`` specialists and feature width ``D``, ``W_l`` has shape
    ``[K_l, D]`` and ``b_l``, ``T_l``, and ``tau_l`` have shape ``[K_l]``.
    The copied tensors are registered buffers, so the compiled router has no
    trainable parameters but preserves gradients with respect to its input
    features. Rows are evaluated with stable single-column reductions so
    appending cannot alter incumbent floating-point results. Appending copies a
    new source head and never mutates or aliases either the source head or
    incumbent compiled values.

    Args:
        feature_size: Shared final dimension of the frozen router features.

    This implementation is framework infrastructure, not a paper-derived
    algorithm. It deliberately introduces no new on-disk format; rebuild it
    from validated :class:`AppendOnlyRouterHeads` payloads.
    """

    architecture = _ROUTER_ARCHITECTURE

    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.feature_size = _validate_positive_integer(
            feature_size,
            field_name="feature_size",
        )
        self.layers = nn.ModuleDict()
        self._head_keys: list[tuple[int, str]] = []
        self._expert_ids_by_layer: dict[int, list[str]] = {}

    def __len__(self) -> int:
        """Return the number of compiled layer-expert columns."""

        return len(self._head_keys)

    @property
    def head_keys(self) -> tuple[tuple[int, str], ...]:
        """Return compiled ownership keys in immutable global append order."""

        return tuple(self._head_keys)

    @classmethod
    def from_independent_heads(
        cls,
        router_heads: AppendOnlyRouterHeads,
    ) -> LogicalAppendOnlyRouter:
        """Compile an independent head bank without retaining tensor aliases.

        Args:
            router_heads:
                RTR-002 bank whose global append order, ownership, affine
                values, temperatures, and thresholds will be copied.

        Returns:
            A frozen logical router. An empty source produces an empty router,
            which cannot score a layer until a head is appended.

        Raises:
            TypeError: If ``router_heads`` is not an independent affine bank.
            ValueError: If a source tensor is non-finite or incompatible.
        """

        if not isinstance(router_heads, AppendOnlyRouterHeads):
            raise TypeError("router_heads must be an AppendOnlyRouterHeads.")
        logical = cls(router_heads.feature_size)
        for head in router_heads.heads:
            logical.append(head)
        return logical

    def expert_ids_for_layer(self, layer_index: int) -> tuple[str, ...]:
        """Return layer-local expert identifiers in global append order."""

        normalized_layer = _validate_layer_index(layer_index)
        return tuple(self._expert_ids_by_layer.get(normalized_layer, ()))

    def append(self, head: AffineRouterHead) -> None:
        """Copy one compatible affine head into its layer-local matrix.

        Existing columns keep the same values and order. The operation validates
        all ownership, shape, dtype, device, and finite-value constraints before
        changing logical-router state.

        Raises:
            TypeError: If ``head`` is not an :class:`AffineRouterHead`.
            ValueError:
                If its feature space, ownership, tensor state, dtype, or device
                is incompatible with the compiled router.
        """

        if not isinstance(head, AffineRouterHead):
            raise TypeError("head must be an AffineRouterHead.")
        if head.feature_size != self.feature_size:
            raise ValueError(
                "Router head feature size does not match the logical router: "
                f"{head.feature_size} != {self.feature_size}."
            )
        if head.key in self._head_keys:
            raise ValueError(
                "Router head ownership key is already compiled: "
                f"{head.key!r}."
            )
        _validate_finite_head(head)

        incumbent_state = next(self.buffers(), None)
        candidate_weight = head.linear.weight
        if incumbent_state is not None:
            if candidate_weight.dtype != incumbent_state.dtype:
                raise ValueError(
                    "Logical router columns must share one dtype: "
                    f"{candidate_weight.dtype} != {incumbent_state.dtype}."
                )
            if candidate_weight.device != incumbent_state.device:
                raise ValueError(
                    "Logical router columns must share one device: "
                    f"{candidate_weight.device} != {incumbent_state.device}."
                )

        layer_index, expert_id = head.key
        layer_key = str(layer_index)
        if layer_key in self.layers:
            self.layers[layer_key].append(head)
        else:
            self.layers[layer_key] = _FrozenAffineColumns(head)

        self._head_keys.append(head.key)
        self._expert_ids_by_layer.setdefault(layer_index, []).append(expert_id)

    def forward(self, features: Tensor, *, layer_index: int) -> Tensor:
        """Score one layer in deterministic compiled-column order.

        Args:
            features: Floating tensor with shape ``[..., feature_size]``.
            layer_index: Non-negative decoder-layer index to score.

        Returns:
            Calibrated scores with shape ``features.shape[:-1] + (K_l,)``.

        Raises:
            TypeError: If features are not a floating-point tensor.
            ValueError: If shape, dtype, device, or layer index is incompatible.
            LookupError: If no columns are compiled for the requested layer.
        """

        _validate_features(features, feature_size=self.feature_size)
        normalized_layer = _validate_layer_index(layer_index)
        layer_key = str(normalized_layer)
        if layer_key not in self.layers:
            raise LookupError(
                "No logical router columns are compiled for layer "
                f"{normalized_layer}."
            )
        layer = self.layers[layer_key]
        if features.dtype != layer.weight.dtype:
            raise ValueError(
                "features dtype does not match the logical router: "
                f"{features.dtype} != {layer.weight.dtype}."
            )
        if features.device != layer.weight.device:
            raise ValueError(
                "features device does not match the logical router: "
                f"{features.device} != {layer.weight.device}."
            )
        return layer(features)

    def extra_repr(self) -> str:
        """Describe feature width, compiled column count, and layer count."""

        return (
            f"feature_size={self.feature_size}, column_count={len(self)}, "
            f"layer_count={len(self.layers)}"
        )


def compile_logical_router(
    router_heads: AppendOnlyRouterHeads,
) -> LogicalAppendOnlyRouter:
    """Compile RTR-002 heads into a frozen RTR-003 logical router.

    This convenience function is equivalent to
    :meth:`LogicalAppendOnlyRouter.from_independent_heads` and makes the
    package-head-to-registry-runtime transition explicit at call sites.
    """

    return LogicalAppendOnlyRouter.from_independent_heads(router_heads)


def _copy_head_state(
    head: AffineRouterHead,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return detached finite row tensors with no source-storage aliases."""

    _validate_finite_head(head)
    weight = head.linear.weight.detach().clone()
    bias = head.linear.bias.detach().clone()
    temperature = weight.new_tensor([head.temperature])
    threshold = weight.new_tensor([head.threshold])
    return weight, bias, temperature, threshold


def _validate_finite_head(head: AffineRouterHead) -> None:
    """Reject non-finite affine state before it enters a deployed matrix."""

    for name, value in head.linear.state_dict().items():
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(
                f"Router head tensor {name!r} must be finite floating-point."
            )


def _validate_features(features: Tensor, *, feature_size: int) -> None:
    if not isinstance(features, Tensor) or not features.is_floating_point():
        raise TypeError("features must be a floating-point torch.Tensor.")
    if features.ndim < 1:
        raise ValueError("features must have a final feature dimension.")
    if features.shape[-1] != feature_size:
        raise ValueError(
            "features final dimension does not match feature_size: "
            f"{features.shape[-1]} != {feature_size}."
        )


def _validate_positive_integer(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_layer_index(layer_index: int) -> int:
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer.")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative.")
    return layer_index
