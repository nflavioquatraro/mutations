"""Privacy-safe route records and aggregate inference telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anchored_moe.policies.capacity import CapacityDecision
from anchored_moe.policies.selection import RouteDecision


@dataclass(frozen=True)
class RouteRecord:
    """Detached route metadata for one routed-layer invocation.

    The record contains candidate indices, calibrated selected scores, and an
    overflow mask. It deliberately contains no token IDs, token text, hidden
    states, logits, or expert activations. Tensors are copied to CPU storage so
    later mutations of inference tensors cannot alter an audit record.
    """

    layer_index: int
    selected_indices: Tensor
    selected_scores: Tensor
    effective_indices: Tensor
    overflow_mask: Tensor
    expert_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate immutable record structure and CPU storage."""

        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        tensors = (
            self.selected_indices,
            self.selected_scores,
            self.effective_indices,
            self.overflow_mask,
        )
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("RouteRecord tensor fields must be torch.Tensor values.")
        if self.selected_indices.dtype != torch.long:
            raise TypeError("selected_indices must use torch.long.")
        if self.effective_indices.dtype != torch.long:
            raise TypeError("effective_indices must use torch.long.")
        if not self.selected_scores.is_floating_point():
            raise TypeError("selected_scores must use a floating-point dtype.")
        if self.overflow_mask.dtype != torch.bool:
            raise TypeError("overflow_mask must use torch.bool.")
        shape = self.selected_indices.shape
        if any(value.shape != shape for value in tensors[1:]):
            raise ValueError("RouteRecord tensor fields must share one shape.")
        if any(value.device.type != "cpu" for value in tensors):
            raise ValueError("RouteRecord tensors must use CPU storage.")
        if not isinstance(self.expert_ids, tuple) or not all(
            isinstance(expert_id, str) for expert_id in self.expert_ids
        ):
            raise TypeError("expert_ids must be a tuple of strings.")

    @property
    def token_count(self) -> int:
        """Return the number of recorded token-layer route decisions."""

        return self.selected_indices.numel()


@dataclass(frozen=True)
class RouteSummary:
    """Aggregate route counts without token-level model inputs.

    Attributes:
        layer_index: Zero-based decoder layer represented by the summary.
        invocation_count: Number of retained routed-layer calls.
        token_count: Total token-layer decisions across those calls.
        anchor_count: Effective candidate-zero decisions, including overflow.
        expert_counts: Stable ``(expert_id, count)`` pairs in router order.
        overflow_count: Specialist selections rewritten to the anchor.
    """

    layer_index: int
    invocation_count: int
    token_count: int
    anchor_count: int
    expert_counts: tuple[tuple[str, int], ...]
    overflow_count: int


class RouteTelemetry:
    """Collect bounded, privacy-safe route decisions for one layer.

    Args:
        layer_index:
            Zero-based decoder layer represented by every record.
        max_records:
            Positive upper bound on retained invocations. Once full, the oldest
            record is discarded before appending a new one.

    Telemetry is opt-in and in-memory. It stores routing metadata but never
    stores model inputs or activations. The bounded history prevents accidental
    unbounded growth during generation.
    """

    def __init__(self, *, layer_index: int, max_records: int = 128) -> None:
        """Create an empty bounded recorder."""

        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer.")
        self.layer_index = layer_index
        self.max_records = max_records
        self._records: list[RouteRecord] = []

    @property
    def records(self) -> tuple[RouteRecord, ...]:
        """Return retained records in invocation order."""

        return tuple(self._records)

    def record(
        self,
        selected_routes: RouteDecision,
        capacity: CapacityDecision,
    ) -> RouteRecord:
        """Copy one dispatch decision into bounded CPU history.

        Args:
            selected_routes: Routes before capacity fallback.
            capacity: Effective routes and overflow flags used for execution.

        Returns:
            The immutable record appended to this recorder.
        """

        if not isinstance(selected_routes, RouteDecision):
            raise TypeError("selected_routes must be a RouteDecision.")
        if not isinstance(capacity, CapacityDecision):
            raise TypeError("capacity must be a CapacityDecision.")
        if selected_routes.expert_ids != capacity.routes.expert_ids:
            raise ValueError("Selected and effective routes must share expert_ids.")
        if (
            selected_routes.selected_indices.shape
            != capacity.routes.selected_indices.shape
        ):
            raise ValueError("Selected and effective routes must share token shape.")

        route_record = RouteRecord(
            layer_index=self.layer_index,
            selected_indices=_cpu_copy(selected_routes.selected_indices),
            selected_scores=_cpu_copy(selected_routes.selected_scores),
            effective_indices=_cpu_copy(capacity.routes.selected_indices),
            overflow_mask=_cpu_copy(capacity.overflow_mask),
            expert_ids=selected_routes.expert_ids,
        )
        if len(self._records) == self.max_records:
            del self._records[0]
        self._records.append(route_record)
        return route_record

    def clear(self) -> None:
        """Discard all retained route records."""

        self._records.clear()

    def summarize(self) -> RouteSummary:
        """Aggregate effective anchor, expert, and overflow counts."""

        if not self._records:
            return RouteSummary(
                layer_index=self.layer_index,
                invocation_count=0,
                token_count=0,
                anchor_count=0,
                expert_counts=(),
                overflow_count=0,
            )

        expert_ids = self._records[0].expert_ids
        if any(record.expert_ids != expert_ids for record in self._records):
            raise RuntimeError(
                "Cannot summarize records from different expert registries."
            )
        anchor_count = 0
        expert_counts = [0] * len(expert_ids)
        overflow_count = 0
        token_count = 0
        for record in self._records:
            token_count += record.token_count
            anchor_count += int((record.effective_indices == 0).sum().item())
            overflow_count += int(record.overflow_mask.sum().item())
            for candidate_index in range(1, len(expert_ids) + 1):
                expert_counts[candidate_index - 1] += int(
                    (record.effective_indices == candidate_index).sum().item()
                )

        return RouteSummary(
            layer_index=self.layer_index,
            invocation_count=len(self._records),
            token_count=token_count,
            anchor_count=anchor_count,
            expert_counts=tuple(zip(expert_ids, expert_counts, strict=True)),
            overflow_count=overflow_count,
        )


def _cpu_copy(value: Tensor) -> Tensor:
    """Detach and clone inference metadata into independent CPU storage."""

    return value.detach().to(device="cpu").clone()
