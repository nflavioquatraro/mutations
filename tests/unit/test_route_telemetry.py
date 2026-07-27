"""Acceptance tests for privacy-safe bounded route telemetry."""

from __future__ import annotations

import torch

from anchored_moe import (
    AnchoredTop1Selection,
    OverflowToAnchorCapacity,
    RouteTelemetry,
)


def _decisions() -> tuple:
    selected = AnchoredTop1Selection()(
        torch.tensor([[0.5, 0.2], [-0.2, -0.1], [0.1, 0.8]]),
        expert_ids=("code", "math"),
    )
    capacity = OverflowToAnchorCapacity()(
        selected,
        expert_availability=torch.tensor(
            [[False, True], [True, True], [True, True]]
        ),
    )
    return selected, capacity


def test_telemetry_copies_only_route_metadata_and_summarizes() -> None:
    selected, capacity = _decisions()
    telemetry = RouteTelemetry(layer_index=3, max_records=2)

    record = telemetry.record(selected, capacity)
    selected.selected_indices.fill_(0)

    assert record.layer_index == 3
    assert record.token_count == 3
    assert torch.equal(record.selected_indices, torch.tensor([1, 0, 2]))
    assert torch.equal(record.effective_indices, torch.tensor([0, 0, 2]))
    assert not hasattr(record, "hidden_states")
    assert not hasattr(record, "token_ids")

    summary = telemetry.summarize()
    assert summary.invocation_count == 1
    assert summary.token_count == 3
    assert summary.anchor_count == 2
    assert summary.expert_counts == (("code", 0), ("math", 1))
    assert summary.overflow_count == 1


def test_telemetry_history_is_bounded_and_clearable() -> None:
    selected, capacity = _decisions()
    telemetry = RouteTelemetry(layer_index=1, max_records=1)

    first = telemetry.record(selected, capacity)
    second = telemetry.record(selected, capacity)

    assert telemetry.records == (second,)
    assert telemetry.records[0] is not first
    telemetry.clear()
    assert telemetry.records == ()
    assert telemetry.summarize().token_count == 0
