"""Acceptance tests for immutable expert-mixture compatibility metadata."""

from __future__ import annotations

import pytest

from anchored_moe import MixtureCompatibilityIndex


def test_compatibility_requires_one_shared_explicit_group() -> None:
    index = MixtureCompatibilityIndex(
        expert_ids=("code:0", "code:1", "general:0"),
        group_ids=("code-v1", "code-v1", None),
    )

    assert index.group_for("code:0") == "code-v1"
    assert index.group_for("general:0") is None
    assert index.are_compatible("code:0", "code:1")
    assert not index.are_compatible("code:0", "general:0")
    assert not index.are_compatible("code:0", "code:0")


def test_groups_align_by_expert_identity_not_construction_order() -> None:
    index = MixtureCompatibilityIndex(
        expert_ids=("first", "second", "third"),
        group_ids=("alpha", None, "beta"),
    )

    assert index.groups_for(("third", "first", "second")) == (
        "beta",
        "alpha",
        None,
    )


def test_compatibility_index_rejects_stale_or_malformed_candidates() -> None:
    index = MixtureCompatibilityIndex(("a", "b"), ("group", "group"))

    with pytest.raises(ValueError, match="exactly match"):
        index.groups_for(("a",))
    with pytest.raises(ValueError, match="exactly match"):
        index.groups_for(("a", "c"))
    with pytest.raises(KeyError, match="Unknown expert_id"):
        index.group_for("missing")
    with pytest.raises(ValueError, match="equal length"):
        MixtureCompatibilityIndex(("a",), ("group", None))
    with pytest.raises(ValueError, match="duplicates"):
        MixtureCompatibilityIndex(("a", "a"), ("group", "group"))
    with pytest.raises(ValueError, match="surrounding whitespace"):
        MixtureCompatibilityIndex(("a",), (" group ",))


def test_compatibility_index_defensively_freezes_sequence_inputs() -> None:
    expert_ids = ["a"]
    group_ids = ["group"]
    index = MixtureCompatibilityIndex(  # type: ignore[arg-type]
        expert_ids=expert_ids,
        group_ids=group_ids,
    )
    expert_ids[0] = "changed"
    group_ids[0] = "changed"

    assert index.expert_ids == ("a",)
    assert index.group_ids == ("group",)

    with pytest.raises(TypeError, match="sequence of strings"):
        MixtureCompatibilityIndex(  # type: ignore[arg-type]
            expert_ids="a",
            group_ids=("group",),
        )
    with pytest.raises(TypeError, match="strings or None"):
        MixtureCompatibilityIndex(  # type: ignore[arg-type]
            expert_ids=("a",),
            group_ids="group",
        )
