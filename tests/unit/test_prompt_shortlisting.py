"""Acceptance tests for RTR-006 prompt-level candidate shortlisting."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    PromptCandidateShortlist,
    PromptShortlistedLogicalRouter,
    PromptShortlistingError,
    PromptTopMShortlister,
    compile_logical_router,
)


def _logical_router() -> tuple[AppendOnlyRouterHeads, object]:
    heads = AppendOnlyRouterHeads(2)
    specifications = (
        ("code:0", (1.0, 0.0), 0.1),
        ("math:0", (0.0, 1.0), 0.2),
        ("science:0", (1.0, 1.0), 0.3),
    )
    for expert_id, weight, bias in specifications:
        head = AffineRouterHead(
            2,
            layer_index=4,
            expert_id=expert_id,
        )
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
            head.linear.bias.fill_(bias)
        heads.append(head)
    return heads, compile_logical_router(heads)


def test_top_m_is_stable_bounded_and_detached_from_retrieval_scores() -> None:
    retrieval_scores = torch.tensor(
        [
            [2.0, 2.0, 1.0],
            [-3.0, -1.0, -2.0],
        ],
        requires_grad=True,
    )
    shortlister = PromptTopMShortlister(max_candidates=2)

    shortlist = shortlister(
        retrieval_scores,
        candidate_ids=("code:0", "math:0", "science:0"),
    )
    retrieval_scores.data.fill_(100.0)

    assert torch.equal(
        shortlist.selected_indices,
        torch.tensor([[0, 1], [1, 2]]),
    )
    assert torch.equal(
        shortlist.selected_scores,
        torch.tensor([[2.0, 2.0], [-1.0, -2.0]]),
    )
    assert shortlist.selected_scores.grad_fn is None
    assert torch.equal(
        shortlist.candidate_mask(),
        torch.tensor(
            [
                [True, True, False],
                [False, True, True],
            ]
        ),
    )


def test_empty_and_undersized_candidate_sets_have_explicit_widths() -> None:
    shortlister = PromptTopMShortlister(max_candidates=4)

    empty = shortlister(
        torch.empty(3, 0),
        candidate_ids=(),
    )
    single = shortlister(
        torch.tensor([[0.5], [-0.5]]),
        candidate_ids=("code:0",),
    )

    assert empty.selected_indices.shape == (3, 0)
    assert empty.candidate_mask().shape == (3, 0)
    assert single.selected_indices.shape == (2, 1)
    assert torch.equal(single.selected_indices, torch.zeros(2, 1, dtype=torch.long))

    empty_router = PromptShortlistedLogicalRouter(
        compile_logical_router(AppendOnlyRouterHeads(2))
    )
    empty_scores = empty_router(
        torch.ones(3, 2, 2),
        shortlist=empty,
        layer_index=0,
    )
    empty_routes = AnchoredTop1Selection()(
        empty_scores.scores,
        expert_ids=empty_scores.expert_ids,
    )
    assert empty_scores.scores.shape == (3, 2, 0)
    assert torch.equal(
        empty_routes.selected_indices,
        torch.zeros(3, 2, dtype=torch.long),
    )


def test_shortlisted_router_scores_only_selected_prompt_columns() -> None:
    _, logical = _logical_router()
    router = PromptShortlistedLogicalRouter(logical)
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    shortlist = PromptTopMShortlister(2)(
        torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 3.0]]),
        candidate_ids=("code:0", "math:0", "science:0"),
    )

    result = router(features, shortlist=shortlist, layer_index=4)

    mask_value = torch.finfo(features.dtype).min
    complete = logical(features, layer_index=4)
    expected = torch.stack(
        (
            torch.stack(
                (
                    complete[0, :, 0],
                    complete[0, :, 1],
                    torch.full((2,), mask_value),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    complete[1, :, 0],
                    torch.full((2,), mask_value),
                    complete[1, :, 2],
                ),
                dim=-1,
            ),
        )
    )

    assert result.expert_ids == ("code:0", "math:0", "science:0")
    assert torch.equal(result.scores, expected)
    assert torch.equal(result.candidate_mask, shortlist.candidate_mask())

    selected_mask = result.candidate_mask[:, None, :].expand_as(result.scores)
    result.scores.masked_select(selected_mask).sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad != 0)


def test_unselected_overflowing_column_is_not_evaluated() -> None:
    heads = AppendOnlyRouterHeads(1)
    safe = AffineRouterHead(1, layer_index=0, expert_id="safe")
    unsafe = AffineRouterHead(1, layer_index=0, expert_id="unsafe")
    with torch.no_grad():
        safe.linear.weight.fill_(1.0)
        safe.linear.bias.zero_()
        unsafe.linear.weight.fill_(torch.finfo(torch.float32).max)
        unsafe.linear.bias.zero_()
    heads.append(safe)
    heads.append(unsafe)
    logical = compile_logical_router(heads)
    router = PromptShortlistedLogicalRouter(logical)
    shortlist = PromptTopMShortlister(1)(
        torch.tensor([[1.0, 0.0]]),
        candidate_ids=("safe", "unsafe"),
    )

    result = router(
        torch.tensor([[[2.0]]]),
        shortlist=shortlist,
        layer_index=0,
    )

    assert torch.isfinite(result.scores).all()
    assert result.scores[0, 0, 0].item() == 2.0
    assert result.scores[0, 0, 1].item() == torch.finfo(torch.float32).min


def test_existing_anchored_top1_policy_respects_shortlist_and_fallback() -> None:
    heads = AppendOnlyRouterHeads(1)
    for expert_id, weight in (("code:0", 2.0), ("math:0", -2.0)):
        head = AffineRouterHead(1, layer_index=0, expert_id=expert_id)
        with torch.no_grad():
            head.linear.weight.fill_(weight)
            head.linear.bias.zero_()
        heads.append(head)
    router = PromptShortlistedLogicalRouter(compile_logical_router(heads))
    shortlist = PromptTopMShortlister(1)(
        torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        candidate_ids=("code:0", "math:0"),
    )
    result = router(
        torch.ones(2, 3, 1),
        shortlist=shortlist,
        layer_index=0,
    )

    routes = AnchoredTop1Selection()(
        result.scores,
        expert_ids=result.expert_ids,
    )

    assert torch.equal(routes.selected_indices[0], torch.ones(3, dtype=torch.long))
    assert torch.equal(routes.selected_indices[1], torch.zeros(3, dtype=torch.long))
    assert routes.anchor_mask[1].all()


def test_wrapper_copies_source_router_without_storage_or_mutation_aliases() -> None:
    _, logical = _logical_router()
    wrapper = PromptShortlistedLogicalRouter(logical)
    shortlist = PromptTopMShortlister(1)(
        torch.tensor([[3.0, 2.0, 1.0]]),
        candidate_ids=("code:0", "math:0", "science:0"),
    )
    features = torch.tensor([[[1.0, 2.0]]])
    expected = wrapper(features, shortlist=shortlist, layer_index=4).scores.clone()
    source_pointers = {tensor.data_ptr() for tensor in logical.state_dict().values()}
    wrapper_pointers = {
        tensor.data_ptr() for tensor in wrapper.state_dict().values()
    }

    with torch.no_grad():
        logical.layers["4"].weight.add_(100.0)

    assert source_pointers.isdisjoint(wrapper_pointers)
    assert torch.equal(
        wrapper(features, shortlist=shortlist, layer_index=4).scores,
        expected,
    )
    assert tuple(wrapper.parameters()) == ()


def test_shortlisting_rejects_malformed_inputs_and_topology_mismatches() -> None:
    with pytest.raises(TypeError, match="integer"):
        PromptTopMShortlister(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        PromptTopMShortlister(0)

    shortlister = PromptTopMShortlister(1)
    with pytest.raises(PromptShortlistingError, match="two-dimensional"):
        shortlister(torch.ones(2), candidate_ids=("code:0",))
    with pytest.raises(PromptShortlistingError, match="final dimension"):
        shortlister(torch.ones(2, 2), candidate_ids=("code:0",))
    with pytest.raises(PromptShortlistingError, match="finite"):
        shortlister(
            torch.tensor([[torch.nan]]),
            candidate_ids=("code:0",),
        )
    with pytest.raises(PromptShortlistingError, match="unique"):
        shortlister(
            torch.ones(1, 2),
            candidate_ids=("code:0", "code:0"),
        )
    with pytest.raises(PromptShortlistingError, match="unique candidates"):
        PromptCandidateShortlist(
            selected_indices=torch.tensor([[0, 0]]),
            selected_scores=torch.tensor([[1.0, 1.0]]),
            candidate_ids=("code:0", "math:0"),
        )

    _, logical = _logical_router()
    router = PromptShortlistedLogicalRouter(logical)
    shortlist = shortlister(
        torch.tensor([[1.0, 0.0]]),
        candidate_ids=("code:0", "math:0"),
    )
    with pytest.raises(PromptShortlistingError, match="exactly match"):
        router(
            torch.ones(1, 2, 2),
            shortlist=shortlist,
            layer_index=4,
        )
    correct = shortlister(
        torch.tensor([[2.0, 1.0, 0.0]]),
        candidate_ids=("code:0", "math:0", "science:0"),
    )
    with pytest.raises(PromptShortlistingError, match="prompt dimension"):
        router(
            torch.ones(2, 2, 2),
            shortlist=correct,
            layer_index=4,
        )
