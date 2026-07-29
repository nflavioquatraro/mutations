"""Verify RTR-006 prompt top-M retrieval and sparse logical scoring."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchoredTop1Selection,
    AppendOnlyRouterHeads,
    PromptShortlistedLogicalRouter,
    PromptTopMShortlister,
    compile_logical_router,
)


def main() -> None:
    """Shortlist one expert per prompt and reuse anchored token top-1."""

    heads = AppendOnlyRouterHeads(feature_size=2)
    for expert_id, weight in (
        ("code:0", (1.0, 0.0)),
        ("math:0", (0.0, 1.0)),
        ("science:0", (-1.0, -1.0)),
    ):
        head = AffineRouterHead(
            2,
            layer_index=0,
            expert_id=expert_id,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([weight]))
        heads.append(head)

    candidate_ids = ("code:0", "math:0", "science:0")
    shortlist = PromptTopMShortlister(max_candidates=1)(
        torch.tensor(
            [
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        candidate_ids=candidate_ids,
    )
    features = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    shortlisted_router = PromptShortlistedLogicalRouter(
        compile_logical_router(heads)
    )
    result = shortlisted_router(
        features,
        shortlist=shortlist,
        layer_index=0,
    )
    routes = AnchoredTop1Selection()(
        result.scores,
        expert_ids=result.expert_ids,
    )

    assert shortlist.selected_indices.tolist() == [[0], [1], [2]]
    assert routes.selected_indices.tolist() == [[1, 1], [2, 2], [0, 0]]
    assert tuple(shortlisted_router.parameters()) == ()
    print(
        "RTR-006 validation passed: "
        f"shortlists={shortlist.selected_indices.tolist()}, "
        f"routes={routes.selected_indices.tolist()}"
    )


if __name__ == "__main__":
    main()
