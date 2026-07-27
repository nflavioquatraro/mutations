"""Integration of affine router scores with top-2 selection and combination."""

from __future__ import annotations

import torch

from anchored_moe import (
    AffineRouterHead,
    AnchorVetoTop2Selection,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    ScoreSoftmaxCombination,
)


def test_affine_scores_select_and_mix_two_positive_specialists() -> None:
    hidden_states = torch.tensor([[[1.0, 2.0], [1.0, -1.0]]])
    features = IdentityRouterFeatureExtractor(2)(hidden_states)
    router = AppendOnlyRouterHeads(feature_size=2)
    code = AffineRouterHead(2, layer_index=1, expert_id="code")
    math = AffineRouterHead(2, layer_index=1, expert_id="math")
    with torch.no_grad():
        code.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        math.linear.weight.copy_(torch.tensor([[0.0, 1.0]]))
    router.append(code)
    router.append(math)

    decision = AnchorVetoTop2Selection()(
        router(features, layer_index=1),
        expert_ids=router.expert_ids_for_layer(1),
    )

    assert decision.expert_ids == ("code", "math")
    assert torch.equal(
        decision.selected_indices,
        torch.tensor([[[2, 1], [1, 0]]]),
    )
    assert torch.equal(
        decision.expert_pair_mask,
        torch.tensor([[True, False]]),
    )

    # Outputs are already ordered to match the approved (math, code) pair.
    selected_outputs = torch.tensor([[[[8.0, 12.0], [2.0, 4.0]]]])
    mixed = ScoreSoftmaxCombination()(
        selected_outputs,
        selected_scores=decision.selected_scores[decision.expert_pair_mask].reshape(
            1, 1, 2
        ),
    )
    expected_weights = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
    assert torch.allclose(mixed.weights[0, 0], expected_weights)
    assert torch.allclose(
        mixed.output[0, 0],
        expected_weights[0] * selected_outputs[0, 0, 0]
        + expected_weights[1] * selected_outputs[0, 0, 1],
    )
