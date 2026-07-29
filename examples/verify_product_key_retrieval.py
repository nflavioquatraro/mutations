"""Verify EXP-005 exact product-key retrieval without micro-expert execution."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import (
    FineGrainedMicroExpertConfig,
    FineGrainedMicroExpertSet,
    ProductKeyMicroExpertRetriever,
    ProductKeyRetrievalConfig,
    PromptTopMShortlister,
)


def main() -> None:
    """Exercise exact search, stable ties, gradients, and shortlist composition."""

    torch.manual_seed(397)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
    ).eval()
    anchor_mlp = model.model.layers[-1].mlp
    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor_mlp,
        config=FineGrainedMicroExpertConfig(segmentation_factor=4),
    )
    expert_state = {
        name: value.detach().clone()
        for name, value in expert_set.state_dict().items()
    }
    first_subkeys = torch.randn(2, 3)
    second_subkeys = torch.randn(2, 3)
    retriever = ProductKeyMicroExpertRetriever.from_micro_expert_set(
        expert_set,
        first_subkeys,
        second_subkeys,
        config=ProductKeyRetrievalConfig(top_k=3),
        expert_id_prefix="code-micro",
    )
    queries = torch.randn(2, 4, 6, requires_grad=True)

    result = retriever(queries)
    first_query, second_query = queries.split(3, dim=-1)
    exhaustive_scores = (
        (first_query @ first_subkeys.T).unsqueeze(-1)
        + (second_query @ second_subkeys.T).unsqueeze(-2)
    ).flatten(start_dim=-2)
    exhaustive_indices = torch.argsort(
        exhaustive_scores,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :3]
    exhaustive_selected_scores = torch.gather(
        exhaustive_scores,
        dim=-1,
        index=exhaustive_indices,
    )
    assert torch.equal(result.selected_indices, exhaustive_indices)
    assert torch.equal(result.selected_scores, exhaustive_selected_scores)

    result.selected_scores.sum().backward()
    assert queries.grad is not None
    assert retriever.first_subkeys.grad is not None
    assert retriever.second_subkeys.grad is not None
    assert all(
        parameter.grad is None
        for parameter in expert_set.parameters()
    )
    for name, value in expert_state.items():
        assert torch.equal(expert_set.state_dict()[name], value)

    tied = ProductKeyMicroExpertRetriever(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        expert_ids=("tie:0", "tie:1", "tie:2", "tie:3"),
        config=ProductKeyRetrievalConfig(top_k=3),
        trainable=False,
    )(torch.ones(1, 4))
    assert tied.selected_indices.tolist() == [[0, 1, 2]]

    # RTR-006 remains an explicit outer boundary: callers shortlist packages
    # first and invoke only the selected package-local product-key retriever.
    shortlist = PromptTopMShortlister(max_candidates=1)(
        torch.tensor([[2.0, 1.0]]),
        candidate_ids=("code", "math"),
    )
    selected_package = shortlist.candidate_ids[
        shortlist.selected_indices.item()
    ]
    assert selected_package == "code"

    print(
        "Product-key retrieval validation passed: "
        f"experts={retriever.expert_count}, "
        f"top_k={result.top_k}, "
        f"exact_exhaustive={torch.equal(result.selected_indices, exhaustive_indices)}, "
        f"stable_ties={tied.selected_indices.tolist()[0]}, "
        f"shortlisted_package={selected_package}"
    )


if __name__ == "__main__":
    main()
