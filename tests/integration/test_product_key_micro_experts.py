"""EXP-005 integration with EXP-004 micro-experts and RTR-006 shortlists."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    FineGrainedMicroExpertConfig,
    FineGrainedMicroExpertSet,
    ProductKeyMicroExpertRetriever,
    ProductKeyRetrievalConfig,
    ProductKeyRetrievalError,
    PromptTopMShortlister,
)


def test_product_keys_bind_to_qwen3_micro_expert_count_without_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[2].mlp
    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor,
        config=FineGrainedMicroExpertConfig(segmentation_factor=4),
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in expert_set.state_dict().items()
    }
    retriever = ProductKeyMicroExpertRetriever.from_micro_expert_set(
        expert_set,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
        config=ProductKeyRetrievalConfig(top_k=2),
        expert_id_prefix="code-micro",
        trainable=False,
    )

    result = retriever(
        torch.tensor(
            [
                [[2.0, 1.0, 1.0, 0.0], [0.0, 2.0, 1.0, 1.0]],
                [[1.0, 0.0, -1.0, 2.0], [1.0, 1.0, 0.0, 1.0]],
            ]
        )
    )

    assert result.selected_indices.shape == (2, 2, 2)
    assert result.expert_ids == (
        "code-micro:0",
        "code-micro:1",
        "code-micro:2",
        "code-micro:3",
    )
    assert retriever.expert_count == expert_set.expert_count
    for name, tensor in before.items():
        assert torch.equal(expert_set.state_dict()[name], tensor)
    assert all(
        parameter.requires_grad
        for parameter in expert_set.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in retriever.parameters()
    )


def test_prompt_shortlist_can_bound_package_local_retrieval_explicitly() -> None:
    package_ids = ("code", "math")
    shortlist = PromptTopMShortlister(max_candidates=1)(
        torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        candidate_ids=package_ids,
    )
    retrievers = {
        package_id: ProductKeyMicroExpertRetriever(
            torch.eye(2),
            torch.eye(2),
            expert_ids=tuple(
                f"{package_id}-micro:{index}" for index in range(4)
            ),
            config=ProductKeyRetrievalConfig(top_k=1),
            trainable=False,
        )
        for package_id in package_ids
    }
    queries = torch.tensor(
        [[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0]]
    )

    results = {}
    for prompt_index, selected_index in enumerate(
        shortlist.selected_indices[:, 0].tolist()
    ):
        package_id = package_ids[selected_index]
        results[package_id] = retrievers[package_id](
            queries[prompt_index : prompt_index + 1]
        )

    assert set(results) == {"code", "math"}
    assert results["code"].selected_indices.item() == 0
    assert results["math"].selected_indices.item() == 3
    assert shortlist.selected_indices.tolist() == [[0], [1]]


def test_non_square_micro_expert_set_is_an_explicit_version_one_limit(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    anchor = tiny_qwen3_model.model.layers[1].mlp
    expert_set = FineGrainedMicroExpertSet.from_anchor(
        anchor,
        config=FineGrainedMicroExpertConfig(segmentation_factor=2),
    )

    with pytest.raises(ProductKeyRetrievalError, match="perfect-square"):
        ProductKeyMicroExpertRetriever.from_micro_expert_set(
            expert_set,
            torch.ones(1, 2),
            torch.ones(1, 2),
            config=ProductKeyRetrievalConfig(top_k=1),
        )
