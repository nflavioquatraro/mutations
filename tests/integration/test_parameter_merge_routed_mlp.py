"""Tiny-Qwen3 integration coverage for COM-005 routed execution."""

from __future__ import annotations

import math

import torch
from transformers import Qwen3ForCausalLM

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    LearnedMixtureWeightHead,
    LoRAExpertConfig,
    LoRAParameterSpaceMerge,
    PEFTLoRAExpert,
    RoutedMLP,
)


def test_qwen3_routed_mlp_matches_explicit_parameter_merge_and_resets(
    tiny_qwen3_model: Qwen3ForCausalLM,
) -> None:
    torch.manual_seed(641)
    anchor = tiny_qwen3_model.model.layers[2].mlp.eval()
    hidden_size = tiny_qwen3_model.config.hidden_size
    hidden_states = torch.randn(2, 3, hidden_size)
    anchor_state = {
        name: tensor.detach().clone()
        for name, tensor in anchor.state_dict().items()
    }
    config = LoRAExpertConfig(
        rank=2,
        alpha=4,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    experts = (
        PEFTLoRAExpert(anchor, config=config),
        PEFTLoRAExpert(anchor, config=config),
    )
    with torch.no_grad():
        for expert_index, expert in enumerate(experts, start=1):
            for name in expert.adapter_parameter_names:
                if ".lora_A." in name:
                    expert.expert_mlp.get_parameter(name).fill_(0.04)
                else:
                    expert.expert_mlp.get_parameter(name).fill_(
                        expert_index * 0.03
                    )

    router_heads = AppendOnlyRouterHeads(hidden_size)
    for expert_id, bias in (("code-a", 2.0), ("code-b", 1.0)):
        head = AffineRouterHead(
            hidden_size,
            layer_index=2,
            expert_id=expert_id,
            trainable=False,
        )
        with torch.no_grad():
            head.linear.weight.zero_()
            head.linear.bias.fill_(bias)
        router_heads.append(head)
    merge_head = LearnedMixtureWeightHead(
        weight=torch.zeros(2),
        bias=torch.tensor(math.log(3.0)),
    )
    merge_policy = LoRAParameterSpaceMerge(merge_head)
    wrapper = RoutedMLP(anchor)
    wrapper.configure_parameter_space_lora_merge(
        layer_index=2,
        router_feature_extractor=IdentityRouterFeatureExtractor(hidden_size),
        router_heads=router_heads,
        experts={"code-b": experts[1], "code-a": experts[0]},
        merge_policy=merge_policy,
    )

    actual = wrapper(hidden_states)
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    expected = merge_policy.merge(
        flat_hidden,
        anchor_mlp=anchor,
        experts=experts,
        weights=flat_hidden.new_tensor([0.75, 0.25]).expand(
            flat_hidden.shape[0],
            2,
        ),
    ).output.reshape_as(actual)

    assert wrapper.routing_mode == "parameter_space_lora_merge"
    assert wrapper.expert_ids == ("code-a", "code-b")
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert "_parameter_merge_head.weight" in dict(wrapper.named_buffers())
    for name, tensor in anchor.state_dict().items():
        assert torch.equal(tensor, anchor_state[name])
    assert all(not parameter.requires_grad for parameter in anchor.parameters())

    wrapper.reset_to_forced_anchor()
    assert wrapper.routing_mode == "forced_anchor"
    assert torch.equal(wrapper(hidden_states), anchor(hidden_states))
