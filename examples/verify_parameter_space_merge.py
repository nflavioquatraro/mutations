"""Exercise COM-005 pairwise effective LoRA parameter merging offline."""

from __future__ import annotations

import json
import math

import torch
from torch import Tensor, nn

from anchored_moe import (
    AnchoredLoRAParameterMergeDispatcher,
    LearnedMixtureWeightHead,
    LoRAExpertConfig,
    LoRAParameterSpaceMerge,
    PEFTLoRAExpert,
)


class NamedProjectionMLP(nn.Module):
    """Small nonlinear MLP using Qwen3-compatible projection names."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the gated projection."""

        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _set_adapter(expert: PEFTLoRAExpert, value: float) -> None:
    """Set deterministic non-zero factors for an auditable merged result."""

    with torch.no_grad():
        for name in expert.adapter_parameter_names:
            expert.expert_mlp.get_parameter(name).fill_(value)


def main() -> None:
    """Run anchor, degraded top-1, and learned top-2 parameter routes."""

    torch.manual_seed(653)
    anchor = NamedProjectionMLP().eval()
    config = LoRAExpertConfig(
        rank=2,
        alpha=4,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    first = PEFTLoRAExpert(anchor, config=config, trainable=False)
    second = PEFTLoRAExpert(anchor, config=config, trainable=False)
    _set_adapter(first, 0.04)
    _set_adapter(second, 0.07)
    merge_policy = LoRAParameterSpaceMerge(
        LearnedMixtureWeightHead(
            weight=torch.zeros(2),
            bias=torch.tensor(math.log(3.0)),
        )
    )
    hidden_states = torch.randn(3, 4)
    result = AnchoredLoRAParameterMergeDispatcher(
        merge_policy=merge_policy
    ).dispatch(
        hidden_states,
        expert_scores=torch.tensor(
            [[-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]]
        ),
        expert_ids=("code:0", "code:1"),
        anchor_mlp=anchor,
        experts=(first, second),
        expert_availability=torch.tensor(
            [[True, True], [True, False], [True, True]]
        ),
    )

    assert torch.equal(result.output[0], anchor(hidden_states[0:1])[0])
    assert torch.equal(result.capacity.route_arity, torch.tensor([0, 1, 2]))
    assert torch.allclose(
        result.merge_weights,
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.75, 0.25]]),
    )
    assert all(not parameter.requires_grad for parameter in anchor.parameters())
    print(
        json.dumps(
            {
                "merge_weights": result.merge_weights.tolist(),
                "route_arity": result.capacity.route_arity.tolist(),
                "strategy": merge_policy.policy_name,
                "target_modules": list(config.target_modules),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
