"""Exercise COM-004 always-on-anchor LoRA delta execution offline."""

from __future__ import annotations

import json

import torch
from torch import Tensor, nn

from anchored_moe import (
    AdaptiveTop1Top2Selection,
    AlwaysOnAnchorDeltaDispatcher,
    LoRAExpertConfig,
    PEFTLoRAExpert,
)


class NamedProjectionMLP(nn.Module):
    """Small gated MLP using the Qwen3 projection naming convention."""

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


def _set_adapter_output(expert: PEFTLoRAExpert, value: float) -> None:
    """Give zero-initialized LoRA B factors a deterministic nonzero value."""

    with torch.no_grad():
        for name in expert.adapter_parameter_names:
            if ".lora_B." in name:
                expert.expert_mlp.get_parameter(name).fill_(value)


def main() -> None:
    """Run anchor, top-1, and top-2 delta routes and print a summary."""

    torch.manual_seed(401)
    anchor = NamedProjectionMLP().eval()
    config = LoRAExpertConfig(
        rank=2,
        alpha=4,
        target_modules=("gate_proj", "up_proj", "down_proj"),
    )
    code = PEFTLoRAExpert(anchor, config=config, trainable=False)
    math = PEFTLoRAExpert(anchor, config=config, trainable=False)
    _set_adapter_output(code, 0.02)
    _set_adapter_output(math, -0.015)

    hidden_states = torch.randn(3, 4)
    scores = torch.tensor(
        [
            [-1.0, -2.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ]
    )
    result = AlwaysOnAnchorDeltaDispatcher(
        selection_policy=AdaptiveTop1Top2Selection(mix_threshold=0.25),
    ).dispatch(
        hidden_states,
        expert_scores=scores,
        expert_ids=("code", "math"),
        anchor_mlp=anchor,
        experts=(code, math),
    )

    anchor_output = anchor(hidden_states)
    assert torch.equal(result.output[0], anchor_output[0])
    assert torch.equal(result.capacity.route_arity, torch.tensor([0, 1, 2]))
    assert torch.equal(result.delta_weights[0], torch.tensor([0.0, 0.0]))
    assert torch.equal(result.delta_weights[1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.delta_weights[2].sum(), torch.tensor(1.0))
    assert all(not parameter.requires_grad for parameter in anchor.parameters())
    assert torch.isfinite(result.output).all()

    print(
        json.dumps(
            {
                "anchor_exact_rows": 1,
                "delta_weights": result.delta_weights.tolist(),
                "expert_parameterization": code.parameterization,
                "route_arity": result.capacity.route_arity.tolist(),
                "strategy": "always_on_anchor_deltas",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
