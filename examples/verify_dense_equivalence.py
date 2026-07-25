"""Verify exact forced-anchor equivalence on a tiny local Qwen3 model."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import AnchoredMoEConfig
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import assert_tensors_equivalent


def main() -> None:
    """Construct, convert, and compare a deterministic tiny Qwen3 model."""

    torch.manual_seed(7)
    base_config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    model = Qwen3ForCausalLM(base_config).eval()
    input_ids = torch.tensor([[1, 7, 11, 3]])

    with torch.no_grad():
        dense = model(input_ids=input_ids, output_hidden_states=True)

    report = Qwen3Backend().convert(
        model,
        AnchoredMoEConfig(converted_layers="upper_half"),
    )
    with torch.no_grad():
        converted = model(input_ids=input_ids, output_hidden_states=True)

    logits = assert_tensors_equivalent(
        converted.logits,
        dense.logits,
        name="logits",
    )
    for index, (actual, expected) in enumerate(
        zip(converted.hidden_states, dense.hidden_states, strict=True)
    ):
        assert_tensors_equivalent(
            actual,
            expected,
            name=f"hidden_states[{index}]",
        )

    print(
        "Dense equivalence passed: "
        f"layers={report.converted_layers}, "
        f"exact_logits={logits.exact}, "
        f"max_abs={logits.max_absolute_difference}"
    )


if __name__ == "__main__":
    main()
