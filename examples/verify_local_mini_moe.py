"""Verify joint two-expert local mini-MoE training on a tiny Qwen3 model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    LocalMiniMoEExpertBinding,
    LocalMiniMoETrainer,
    LocalMiniMoETrainingConfig,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend


def _tiny_model() -> Qwen3ForCausalLM:
    """Return a deterministic model small enough for two CPU steps."""

    torch.manual_seed(83)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
    ).eval()


def main() -> None:
    """Train a local pair and prove joint updates plus exact cleanup."""

    model = _tiny_model()
    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)

    expert_ids = ("code:0", "code:1")
    experts = tuple(
        FullMLPExpert.from_anchor(wrapper.anchor_mlp)
        for _ in expert_ids
    )
    bindings = tuple(
        LocalMiniMoEExpertBinding(3, expert_id, expert)
        for expert_id, expert in zip(expert_ids, experts, strict=True)
    )
    heads = AppendOnlyRouterHeads(16)
    for expert_id in expert_ids:
        heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )
    expert_before = tuple(
        {
            name: tensor.detach().clone()
            for name, tensor in expert.state_dict().items()
        }
        for expert in experts
    )
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    dataset = [
        {
            "input_ids": (tokens := torch.tensor(row)),
            "labels": tokens.clone(),
        }
        for row in ((1, 3, 5, 7), (2, 4, 6, 8))
    ]

    with TemporaryDirectory(prefix="anchored-moe-local-mini-") as directory:
        trainer = LocalMiniMoETrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(Path(directory) / "run"),
                max_steps=2,
                per_device_train_batch_size=1,
                learning_rate=1e-2,
                optim="adamw_torch",
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                dataloader_pin_memory=False,
                seed=89,
            ),
            backend=backend,
            expert_bindings=bindings,
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=heads,
            config=LocalMiniMoETrainingConfig(),
            train_dataset=dataset,
        )
        trainer.train()

    expert_changed = tuple(
        any(
            not torch.equal(tensor, initial[name])
            for name, tensor in expert.state_dict().items()
        )
        for expert, initial in zip(experts, expert_before, strict=True)
    )
    model_after = model.state_dict()
    anchor_exact = (
        model_after.keys() == anchor_before.keys()
        and all(
            torch.equal(model_after[name], tensor)
            for name, tensor in anchor_before.items()
        )
    )
    router_trained = any(
        torch.count_nonzero(head.linear.weight.detach()).item()
        for head in heads.heads
    )
    assert expert_changed == (True, True)
    assert router_trained
    assert anchor_exact
    assert wrapper.routing_mode == "forced_anchor"
    assert trainer.training_report is not None

    print(
        "TRN-003 local mini-MoE validation passed: "
        f"experts={trainer.training_report.topology.expert_ids}, "
        f"steps={trainer.training_report.global_step}, "
        f"expert_changed={expert_changed}, router_trained={router_trained}, "
        f"anchor_exact={anchor_exact}"
    )


if __name__ == "__main__":
    main()
