"""Verify independent training followed by local mixture-aware refinement."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    ForcedExpertBinding,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    IndependentExpertTrainingRun,
    LocalMiniMoEExpertBinding,
    LocalMiniMoETrainingConfig,
    RoutedMLP,
    refine_independent_expert_set,
    train_independent_expert_set,
)
from anchored_moe.backends import Qwen3Backend


def _tiny_model() -> Qwen3ForCausalLM:
    """Return a deterministic model small enough for three CPU steps."""

    torch.manual_seed(149)
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


def _dataset(offset: int = 0) -> list[dict[str, torch.Tensor]]:
    """Return a caller-owned deterministic specialization partition."""

    return [
        {
            "input_ids": (tokens := torch.tensor(row) + offset),
            "attention_mask": torch.ones_like(tokens),
            "labels": tokens.clone(),
        }
        for row in ((1, 3, 5, 7), (2, 4, 6, 8))
    ]


def _arguments(path: Path, *, seed: int) -> TrainingArguments:
    """Return one-step no-checkpoint CPU arguments."""

    return TrainingArguments(
        output_dir=str(path),
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        optim="adamw_torch",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        report_to=[],
        use_cpu=True,
        dataloader_pin_memory=False,
        seed=seed,
    )


def main() -> None:
    """Train independently, refine jointly, and prove exact anchor cleanup."""

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
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    with TemporaryDirectory(prefix="anchored-moe-refinement-") as directory:
        root = Path(directory)
        independent_report = train_independent_expert_set(
            model,
            backend=backend,
            runs=tuple(
                IndependentExpertTrainingRun(
                    expert_bindings=[
                        ForcedExpertBinding(3, expert_id, expert)
                    ],
                    args=_arguments(
                        root / f"independent-{index}",
                        seed=151 + index,
                    ),
                    trainer_kwargs={"train_dataset": _dataset(index * 5)},
                )
                for index, (expert_id, expert) in enumerate(
                    zip(expert_ids, experts, strict=True)
                )
            ),
        )
        independently_trained = tuple(
            {
                name: tensor.detach().clone()
                for name, tensor in expert.state_dict().items()
            }
            for expert in experts
        )

        heads = AppendOnlyRouterHeads(16)
        for expert_id in expert_ids:
            heads.append(
                AffineRouterHead(16, layer_index=3, expert_id=expert_id)
            )
        report = refine_independent_expert_set(
            model,
            backend=backend,
            source_training=independent_report,
            expert_bindings=tuple(
                LocalMiniMoEExpertBinding(3, expert_id, expert)
                for expert_id, expert in zip(
                    expert_ids,
                    experts,
                    strict=True,
                )
            ),
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=heads,
            args=_arguments(root / "joint-refinement", seed=157),
            config=LocalMiniMoETrainingConfig(
                load_balancing_weight=0.05,
            ),
            trainer_kwargs={"train_dataset": _dataset()},
        )

    expert_refined = tuple(
        any(
            not torch.equal(tensor, before[name])
            for name, tensor in expert.state_dict().items()
        )
        for expert, before in zip(
            experts,
            independently_trained,
            strict=True,
        )
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

    assert expert_refined == (True, True)
    assert router_trained
    assert anchor_exact
    assert wrapper.routing_mode == "forced_anchor"
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for module in (*experts, heads)
        for parameter in module.parameters()
    )

    print(
        "TRN-004 mixture-aware refinement validation passed: "
        f"experts={report.expert_ids}, layers={report.layer_indices}, "
        f"total_steps={report.total_global_steps}, "
        f"expert_refined={expert_refined}, "
        f"router_trained={router_trained}, anchor_exact={anchor_exact}"
    )


if __name__ == "__main__":
    main()
