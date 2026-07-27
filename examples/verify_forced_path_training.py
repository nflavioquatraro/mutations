"""Verify isolated forced-path expert training on a tiny local Qwen3 model."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ExpertObjectiveConfig,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    FullMLPExpert,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend


def _tiny_model() -> Qwen3ForCausalLM:
    """Return a deterministic model small enough for one CPU optimizer step."""

    torch.manual_seed(13)
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
    """Train with replay and drift while preserving the anchor byte-exactly."""

    model = _tiny_model()
    dataset = []
    for index, values in enumerate(
        (
            (1, 7, 11, 3),
            (2, 5, 13, 9),
            (4, 6, 8, 10),
            (3, 12, 15, 1),
        )
    ):
        tokens = torch.tensor(values, dtype=torch.long)
        with torch.no_grad():
            anchor_logits = model(
                input_ids=tokens.unsqueeze(0)
            ).logits.squeeze(0)
        replay_mask = torch.ones_like(tokens, dtype=torch.bool)
        if index % 2:
            replay_mask[0] = False
        dataset.append(
            {
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "labels": tokens.clone(),
                "anchor_logits": anchor_logits.detach().clone(),
                "replay_token_mask": replay_mask,
            }
        )

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)

    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    expert_before = {
        name: tensor.detach().clone()
        for name, tensor in expert.state_dict().items()
    }
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in wrapper.anchor_mlp.state_dict().items()
    }
    with TemporaryDirectory(prefix="anchored-moe-train-") as output_dir:
        trainer = ForcedPathExpertTrainer(
            model=model,
            backend=backend,
            expert_bindings=[
                ForcedExpertBinding(3, "code:0", expert)
            ],
            objective_config=ExpertObjectiveConfig(
                replay_kl_weight=0.5,
                drift_weight=0.1,
                replay_temperature=2.0,
            ),
            args=TrainingArguments(
                output_dir=output_dir,
                max_steps=2,
                per_device_train_batch_size=1,
                learning_rate=1e-2,
                optim="adamw_torch",
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                dataloader_pin_memory=False,
                seed=41,
            ),
            train_dataset=dataset,
        )
        result = trainer.train()

    expert_changed = any(
        not torch.equal(tensor, expert_before[name])
        for name, tensor in expert.state_dict().items()
    )
    anchor_exact = all(
        torch.equal(tensor, anchor_before[name])
        for name, tensor in wrapper.anchor_mlp.state_dict().items()
    )
    assert expert_changed
    assert anchor_exact
    assert trainer.last_objective_report is not None
    assert trainer.last_objective_report.replay_kl_loss > 0
    assert trainer.last_objective_report.drift_loss > 0

    print(
        "TRN-001/TRN-005 forced-path validation passed: "
        f"steps={result.global_step}, "
        f"layers={trainer.forced_path_report.layer_indices}, "
        f"trainable={trainer.forced_path_report.trainable_parameter_count}, "
        f"expert_changed={expert_changed}, anchor_exact={anchor_exact}, "
        f"objectives={trainer.last_objective_report}"
    )


if __name__ == "__main__":
    main()
