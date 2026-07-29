"""Forced-path Trainer integration for EXP-004 atomic micro-expert sets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    FineGrainedMicroExpertConfig,
    FineGrainedMicroExpertSet,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend


class TinyMicroExpertDataset(Dataset):
    """Deterministic causal rows for two atomic-set optimizer steps."""

    rows = (
        (1, 7, 11, 3),
        (2, 5, 13, 9),
    )

    def __len__(self) -> int:
        """Return the fixed row count."""

        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one causal input and copied labels."""

        tokens = torch.tensor(self.rows[index], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def test_forced_path_updates_only_atomic_micro_expert_parameters(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[3]),
    )
    wrapper = tiny_qwen3_model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    expert = FineGrainedMicroExpertSet.from_anchor(
        wrapper.anchor_mlp,
        config=FineGrainedMicroExpertConfig(segmentation_factor=4),
    )
    expert_before = {
        name: tensor.detach().clone()
        for name, tensor in expert.state_dict().items()
    }
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in wrapper.anchor_mlp.state_dict().items()
    }
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[
            ForcedExpertBinding(3, "code:micro-set", expert)
        ],
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-output"),
            max_steps=2,
            per_device_train_batch_size=1,
            learning_rate=1e-2,
            optim="adamw_torch",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            seed=367,
        ),
        train_dataset=TinyMicroExpertDataset(),
    )

    result = trainer.train()

    assert result.global_step == 2
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_parameter_ids = {id(parameter) for parameter in expert.parameters()}
    assert optimizer_parameter_ids == expected_parameter_ids
    assert trainer.forced_path_report.trainable_parameter_count == (
        expert.parameter_count
    )
    assert any(
        not torch.equal(expert.state_dict()[name], before)
        for name, before in expert_before.items()
    )
    for name, tensor in anchor_before.items():
        assert torch.equal(wrapper.anchor_mlp.state_dict()[name], tensor)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in wrapper.anchor_mlp.parameters()
    )
    assert wrapper.routing_mode == "forced_expert"
