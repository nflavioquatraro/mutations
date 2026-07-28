"""Forced-path Trainer integration for adapter-only EXP-002 optimization."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ExpertObjectiveConfig,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    LoRAExpertConfig,
    PEFTLoRAExpert,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend


class TinyLoRADataset(Dataset):
    """Deterministic causal rows for one adapter-only optimizer step."""

    rows = (
        (1, 7, 11, 3),
        (2, 5, 13, 9),
    )

    def __len__(self) -> int:
        """Return the deterministic row count."""

        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one causal input and its copied labels."""

        tokens = torch.tensor(self.rows[index], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def test_forced_path_trainer_updates_only_lora_adapter_parameters(
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
    expert = PEFTLoRAExpert.from_anchor(
        wrapper.anchor_mlp,
        config=LoRAExpertConfig(
            rank=2,
            alpha=4,
            target_modules=("gate_proj", "up_proj", "down_proj"),
        ),
    )
    private_base_before = {
        name: tensor.detach().clone()
        for name, tensor in expert.expert_mlp.state_dict().items()
        if ".base_layer." in name
    }
    adapter_before = {
        name: expert.expert_mlp.get_parameter(name).detach().clone()
        for name in expert.adapter_parameter_names
    }
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in wrapper.anchor_mlp.state_dict().items()
    }
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[ForcedExpertBinding(3, "code:lora", expert)],
        objective_config=ExpertObjectiveConfig(drift_weight=0.1),
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
            seed=229,
        ),
        train_dataset=TinyLoRADataset(),
    )

    result = trainer.train()

    assert result.global_step == 2
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_parameter_ids = {
        id(expert.expert_mlp.get_parameter(name))
        for name in expert.adapter_parameter_names
    }
    assert optimizer_parameter_ids == expected_parameter_ids
    assert trainer.forced_path_report.trainable_parameter_count == (
        expert.adapter_parameter_count
    )
    assert trainer.last_objective_report is not None
    assert trainer.last_objective_report.drift_loss > 0
    assert any(
        not torch.equal(
            expert.expert_mlp.get_parameter(name),
            adapter_before[name],
        )
        for name in expert.adapter_parameter_names
    )
    for name, tensor in private_base_before.items():
        assert torch.equal(expert.expert_mlp.state_dict()[name], tensor)
    for name, tensor in anchor_before.items():
        assert torch.equal(wrapper.anchor_mlp.state_dict()[name], tensor)
    assert all(
        not parameter.requires_grad
        for name, parameter in expert.expert_mlp.named_parameters()
        if name not in expert.adapter_parameter_names
    )
