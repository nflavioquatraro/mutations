"""Real Hugging Face Trainer integration for isolated expert optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    ForcedPathTrainingError,
    FullMLPExpert,
    RoutedMLP,
)
from anchored_moe.backends import Qwen3Backend


class TinyCausalDataset(Dataset):
    """Deterministic token sequences with next-token labels."""

    examples = (
        (1, 7, 11, 3),
        (2, 5, 13, 9),
        (4, 6, 8, 10),
        (3, 12, 15, 1),
    )

    def __len__(self) -> int:
        """Return the number of deterministic training rows."""

        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one input/label pair for causal language modeling."""

        tokens = torch.tensor(self.examples[index], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def test_forced_path_trainer_updates_expert_and_preserves_anchor(
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
    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    expert_before = {
        name: tensor.detach().clone()
        for name, tensor in expert.state_dict().items()
    }
    frozen_model_before = {
        name: tensor.detach().clone()
        for name, tensor in tiny_qwen3_model.state_dict().items()
    }
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[ForcedExpertBinding(3, "code:0", expert)],
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-output"),
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-2,
            optim="adamw_torch",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            seed=41,
        ),
        train_dataset=TinyCausalDataset(),
    )

    result = trainer.train()

    assert result.global_step == 2
    assert trainer.forced_path_report.layer_indices == (3,)
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {
        id(parameter) for parameter in expert.parameters()
    }
    assert any(
        not torch.equal(tensor, expert_before[name])
        for name, tensor in expert.state_dict().items()
    )
    model_after = tiny_qwen3_model.state_dict()
    for name, tensor in frozen_model_before.items():
        assert torch.equal(model_after[name], tensor)
    assert all(
        not parameter.requires_grad
        for parameter in wrapper.anchor_mlp.parameters()
    )


def test_trainer_rechecks_trainable_parameters_before_each_run(
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
    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[ForcedExpertBinding(3, "code:0", expert)],
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-output"),
            max_steps=1,
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=TinyCausalDataset(),
    )
    next(wrapper.anchor_mlp.parameters()).requires_grad_(True)

    with pytest.raises(ForcedPathTrainingError, match="invariants changed"):
        trainer.train()


def test_trainer_rejects_full_model_checkpoint_saving(
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
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[
            ForcedExpertBinding(
                3,
                "code:0",
                FullMLPExpert.from_anchor(wrapper.anchor_mlp),
            )
        ],
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-output"),
            save_strategy="no",
            use_cpu=True,
        ),
        train_dataset=TinyCausalDataset(),
    )

    with pytest.raises(ForcedPathTrainingError, match="does not save"):
        trainer.save_model()
