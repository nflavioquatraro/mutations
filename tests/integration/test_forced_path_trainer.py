"""Real Hugging Face Trainer integration for isolated expert optimization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ExpertObjectiveConfig,
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


class TinyReplayDataset(Dataset):
    """Causal rows with precomputed immutable-anchor logits and replay masks."""

    def __init__(self, anchor: Qwen3ForCausalLM) -> None:
        """Evaluate and detach the teacher before expert-path conversion."""

        self.rows: list[dict[str, torch.Tensor]] = []
        anchor.eval()
        with torch.no_grad():
            for index, values in enumerate(TinyCausalDataset.examples):
                tokens = torch.tensor(values, dtype=torch.long)
                logits = anchor(input_ids=tokens.unsqueeze(0)).logits.squeeze(0)
                replay_mask = torch.ones(tokens.shape, dtype=torch.bool)
                if index % 2:
                    replay_mask[0] = False
                self.rows.append(
                    {
                        "input_ids": tokens,
                        "attention_mask": torch.ones_like(tokens),
                        "labels": tokens.clone(),
                        "anchor_logits": logits.detach().clone(),
                        "replay_token_mask": replay_mask,
                    }
                )

    def __len__(self) -> int:
        """Return the deterministic replay row count."""

        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one specialization/replay training row."""

        return self.rows[index]


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


def test_trainer_combines_base_replay_and_drift_losses(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    input_ids = torch.tensor([[1, 7, 11, 3]])
    with torch.no_grad():
        anchor_logits = tiny_qwen3_model(input_ids=input_ids).logits.detach()

    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[3]),
    )
    wrapper = tiny_qwen3_model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    config = ExpertObjectiveConfig(
        replay_kl_weight=0.7,
        drift_weight=0.3,
        replay_temperature=1.5,
    )
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[ForcedExpertBinding(3, "code:0", expert)],
        objective_config=config,
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
    with torch.no_grad():
        next(expert.parameters()).add_(0.25)

    loss = trainer.compute_loss(
        trainer.model,
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "anchor_logits": anchor_logits,
            "replay_token_mask": torch.tensor(
                [[True, True, False, True]]
            ),
        },
    )
    report = trainer.last_objective_report

    assert report is not None
    assert report.replay_kl_loss > 0
    assert report.drift_loss > 0
    assert report.total_loss == pytest.approx(
        report.specialization_loss
        + config.replay_kl_weight * report.replay_kl_loss
        + config.drift_weight * report.drift_loss,
        rel=1e-6,
    )
    assert loss.item() == pytest.approx(report.total_loss, rel=1e-6)


def test_trainer_retains_replay_columns_and_optimizes_only_experts(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    dataset = TinyReplayDataset(tiny_qwen3_model)
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[3]),
    )
    wrapper = tiny_qwen3_model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    frozen_before = {
        name: tensor.detach().clone()
        for name, tensor in tiny_qwen3_model.state_dict().items()
    }
    trainer = ForcedPathExpertTrainer(
        model=tiny_qwen3_model,
        backend=backend,
        expert_bindings=[ForcedExpertBinding(3, "code:0", expert)],
        objective_config=ExpertObjectiveConfig(
            replay_kl_weight=0.5,
            drift_weight=0.1,
            replay_temperature=2.0,
        ),
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
            seed=47,
        ),
        train_dataset=dataset,
    )

    result = trainer.train()
    report = trainer.last_objective_report

    assert result.global_step == 2
    assert report is not None
    assert report.replay_kl_loss > 0
    assert report.drift_loss > 0
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == {
        id(parameter) for parameter in expert.parameters()
    }
    model_after = tiny_qwen3_model.state_dict()
    for name, tensor in frozen_before.items():
        assert torch.equal(model_after[name], tensor)


def test_trainer_requires_teacher_logits_when_replay_kl_is_enabled(
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
        objective_config=ExpertObjectiveConfig(replay_kl_weight=1.0),
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-output"),
            max_steps=1,
            save_strategy="no",
            use_cpu=True,
        ),
        train_dataset=TinyCausalDataset(),
    )

    with pytest.raises(ForcedPathTrainingError, match="missing teacher logits"):
        trainer.compute_loss(
            trainer.model,
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[1, 2, 3]]),
            },
        )
