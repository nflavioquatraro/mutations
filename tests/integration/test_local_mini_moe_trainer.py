"""Hugging Face Trainer acceptance tests for TRN-003."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

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


class LocalMixtureDataset(Dataset):
    """Two deterministic causal rows used for two optimizer steps."""

    rows = (
        torch.tensor([1, 3, 5, 7]),
        torch.tensor([2, 4, 6, 8]),
    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.rows[index]
        return {"input_ids": tokens, "labels": tokens.clone()}


class FailingLocalDataset(Dataset):
    """Raise after Trainer enters its data-loading boundary."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise RuntimeError("synthetic local mixture failure")


def _arguments(path: Path, *, max_steps: int = 2) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(path),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        optim="adamw_torch",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        dataloader_pin_memory=False,
        seed=61,
    )


def _trainer(
    model: Qwen3ForCausalLM,
    *,
    output_dir: Path,
    dataset: Dataset,
    max_steps: int = 2,
    config: LocalMiniMoETrainingConfig | None = None,
) -> tuple[
    LocalMiniMoETrainer,
    RoutedMLP,
    tuple[FullMLPExpert, FullMLPExpert],
    AppendOnlyRouterHeads,
]:
    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    experts = (
        FullMLPExpert.from_anchor(wrapper.anchor_mlp),
        FullMLPExpert.from_anchor(wrapper.anchor_mlp),
    )
    bindings = tuple(
        LocalMiniMoEExpertBinding(3, expert_id, expert)
        for expert_id, expert in zip(
            ("code:0", "code:1"),
            experts,
            strict=True,
        )
    )
    heads = AppendOnlyRouterHeads(16)
    for expert_id in ("code:0", "code:1"):
        heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )
    trainer = LocalMiniMoETrainer(
        model=model,
        args=_arguments(output_dir, max_steps=max_steps),
        backend=backend,
        expert_bindings=bindings,
        router_feature_extractor=IdentityRouterFeatureExtractor(16),
        router_heads=heads,
        config=config,
        train_dataset=dataset,
    )
    return trainer, wrapper, experts, heads


def test_joint_training_updates_both_experts_and_restores_exact_anchor(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    trainer, wrapper, experts, heads = _trainer(
        tiny_qwen3_model,
        output_dir=tmp_path / "local",
        dataset=LocalMixtureDataset(),
    )
    # The local runtime temporarily registers experts and router heads beneath
    # the converted wrapper. Excluding those package-owned paths yields the
    # complete immutable converted-anchor state expected after cleanup.
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in tiny_qwen3_model.state_dict().items()
        if ".experts." not in name and ".router_heads." not in name
    }
    expert_before = tuple(
        {
            name: tensor.detach().clone()
            for name, tensor in expert.state_dict().items()
        }
        for expert in experts
    )

    output = trainer.train()

    assert output.global_step == 2
    assert trainer.training_report is not None
    assert trainer.training_report.global_step == 2
    assert trainer.training_report.training_loss > 0
    assert trainer.training_report.final_objectives.load_balancing_loss == 0
    assert trainer.last_objective_report is not None
    assert trainer.last_objective_report.load_balancing_loss == 0
    with pytest.raises(FrozenInstanceError):
        trainer.training_report.global_step = 3  # type: ignore[misc]
    for expert, initial in zip(experts, expert_before, strict=True):
        assert any(
            not torch.equal(tensor, initial[name])
            for name, tensor in expert.state_dict().items()
        )
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in expert.parameters()
        )
    assert any(
        torch.count_nonzero(head.linear.weight.detach()).item()
        for head in heads.heads
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in heads.parameters()
    )
    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
    assert len(wrapper.experts) == 0
    model_after = tiny_qwen3_model.state_dict()
    assert model_after.keys() == anchor_before.keys()
    for name, tensor in anchor_before.items():
        assert torch.equal(model_after[name], tensor)


def test_enabled_balancing_reports_weighted_loss_without_changing_ownership(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    trainer, wrapper, experts, heads = _trainer(
        tiny_qwen3_model,
        output_dir=tmp_path / "balanced",
        dataset=LocalMixtureDataset(),
        max_steps=1,
        config=LocalMiniMoETrainingConfig(load_balancing_weight=0.1),
    )
    expected_trainable_ids = {
        id(parameter)
        for module in [*experts, heads]
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    actual_trainable_ids = {
        id(parameter)
        for parameter in tiny_qwen3_model.parameters()
        if parameter.requires_grad
    }
    assert actual_trainable_ids == expected_trainable_ids
    assert wrapper._capture_local_mixture_weights

    trainer.train()

    report = trainer.training_report
    assert report is not None
    assert report.final_objectives.load_balancing_loss > 0
    assert report.final_objectives.total_loss == pytest.approx(
        report.final_objectives.specialization_loss
        + 0.1 * report.final_objectives.load_balancing_loss
    )
    assert wrapper.routing_mode == "forced_anchor"
    assert not wrapper._capture_local_mixture_weights
    assert wrapper._local_mixture_weights is None


def test_training_failure_restores_anchor_and_freezes_artifacts(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    trainer, wrapper, experts, heads = _trainer(
        tiny_qwen3_model,
        output_dir=tmp_path / "broken",
        dataset=FailingLocalDataset(),
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="synthetic local mixture failure"):
        trainer.train()

    assert trainer.training_report is None
    assert wrapper.routing_mode == "forced_anchor"
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for expert in experts
        for parameter in expert.parameters()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in heads.parameters()
    )
