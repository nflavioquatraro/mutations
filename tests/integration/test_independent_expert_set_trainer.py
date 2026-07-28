"""Hugging Face integration tests for independent latent expert sets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ForcedExpertBinding,
    FullMLPExpert,
    IndependentExpertTrainingRun,
    RoutedMLP,
    train_independent_expert_set,
)
from anchored_moe.backends import Qwen3Backend


class PartitionDataset(Dataset):
    """Small deterministic causal partition owned by one expert run."""

    def __init__(self, offset: int) -> None:
        """Create two distinct token rows inside the tiny fixture vocabulary."""

        self.rows = (
            torch.tensor([1 + offset, 3 + offset, 5 + offset, 7 + offset]),
            torch.tensor([2 + offset, 4 + offset, 6 + offset, 8 + offset]),
        )

    def __len__(self) -> int:
        """Return the deterministic row count."""

        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one causal input and an independent label tensor."""

        tokens = self.rows[index]
        return {"input_ids": tokens, "labels": tokens.clone()}


class FailingDataset(Dataset):
    """Dataset that raises when Trainer requests its only row."""

    def __len__(self) -> int:
        """Expose one row so Trainer enters data loading."""

        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Raise a stable error used to verify failure-safe cleanup."""

        raise RuntimeError("synthetic partition failure")


def _arguments(path: Path, *, seed: int) -> TrainingArguments:
    """Return deterministic one-step CPU Trainer arguments."""

    return TrainingArguments(
        output_dir=str(path),
        max_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        optim="adamw_torch",
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        dataloader_pin_memory=False,
        seed=seed,
    )


def _run(
    wrapper: RoutedMLP,
    *,
    expert_id: str,
    dataset: Dataset,
    output_dir: Path,
    seed: int,
) -> tuple[IndependentExpertTrainingRun, FullMLPExpert]:
    """Create one independently owned expert run for converted layer three."""

    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    return (
        IndependentExpertTrainingRun(
            expert_bindings=[ForcedExpertBinding(3, expert_id, expert)],
            args=_arguments(output_dir, seed=seed),
            trainer_kwargs={"train_dataset": dataset},
        ),
        expert,
    )


def test_two_experts_train_independently_and_restore_exact_anchor(
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
    code_run, code_expert = _run(
        wrapper,
        expert_id="code:0",
        dataset=PartitionDataset(0),
        output_dir=tmp_path / "code",
        seed=41,
    )
    code_one_run, code_one_expert = _run(
        wrapper,
        expert_id="code:1",
        dataset=PartitionDataset(9),
        output_dir=tmp_path / "code-1",
        seed=43,
    )
    expert_before = {
        "code:0": {
            name: tensor.detach().clone()
            for name, tensor in code_expert.state_dict().items()
        },
        "code:1": {
            name: tensor.detach().clone()
            for name, tensor in code_one_expert.state_dict().items()
        },
    }
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in tiny_qwen3_model.state_dict().items()
    }

    report = train_independent_expert_set(
        tiny_qwen3_model,
        backend=backend,
        runs=[code_run, code_one_run],
    )

    assert report.expert_ids == ("code:0", "code:1")
    assert report.layer_indices == (3,)
    assert report.total_global_steps == 2
    assert tuple(item.global_step for item in report.reports) == (1, 1)
    assert all(item.training_loss > 0 for item in report.reports)
    with pytest.raises(FrozenInstanceError):
        report.reports = ()  # type: ignore[misc]

    for expert_id, expert in (
        ("code:0", code_expert),
        ("code:1", code_one_expert),
    ):
        assert any(
            not torch.equal(tensor, expert_before[expert_id][name])
            for name, tensor in expert.state_dict().items()
        )
        assert all(not parameter.requires_grad for parameter in expert.parameters())
        assert all(parameter.grad is None for parameter in expert.parameters())

    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
    assert len(wrapper.experts) == 0
    assert all(
        not parameter.requires_grad
        for parameter in tiny_qwen3_model.parameters()
    )
    model_after = tiny_qwen3_model.state_dict()
    assert model_after.keys() == anchor_before.keys()
    for name, tensor in anchor_before.items():
        assert torch.equal(model_after[name], tensor)


def test_second_run_failure_restores_anchor_and_freezes_every_expert(
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
    code_run, code_expert = _run(
        wrapper,
        expert_id="code:0",
        dataset=PartitionDataset(0),
        output_dir=tmp_path / "code",
        seed=47,
    )
    broken_run, broken_expert = _run(
        wrapper,
        expert_id="broken:0",
        dataset=FailingDataset(),
        output_dir=tmp_path / "broken",
        seed=53,
    )

    with pytest.raises(RuntimeError, match="synthetic partition failure"):
        train_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            runs=[code_run, broken_run],
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
    assert len(wrapper.experts) == 0
    assert all(
        not parameter.requires_grad
        for parameter in tiny_qwen3_model.parameters()
    )
    for expert in (code_expert, broken_expert):
        assert all(not parameter.requires_grad for parameter in expert.parameters())
        assert all(parameter.grad is None for parameter in expert.parameters())
