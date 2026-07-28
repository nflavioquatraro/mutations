"""End-to-end TRN-002 to TRN-004 refinement acceptance test."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Qwen3ForCausalLM, TrainingArguments

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


class RefinementDataset(Dataset):
    """Small deterministic specialization partition for CPU acceptance."""

    rows = (
        torch.tensor([1, 3, 5, 7]),
        torch.tensor([2, 4, 6, 8]),
    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.rows[index]
        return {
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens),
            "labels": tokens.clone(),
        }


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
        report_to=[],
        use_cpu=True,
        dataloader_pin_memory=False,
        seed=seed,
    )


def test_independent_pair_is_jointly_refined_and_anchor_stays_exact(
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
    expert_ids = ("code:0", "code:1")
    experts = tuple(
        FullMLPExpert.from_anchor(wrapper.anchor_mlp)
        for _ in expert_ids
    )
    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in tiny_qwen3_model.state_dict().items()
    }
    independent_runs = tuple(
        IndependentExpertTrainingRun(
            expert_bindings=[
                ForcedExpertBinding(3, expert_id, expert)
            ],
            args=_arguments(
                tmp_path / f"independent-{index}",
                seed=131 + index,
            ),
            trainer_kwargs={"train_dataset": RefinementDataset()},
        )
        for index, (expert_id, expert) in enumerate(
            zip(expert_ids, experts, strict=True)
        )
    )
    independent_report = train_independent_expert_set(
        tiny_qwen3_model,
        backend=backend,
        runs=independent_runs,
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
        tiny_qwen3_model,
        backend=backend,
        source_training=independent_report,
        expert_bindings=tuple(
            LocalMiniMoEExpertBinding(3, expert_id, expert)
            for expert_id, expert in zip(expert_ids, experts, strict=True)
        ),
        router_feature_extractor=IdentityRouterFeatureExtractor(16),
        router_heads=heads,
        args=_arguments(tmp_path / "refinement", seed=137),
        config=LocalMiniMoETrainingConfig(load_balancing_weight=0.05),
        trainer_kwargs={"train_dataset": RefinementDataset()},
    )

    refinement_changed = tuple(
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
    model_after = tiny_qwen3_model.state_dict()
    anchor_exact = (
        model_after.keys() == anchor_before.keys()
        and all(
            torch.equal(model_after[name], tensor)
            for name, tensor in anchor_before.items()
        )
    )

    assert report.source_training is independent_report
    assert report.expert_ids == expert_ids
    assert report.layer_indices == (3,)
    assert report.total_global_steps == 3
    assert refinement_changed == (True, True)
    assert any(
        torch.count_nonzero(head.linear.weight.detach()).item()
        for head in heads.heads
    )
    assert report.refinement.final_objectives.load_balancing_loss > 0
    assert anchor_exact
    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for module in (*experts, heads)
        for parameter in module.parameters()
    )
