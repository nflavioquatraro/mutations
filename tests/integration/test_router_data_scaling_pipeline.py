"""Offline RTR-004/TRN-003 composition test for EVAL-007."""

from __future__ import annotations

from pathlib import Path

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
    RouterDataScalingMetrics,
    RouterDataScalingReport,
    RouterDataScalingTrial,
    RouterFeatureBank,
    RoutedMLP,
    UtilityLabelBank,
    compile_logical_router,
    evaluate_router_data_scaling,
    recalibrate_global_router,
)
from anchored_moe.backends import Qwen3Backend


class _LocalTrainingDataset(Dataset):
    """Two deterministic causal rows for one matched partial-MoE step."""

    rows = (
        torch.tensor([1, 3, 5, 7]),
        torch.tensor([2, 4, 6, 8]),
    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.rows[index]
        return {"input_ids": tokens, "labels": tokens.clone()}


def _train_matched_partial_moe(
    model: Qwen3ForCausalLM,
    output_dir: Path,
):
    """Run the supported one-step TRN-003 baseline and return its report."""

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    experts = tuple(
        FullMLPExpert.from_anchor(wrapper.anchor_mlp) for _ in range(2)
    )
    expert_ids = ("code:0", "code:1")
    bindings = tuple(
        LocalMiniMoEExpertBinding(3, expert_id, expert)
        for expert_id, expert in zip(expert_ids, experts, strict=True)
    )
    heads = AppendOnlyRouterHeads(16)
    for expert_id in expert_ids:
        heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )
    trainer = LocalMiniMoETrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=1e-2,
            optim="adamw_torch",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            seed=71,
        ),
        backend=backend,
        expert_bindings=bindings,
        router_feature_extractor=IdentityRouterFeatureExtractor(16),
        router_heads=heads,
        train_dataset=_LocalTrainingDataset(),
    )
    trainer.train()
    assert trainer.training_report is not None
    return trainer.training_report


def _router_trial(
    source,
    sample_count: int,
) -> RouterDataScalingTrial:
    """Fit one real RTR-004 subset and attach synthetic held-out metrics."""

    positions = torch.linspace(-2.0, 2.0, sample_count)
    features_tensor = torch.zeros(sample_count, 16)
    features_tensor[:, 0] = positions
    features = RouterFeatureBank(
        {3: features_tensor},
        feature_specification={
            "feature_type": "identity",
            "input": "pre_mlp_hidden_state",
            "output_size": 16,
            "version": 1,
        },
        dataset_provenance={
            "dataset_id": "tests/eval-007-replay",
            "revision": "v1",
            "split": "train",
            "filters": [f"prefix:{sample_count}"],
        },
        row_ids=torch.arange(sample_count),
    )
    utilities = torch.stack((-positions, positions), dim=-1)
    labels = UtilityLabelBank(
        {3: utilities},
        expert_ids_by_layer={3: ("code:0", "code:1")},
        row_ids=features.row_ids,
        feature_bank_metadata_sha256=features.metadata_sha256,
        loss_name="token_nll",
        incumbent_route="anchor",
    )
    _, recalibration_report = recalibrate_global_router(
        source,
        features,
        labels,
        registry_sha256="a" * 64,
        compatibility_sha256="b" * 64,
        learning_rate=0.1,
        optimization_steps=16,
    )
    improvement = 0.02 * sample_count
    return RouterDataScalingTrial(
        router_training_sample_count=sample_count,
        router_training_row_ids=tuple(range(sample_count)),
        metrics=tuple(
            RouterDataScalingMetrics(
                stratum=stratum,
                evaluation_row_count=2,
                performance=base_loss - improvement,
                mean_router_regret=base_regret - improvement,
            )
            for stratum, base_loss, base_regret in (
                ("specialization_interior", 0.9, 0.2),
                ("boundary", 1.1, 0.4),
                ("mixed", 1.3, 0.6),
            )
        ),
        recalibration_report=recalibration_report,
    )


def _metrics(
    rows: tuple[tuple[str, float, float], ...],
) -> tuple[RouterDataScalingMetrics, ...]:
    return tuple(
        RouterDataScalingMetrics(
            stratum=stratum,
            evaluation_row_count=2,
            performance=performance,
            mean_router_regret=regret,
        )
        for stratum, performance, regret in rows
    )


def test_real_recalibration_and_local_training_reports_form_scaling_grid(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    matched_training = _train_matched_partial_moe(
        tiny_qwen3_model,
        tmp_path / "matched-partial-moe",
    )
    heads = AppendOnlyRouterHeads(16)
    for expert_id in ("code:0", "code:1"):
        heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )
    source = compile_logical_router(heads)
    source_state = {
        name: tensor.clone() for name, tensor in source.state_dict().items()
    }
    trials = (_router_trial(source, 2), _router_trial(source, 4))

    report = evaluate_router_data_scaling(
        metric_name="mean_token_nll",
        higher_is_better=False,
        evaluation_row_ids_by_stratum={
            "specialization_interior": (100, 101),
            "boundary": (102, 103),
            "mixed": (104, 105),
        },
        append_only_metrics=_metrics(
            (
                ("specialization_interior", 1.0, 0.3),
                ("boundary", 1.2, 0.5),
                ("mixed", 1.4, 0.7),
            )
        ),
        router_only_trials=trials,
        matched_partial_moe_metrics=_metrics(
            (
                ("specialization_interior", 0.7, 0.05),
                ("boundary", 0.8, 0.1),
                ("mixed", 1.0, 0.2),
            )
        ),
        matched_partial_moe_training=matched_training,
    )

    assert report.router_training_sample_counts == (2, 4)
    assert len(report.gaps) == 6
    assert report.matched_partial_moe_provenance.global_step == 1
    assert report.matched_partial_moe_provenance.layer_indices == (3,)
    assert RouterDataScalingReport.from_dict(report.to_dict()) == report
    assert all(
        torch.equal(source.state_dict()[name], tensor)
        for name, tensor in source_state.items()
    )
