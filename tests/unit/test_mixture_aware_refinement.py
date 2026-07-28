"""Topology, provenance, and report tests for TRN-004."""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AffineRouterHead,
    AnchoredMoEConfig,
    AppendOnlyRouterHeads,
    ForcedPathTrainingReport,
    FullMLPExpert,
    IdentityRouterFeatureExtractor,
    IndependentExpertSetTrainingReport,
    IndependentExpertTrainingReport,
    LocalMiniMoEExpertBinding,
    LocalMiniMoEObjectiveReport,
    LocalMiniMoETrainingConfig,
    LocalMiniMoETrainingReport,
    LocalMiniMoETrainingTopology,
    MixtureAwareRefinementError,
    MixtureAwareRefinementReport,
    RoutedMLP,
    refine_independent_expert_set,
)
from anchored_moe.backends import Qwen3Backend


def _source_report(
    *,
    expert_ids: tuple[str, ...] = ("code:0", "code:1"),
    layer_indices: tuple[int, ...] = (3,),
) -> IndependentExpertSetTrainingReport:
    """Build valid immutable TRN-002 provenance without running a Trainer."""

    return IndependentExpertSetTrainingReport(
        reports=tuple(
            IndependentExpertTrainingReport(
                expert_id=expert_id,
                forced_path=ForcedPathTrainingReport(
                    layer_indices=layer_indices,
                    expert_id=expert_id,
                    trainable_parameter_names=(
                        f"experts.{expert_id}.weight",
                    ),
                    total_parameter_count=10,
                    trainable_parameter_count=2,
                ),
                global_step=1,
                training_loss=1.0,
                objective=None,
            )
            for expert_id in expert_ids
        )
    )


def _local_report(
    *,
    expert_ids: tuple[str, str] = ("code:0", "code:1"),
    layer_indices: tuple[int, ...] = (3,),
) -> LocalMiniMoETrainingReport:
    """Build one compatible immutable TRN-003 report."""

    return LocalMiniMoETrainingReport(
        topology=LocalMiniMoETrainingTopology(
            layer_indices=layer_indices,
            expert_ids=expert_ids,
            router_head_keys=tuple(
                (layer_index, expert_id)
                for layer_index in layer_indices
                for expert_id in expert_ids
            ),
            trainable_parameter_names=("experts.weight", "router.weight"),
            total_parameter_count=10,
            trainable_parameter_count=4,
            config=LocalMiniMoETrainingConfig(),
        ),
        global_step=2,
        training_loss=0.75,
        final_objectives=LocalMiniMoEObjectiveReport(
            specialization_loss=0.75,
            load_balancing_loss=0.0,
            total_loss=0.75,
        ),
    )


def _components(
    model: Qwen3ForCausalLM,
) -> tuple[
    Qwen3Backend,
    RoutedMLP,
    tuple[LocalMiniMoEExpertBinding, ...],
    AppendOnlyRouterHeads,
]:
    """Build one local pair aligned with the default source report."""

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)
    expert_ids = ("code:0", "code:1")
    bindings = tuple(
        LocalMiniMoEExpertBinding(
            3,
            expert_id,
            FullMLPExpert.from_anchor(wrapper.anchor_mlp),
        )
        for expert_id in expert_ids
    )
    heads = AppendOnlyRouterHeads(16)
    for expert_id in expert_ids:
        heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )
    return backend, wrapper, bindings, heads


def _arguments(path: Path) -> TrainingArguments:
    """Return valid no-checkpoint arguments for preflight tests."""

    return TrainingArguments(
        output_dir=str(path),
        max_steps=1,
        per_device_train_batch_size=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )


def test_report_links_exact_topology_and_total_steps() -> None:
    source = _source_report()
    refinement = _local_report()

    report = MixtureAwareRefinementReport(source, refinement)

    assert report.expert_ids == ("code:0", "code:1")
    assert report.layer_indices == (3,)
    assert report.total_global_steps == 4
    with pytest.raises(
        MixtureAwareRefinementError,
        match="expert order",
    ):
        MixtureAwareRefinementReport(
            source,
            _local_report(expert_ids=("code:1", "code:0")),
        )


def test_source_mismatch_fails_before_model_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend, wrapper, bindings, heads = _components(tiny_qwen3_model)

    with pytest.raises(
        MixtureAwareRefinementError,
        match="exactly cover",
    ):
        refine_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            source_training=_source_report(),
            expert_bindings=bindings[:1],
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=heads,
            args=_arguments(tmp_path / "mismatch"),
            trainer_kwargs={"train_dataset": []},
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert wrapper.expert_ids == ()
    assert all(
        not parameter.requires_grad
        for parameter in tiny_qwen3_model.parameters()
    )


def test_invalid_trainer_keyword_fails_before_route_attachment(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend, wrapper, bindings, heads = _components(tiny_qwen3_model)

    with pytest.raises(
        MixtureAwareRefinementError,
        match="not accepted",
    ):
        refine_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            source_training=_source_report(),
            expert_bindings=bindings,
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=heads,
            args=_arguments(tmp_path / "invalid-keyword"),
            trainer_kwargs={"not_a_trainer_argument": object()},
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert tuple(heads.state_dict().values())[0].count_nonzero().item() == 0


def test_reordered_router_heads_fail_before_refinement(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend, wrapper, bindings, _ = _components(tiny_qwen3_model)
    reversed_heads = AppendOnlyRouterHeads(16)
    for expert_id in ("code:1", "code:0"):
        reversed_heads.append(
            AffineRouterHead(16, layer_index=3, expert_id=expert_id)
        )

    with pytest.raises(
        MixtureAwareRefinementError,
        match="exact TRN-002",
    ):
        refine_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            source_training=_source_report(),
            expert_bindings=bindings,
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=reversed_heads,
            args=_arguments(tmp_path / "reordered"),
            trainer_kwargs={"train_dataset": []},
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert all(
        parameter.count_nonzero().item() == 0
        for parameter in reversed_heads.parameters()
    )


def test_more_than_two_source_experts_are_rejected(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend, wrapper, bindings, heads = _components(tiny_qwen3_model)

    with pytest.raises(
        MixtureAwareRefinementError,
        match="exactly two",
    ):
        refine_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            source_training=_source_report(
                expert_ids=("code:0", "code:1", "code:2")
            ),
            expert_bindings=bindings,
            router_feature_extractor=IdentityRouterFeatureExtractor(16),
            router_heads=heads,
            args=_arguments(tmp_path / "three"),
        )

    assert wrapper.routing_mode == "forced_anchor"
