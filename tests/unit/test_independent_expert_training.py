"""Validation tests for TRN-002 expert-set topology and ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ForcedExpertBinding,
    ForcedPathTrainingError,
    FullMLPExpert,
    IndependentExpertTrainingRun,
    RoutedMLP,
    train_independent_expert_set,
)
from anchored_moe.backends import Qwen3Backend


def _arguments(path: Path, *, seed: int) -> TrainingArguments:
    """Return non-saving CPU arguments without starting a training run."""

    return TrainingArguments(
        output_dir=str(path),
        max_steps=1,
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        seed=seed,
    )


def test_run_normalizes_bindings_and_protects_coordinator_arguments(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    bindings = []
    for layer_index in (2, 3):
        wrapper = tiny_qwen3_model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        bindings.append(
            ForcedExpertBinding(
                layer_index,
                "code:0",
                FullMLPExpert.from_anchor(wrapper.anchor_mlp),
            )
        )

    run = IndependentExpertTrainingRun(
        expert_bindings=bindings[::-1],
        args=_arguments(tmp_path / "code", seed=17),
        trainer_kwargs={"train_dataset": []},
    )

    assert run.expert_id == "code:0"
    assert run.layer_indices == (2, 3)
    with pytest.raises(TypeError):
        run.trainer_kwargs["train_dataset"] = []  # type: ignore[index]
    with pytest.raises(ValueError, match="coordinator-owned"):
        IndependentExpertTrainingRun(
            expert_bindings=bindings,
            args=_arguments(tmp_path / "invalid", seed=19),
            trainer_kwargs={"model": tiny_qwen3_model},
        )
    with pytest.raises(ValueError, match="not accepted"):
        IndependentExpertTrainingRun(
            expert_bindings=bindings,
            args=_arguments(tmp_path / "unknown", seed=20),
            trainer_kwargs={"unknown_trainer_option": True},
        )


def test_nonrectangular_expert_set_fails_before_model_mutation(
    tiny_qwen3_model: Qwen3ForCausalLM,
    tmp_path: Path,
) -> None:
    backend = Qwen3Backend()
    backend.convert(
        tiny_qwen3_model,
        AnchoredMoEConfig(converted_layers=[2, 3]),
    )
    wrappers = tuple(
        tiny_qwen3_model.model.layers[layer_index].mlp
        for layer_index in (2, 3)
    )
    assert all(isinstance(wrapper, RoutedMLP) for wrapper in wrappers)

    code = IndependentExpertTrainingRun(
        expert_bindings=[
            ForcedExpertBinding(
                layer_index,
                "code:0",
                FullMLPExpert.from_anchor(wrapper.anchor_mlp),
            )
            for layer_index, wrapper in zip((2, 3), wrappers, strict=True)
        ],
        args=_arguments(tmp_path / "code", seed=23),
    )
    code_one = IndependentExpertTrainingRun(
        expert_bindings=[
            ForcedExpertBinding(
                2,
                "code:1",
                FullMLPExpert.from_anchor(wrappers[0].anchor_mlp),
            )
        ],
        args=_arguments(tmp_path / "code-1", seed=29),
    )

    with pytest.raises(ForcedPathTrainingError, match="same converted layers"):
        train_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            runs=[code, code_one],
        )

    assert all(wrapper.routing_mode == "forced_anchor" for wrapper in wrappers)
    assert all(not parameter.requires_grad for parameter in tiny_qwen3_model.parameters())


def test_cross_expert_parameter_sharing_fails_before_attachment(
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
    shared = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    code = IndependentExpertTrainingRun(
        expert_bindings=[ForcedExpertBinding(3, "code:0", shared)],
        args=_arguments(tmp_path / "code", seed=31),
    )
    code_one = IndependentExpertTrainingRun(
        expert_bindings=[ForcedExpertBinding(3, "code:1", shared)],
        args=_arguments(tmp_path / "code-1", seed=37),
    )

    with pytest.raises(ForcedPathTrainingError, match="must not share"):
        train_independent_expert_set(
            tiny_qwen3_model,
            backend=backend,
            runs=[code, code_one],
        )

    assert wrapper.routing_mode == "forced_anchor"
    assert len(wrapper.experts) == 0
