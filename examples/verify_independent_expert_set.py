"""Verify independent two-expert training on a tiny local Qwen3 model."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ForcedExpertBinding,
    FullMLPExpert,
    IndependentExpertTrainingRun,
    RoutedMLP,
    train_independent_expert_set,
)
from anchored_moe.backends import Qwen3Backend


def _tiny_model() -> Qwen3ForCausalLM:
    """Return a deterministic model small enough for two CPU optimizer steps."""

    torch.manual_seed(71)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
        )
    ).eval()


def _dataset(offset: int) -> list[dict[str, torch.Tensor]]:
    """Return one caller-owned deterministic specialization partition."""

    rows = (
        (1 + offset, 3 + offset, 5 + offset, 7 + offset),
        (2 + offset, 4 + offset, 6 + offset, 8 + offset),
    )
    return [
        {
            "input_ids": (tokens := torch.tensor(row, dtype=torch.long)),
            "labels": tokens.clone(),
        }
        for row in rows
    ]


def _run(
    wrapper: RoutedMLP,
    *,
    expert_id: str,
    dataset: list[dict[str, torch.Tensor]],
    output_dir: Path,
    seed: int,
) -> tuple[IndependentExpertTrainingRun, FullMLPExpert]:
    """Build one independently owned path with its own data and seed."""

    expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
    run = IndependentExpertTrainingRun(
        expert_bindings=[ForcedExpertBinding(3, expert_id, expert)],
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
            seed=seed,
        ),
        trainer_kwargs={"train_dataset": dataset},
    )
    return run, expert


def main() -> None:
    """Train two paths and prove exact anchor restoration and isolation."""

    model = _tiny_model()
    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[3]))
    wrapper = model.model.layers[3].mlp
    assert isinstance(wrapper, RoutedMLP)

    anchor_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    with TemporaryDirectory(prefix="anchored-moe-independent-") as directory:
        root = Path(directory)
        code_run, code_expert = _run(
            wrapper,
            expert_id="code:0",
            dataset=_dataset(0),
            output_dir=root / "code",
            seed=73,
        )
        code_one_run, code_one_expert = _run(
            wrapper,
            expert_id="code:1",
            dataset=_dataset(9),
            output_dir=root / "code-1",
            seed=79,
        )
        experts = (code_expert, code_one_expert)
        expert_before = tuple(
            {
                name: tensor.detach().clone()
                for name, tensor in expert.state_dict().items()
            }
            for expert in experts
        )

        report = train_independent_expert_set(
            model,
            backend=backend,
            runs=[code_run, code_one_run],
        )

    expert_changed = tuple(
        any(
            not torch.equal(tensor, before[name])
            for name, tensor in expert.state_dict().items()
        )
        for expert, before in zip(experts, expert_before, strict=True)
    )
    model_after = model.state_dict()
    anchor_exact = (
        model_after.keys() == anchor_before.keys()
        and all(
            torch.equal(model_after[name], tensor)
            for name, tensor in anchor_before.items()
        )
    )
    assert expert_changed == (True, True)
    assert anchor_exact
    assert wrapper.routing_mode == "forced_anchor"
    assert all(
        not parameter.requires_grad
        for expert in experts
        for parameter in expert.parameters()
    )

    print(
        "TRN-002 independent expert-set validation passed: "
        f"experts={report.expert_ids}, layers={report.layer_indices}, "
        f"steps={report.total_global_steps}, "
        f"expert_changed={expert_changed}, anchor_exact={anchor_exact}"
    )


if __name__ == "__main__":
    main()
