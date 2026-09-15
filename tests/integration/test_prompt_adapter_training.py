"""Offline attention-inclusive adapter training, persistence, and routing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import TrainingArguments

from anchored_moe import (
    AdapterBaseIdentity,
    AdapterRoute,
    AdapterSpecializationModel,
    AdapterSpecializationTrainer,
    AdapterTargetBackend,
    AdapterTargetConfig,
    ExpertObjectiveConfig,
    FixedAdapterRouter,
    PromptMessage,
    PromptRoutedAdapterGenerator,
    PromptRoutingConfig,
)

IDENTITY = AdapterBaseIdentity("local-qwen", "seed-13", "local-tokenizer", "v1")


def owner(model, **kwargs):
    return AdapterSpecializationModel.from_model(
        model,
        adapter_id="domain",
        config=AdapterTargetConfig(last_n=2, rank=2, alpha=4, **kwargs),
        identity=IDENTITY,
    )


def arguments(path, **kwargs):
    return TrainingArguments(
        output_dir=str(path),
        per_device_train_batch_size=1,
        max_steps=2,
        learning_rate=0.01,
        use_cpu=True,
        dataloader_pin_memory=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=13,
        **kwargs,
    )


def rows():
    return [
        {
            "input_ids": torch.tensor([1, 7, 11, 3]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 11, 3]),
        },
        {
            "input_ids": torch.tensor([2, 5, 13, 9]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([-100, 5, 13, 9]),
        },
    ]


def test_attention_mlp_training_base_preservation_and_reload(
    tiny_qwen3_model, tmp_path
):
    original = deepcopy(tiny_qwen3_model)
    inputs = torch.tensor([[1, 7, 11, 3]])
    with torch.no_grad():
        baseline = original(inputs).logits
    bank = owner(tiny_qwen3_model)
    assert bank.reports["domain"].layer_indices == (2, 3)
    assert len(bank.reports["domain"].module_names) == 14
    with bank.activate("domain") as model, torch.no_grad():
        assert torch.equal(model(inputs).logits, baseline)
    base_before = [p.detach().clone() for p in bank._base_parameters]
    before = {n: p.detach().clone() for n, p in bank.adapter_parameters("domain")}
    # First LoRA step can leave A gradients zero (B starts zero). Explicitly
    # check B gradients across attention and MLP rather than requiring every
    # matrix to change on its first update.
    with bank.activate("domain", training=True) as model:
        model(inputs, labels=inputs).loss.backward()
        gradients = {n: p.grad for n, p in bank.adapter_parameters("domain")}
        for component in ("self_attn", ".mlp."):
            assert any(
                component in n and g is not None and torch.count_nonzero(g)
                for n, g in gradients.items()
            )
        model.zero_grad(set_to_none=True)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        args=arguments(tmp_path / "train", save_strategy="no"),
        train_dataset=rows(),
        objective_config=ExpertObjectiveConfig(drift_weight=0.1),
    )
    assert trainer.train().global_step == 2
    assert all(not p.requires_grad for p in bank.model.parameters())
    assert all(
        torch.equal(p, old) for p, old in zip(bank._base_parameters, base_before)
    )
    for component in ("self_attn", ".mlp."):
        assert any(
            component in n and not torch.equal(p, before[n])
            for n, p in bank.adapter_parameters("domain")
        )
    with bank.activate(None) as model, torch.no_grad():
        assert torch.equal(model(inputs).logits, baseline)
    with bank.activate("domain") as model, torch.no_grad():
        trained = model(inputs).logits
    bank.save_adapter(tmp_path / "artifact", "domain")
    assert {p.name for p in (tmp_path / "artifact").iterdir()} == {
        "adapter.safetensors",
        "manifest.json",
    }
    bank.load_adapter(tmp_path / "artifact", adapter_id="reloaded")
    with bank.activate("reloaded") as model, torch.no_grad():
        assert torch.equal(model(inputs).logits, trained)
    with pytest.raises(FileExistsError):
        bank.save_adapter(tmp_path / "artifact", "domain")
    restored = AdapterSpecializationModel.from_adapter(
        original, tmp_path / "artifact", identity=IDENTITY
    )
    with restored.activate("domain") as model, torch.no_grad():
        assert torch.equal(model(inputs).logits, trained)


def test_resume_restores_adapter_optimizer_and_drift(tiny_qwen3_model, tmp_path):
    base = deepcopy(tiny_qwen3_model)
    bank = owner(tiny_qwen3_model)
    config = ExpertObjectiveConfig(drift_weight=0.2)
    first = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        objective_config=config,
        args=arguments(tmp_path / "run", save_strategy="steps", save_steps=1),
        train_dataset=rows(),
    )
    first.train()
    expected = {n: p.detach().clone() for n, p in bank.adapter_parameters("domain")}
    resumed_bank = owner(base)
    second = AdapterSpecializationTrainer(
        specialization=resumed_bank,
        adapter_id="domain",
        objective_config=config,
        args=arguments(tmp_path / "resume", save_strategy="no"),
        train_dataset=rows(),
    )
    result = second.train(resume_from_checkpoint=str(tmp_path / "run" / "checkpoint-1"))
    assert result.global_step == 2
    for n, p in resumed_bank.adapter_parameters("domain"):
        torch.testing.assert_close(p, expected[n], atol=0, rtol=0)
    assert second.optimizer.state_dict()["state"]
    for name, tensor in first._drift.snapshot_state().items():
        assert torch.equal(tensor, second._drift.snapshot_state()[name])
    assert not (tmp_path / "run" / "checkpoint-1" / "model.safetensors").exists()


def test_generation_base_fallback_and_one_route_per_response(tiny_qwen3_model):
    reference = deepcopy(tiny_qwen3_model)
    bank = owner(tiny_qwen3_model)
    calls = []

    class Router:
        def select(self, messages):
            calls.append(messages)
            return AdapterRoute("domain", 0.2)

    generator = PromptRoutedAdapterGenerator(
        bank, Router(), PromptRoutingConfig(min_confidence=0.7, history_messages=2)
    )
    history = [
        PromptMessage("system", "Helpful"),
        PromptMessage("user", "Pentagon?"),
        PromptMessage("assistant", "A studio"),
        PromptMessage("user", "Can it remember?"),
    ]
    inputs = torch.tensor([[1, 7, 11]])
    result = generator.generate(
        history, input_ids=inputs, max_new_tokens=2, do_sample=False
    )
    assert calls == [tuple(history[-2:])]
    assert result.requested_route.adapter_id == "domain"
    assert result.effective_route.adapter_id is None
    with torch.inference_mode():
        expected = reference.generate(inputs, max_new_tokens=2, do_sample=False)
    assert torch.equal(result.output, expected)
    assert all(not p.requires_grad for p in bank.model.parameters())
    with pytest.raises(ValueError, match="caches"):
        generator.generate(history, input_ids=inputs, past_key_values=object())
    with pytest.raises(ValueError, match="dynamic"):
        generator.generate(history, input_ids=inputs, cache_implementation="static")
    assert len(calls) == 1


def test_generation_failure_restores_state_and_lock(tiny_qwen3_model, monkeypatch):
    bank = owner(tiny_qwen3_model)
    bank.add_adapter(
        "other", AdapterTargetConfig(last_n=1, components=("mlp",), rank=2)
    )
    previous = bank.model.active_adapter
    generator = PromptRoutedAdapterGenerator(
        bank, FixedAdapterRouter(AdapterRoute("other"))
    )

    def fail(**kwargs):
        assert bank.model.active_adapter == "other"
        with (
            pytest.raises(RuntimeError, match="already executing"),
            bank.activate(None),
        ):
            pass
        raise RuntimeError("generation failure")

    monkeypatch.setattr(bank.model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failure"):
        generator.generate(
            [PromptMessage("user", "hello")], input_ids=torch.tensor([[1, 2]])
        )
    assert bank.model.active_adapter == previous
    with bank.activate(None):
        pass


def test_failure_cleanup_and_optimizer_rejection(tiny_qwen3_model, tmp_path):
    bank = owner(tiny_qwen3_model)
    optimizer = torch.optim.AdamW(bank.model.parameters(), lr=0.01)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        args=arguments(tmp_path, save_strategy="no"),
        train_dataset=rows(),
        optimizers=(optimizer, None),
    )
    with pytest.raises(RuntimeError, match="Optimizer"):
        trainer.train()
    assert all(not p.requires_grad for p in bank.model.parameters())
    with bank.activate(None):
        pass


def test_malformed_package_rejected_before_install(tiny_qwen3_model, tmp_path):
    bank = owner(tiny_qwen3_model)
    bank.save_adapter(tmp_path / "package", "domain")
    weights = tmp_path / "package" / "adapter.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest"):
        bank.load_adapter(tmp_path / "package", adapter_id="bad")
    assert bank.adapter_ids == ("domain",)


@pytest.mark.parametrize(
    "config,indices,count",
    [
        (
            AdapterTargetConfig(layers="all", last_n=None, components=("attention",)),
            (0, 1, 2, 3),
            16,
        ),
        (
            AdapterTargetConfig(layers="upper_half", last_n=None, components=("mlp",)),
            (2, 3),
            6,
        ),
        (
            AdapterTargetConfig(
                layers=(3, 1),
                last_n=None,
                components=(),
                module_paths=("self_attn.q_proj",),
            ),
            (1, 3),
            2,
        ),
    ],
)
def test_target_selection(tiny_qwen3_model, config, indices, count):
    report = AdapterTargetBackend().resolve(tiny_qwen3_model, config)
    assert report.layer_indices == indices
    assert len(report.module_names) == count
    assert all(p.requires_grad for p in tiny_qwen3_model.parameters())


def test_exact_rank_overrides(tiny_qwen3_model):
    name = "model.layers.3.self_attn.q_proj"
    bank = AdapterSpecializationModel.from_model(
        tiny_qwen3_model,
        adapter_id="domain",
        identity=IDENTITY,
        config=AdapterTargetConfig(
            last_n=1, rank=2, rank_pattern=((name, 3),), alpha_pattern=((name, 9),)
        ),
    )
    module = bank.model.get_base_model().get_submodule(name)
    assert module.lora_A["domain"].weight.shape[0] == 3
    assert module.scaling["domain"] == 3


def test_second_adapter_training_preserves_first_and_base(tiny_qwen3_model, tmp_path):
    bank = owner(tiny_qwen3_model)
    first = {n: p.detach().clone() for n, p in bank.adapter_parameters("domain")}
    base = [p.detach().clone() for p in bank._base_parameters]
    bank.add_adapter(
        "second", AdapterTargetConfig(last_n=1, components=("attention",), rank=3)
    )
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="second",
        args=arguments(tmp_path, save_strategy="no"),
        train_dataset=rows(),
        eval_dataset=rows(),
    )
    trainer.train()
    assert all(torch.equal(p, first[n]) for n, p in bank.adapter_parameters("domain"))
    assert all(torch.equal(p, old) for p, old in zip(bank._base_parameters, base))
    assert "eval_loss" in trainer.evaluate()
    assert all(not p.requires_grad for p in bank.model.parameters())


def test_replay_objective_and_checkpointed_gradients(tiny_qwen3_model, tmp_path):
    data = rows()
    for row in data:
        with torch.no_grad():
            row["anchor_logits"] = tiny_qwen3_model(
                row["input_ids"].unsqueeze(0)
            ).logits[0]
        row["replay_token_mask"] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    bank = owner(tiny_qwen3_model)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        train_dataset=data,
        objective_config=ExpertObjectiveConfig(replay_kl_weight=0.1),
        args=arguments(
            tmp_path,
            save_strategy="no",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    trainer.train()
    assert trainer.last_objective_report is not None
    assert trainer.last_objective_report.replay_kl_loss >= -1e-6


@pytest.mark.parametrize(
    "route,match",
    [
        (AdapterRoute("missing", 0.1), "unknown"),
        (AdapterRoute("domain"), "Confidence"),
    ],
)
def test_routing_rejects_invalid_decisions(tiny_qwen3_model, route, match):
    bank = owner(tiny_qwen3_model)
    generator = PromptRoutedAdapterGenerator(
        bank, FixedAdapterRouter(route), PromptRoutingConfig(min_confidence=0.7)
    )
    with pytest.raises(ValueError, match=match):
        generator.generate(
            [PromptMessage("user", "hello")], input_ids=torch.tensor([[1, 2]])
        )


def test_bad_base_and_target_mismatches_do_not_mutate(tiny_qwen3_model, tmp_path):
    import json

    from anchored_moe.utils import canonical_json_bytes

    pristine = deepcopy(tiny_qwen3_model)
    bank = owner(tiny_qwen3_model)
    bank.save_adapter(tmp_path / "package", "domain")
    with torch.no_grad():
        next(pristine.parameters()).add_(1)
    with pytest.raises(ValueError, match="base_weight"):
        AdapterSpecializationModel.from_adapter(
            pristine, tmp_path / "package", identity=IDENTITY
        )
    assert not hasattr(pristine, "peft_config")
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["targets"]["layer_indices"] = [0]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="topology"):
        bank.load_adapter(tmp_path / "package", adapter_id="bad")
    assert bank.adapter_ids == ("domain",)


@pytest.mark.parametrize(
    "config",
    [
        AdapterTargetConfig(last_n=5),
        AdapterTargetConfig(layers=(99,), last_n=None),
        AdapterTargetConfig(components=(), module_paths=("input_layernorm",), last_n=1),
        AdapterTargetConfig(
            rank_pattern=(("model.layers.0.mlp.up_proj", 4),), last_n=1
        ),
    ],
)
def test_target_resolution_errors_are_nonmutating(tiny_qwen3_model, config):
    with pytest.raises((ValueError, AttributeError)):
        AdapterTargetBackend().resolve(tiny_qwen3_model, config)
    assert not hasattr(tiny_qwen3_model, "peft_config")
    assert all(p.requires_grad for p in tiny_qwen3_model.parameters())


def test_last_eight_layers_resolve_on_deeper_model(tiny_qwen3_model):
    from transformers import Qwen3ForCausalLM

    config = deepcopy(tiny_qwen3_model.config)
    config.num_hidden_layers = 12
    config.layer_types = ["full_attention"] * 12
    model = Qwen3ForCausalLM(config)
    bank = owner(model)
    bank.add_adapter("last_eight", AdapterTargetConfig(last_n=8, rank=2))
    assert bank.reports["last_eight"].layer_indices == tuple(range(4, 12))
    assert len(bank.reports["last_eight"].module_names) == 56


def test_training_exception_restores_owner(tiny_qwen3_model, tmp_path):
    from transformers import TrainerCallback

    class FailAfterStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            raise RuntimeError("injected callback failure")

    bank = owner(tiny_qwen3_model)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        args=arguments(tmp_path, save_strategy="no"),
        train_dataset=rows(),
        callbacks=[FailAfterStep()],
    )
    with pytest.raises(RuntimeError, match="callback failure"):
        trainer.train()
    assert all(not p.requires_grad and p.grad is None for p in bank.model.parameters())
    with bank.activate(None):
        pass


def test_deployment_save_is_not_a_resumable_checkpoint(tiny_qwen3_model, tmp_path):
    bank = owner(tiny_qwen3_model)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        args=arguments(tmp_path, save_strategy="no"),
        train_dataset=rows(),
    )
    trainer.save_model(str(tmp_path / "model-only"))
    with pytest.raises(ValueError, match="Incomplete training checkpoint"):
        trainer.train(resume_from_checkpoint=str(tmp_path / "model-only"))


def test_drift_snapshot_restore_is_failure_atomic():
    from anchored_moe import ExpertObjectiveError, ExpertWeightDrift

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    drift = ExpertWeightDrift((("p", parameter),))
    snapshot = drift.snapshot_state()
    snapshot["p"].fill_(float("nan"))
    with pytest.raises(ExpertObjectiveError, match="finite"):
        drift.load_snapshot_state(snapshot)
    assert drift.snapshot_state()["p"].item() == 2


def test_best_checkpoint_reload(tiny_qwen3_model, tmp_path):
    from peft.utils import get_peft_model_state_dict

    bank = owner(tiny_qwen3_model)
    trainer = AdapterSpecializationTrainer(
        specialization=bank,
        adapter_id="domain",
        train_dataset=rows(),
        eval_dataset=rows(),
        args=arguments(
            tmp_path,
            save_strategy="steps",
            save_steps=1,
            eval_strategy="steps",
            eval_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        ),
    )
    trainer.train()
    assert trainer.state.best_model_checkpoint
    manifest, state, _, _ = bank._read_adapter(
        Path(trainer.state.best_model_checkpoint) / "specialist"
    )
    assert manifest["adapter_id"] == "domain"
    actual = get_peft_model_state_dict(
        bank.model, adapter_name="domain", save_embedding_layers=False
    )
    assert all(torch.equal(actual[n].cpu(), state[n]) for n in state)
