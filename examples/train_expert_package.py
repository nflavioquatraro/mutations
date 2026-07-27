"""Train and export one deterministic expert-only package without downloads."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AnchoredMoEConfig,
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    FullMLPExpert,
    RoutedMLP,
    export_full_mlp_expert_package,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import fingerprint_module
from anchored_moe.utils import sha256_json


def _package_version(name: str) -> str:
    """Return installed distribution metadata without importing optional code."""

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _tiny_model() -> Qwen3ForCausalLM:
    """Create the deterministic Qwen3 smoke fixture."""

    torch.manual_seed(13)
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


def main() -> None:
    """Train isolated layer experts, publish a package, and verify invariants."""

    model = _tiny_model()
    base_weight_sha256 = fingerprint_module(model)
    dataset = []
    for values in ((1, 7, 11, 3), (2, 5, 13, 9)):
        tokens = torch.tensor(values, dtype=torch.long)
        with torch.no_grad():
            anchor_logits = model(input_ids=tokens.unsqueeze(0)).logits.squeeze(0)
        dataset.append(
            {
                "input_ids": tokens,
                "attention_mask": torch.ones_like(tokens),
                "labels": tokens.clone(),
                "anchor_logits": anchor_logits.detach().clone(),
            }
        )

    backend = Qwen3Backend()
    backend.convert(model, AnchoredMoEConfig(converted_layers=[2, 3]))
    training_bindings: list[ForcedExpertBinding] = []
    anchor_fingerprints: dict[int, str] = {}
    expert_initial_state: dict[int, dict[str, torch.Tensor]] = {}
    for layer_index in (2, 3):
        wrapper = model.model.layers[layer_index].mlp
        assert isinstance(wrapper, RoutedMLP)
        anchor_fingerprints[layer_index] = fingerprint_module(
            wrapper.anchor_mlp
        )
        expert = FullMLPExpert.from_anchor(wrapper.anchor_mlp)
        expert_initial_state[layer_index] = {
            name: tensor.detach().clone()
            for name, tensor in expert.state_dict().items()
        }
        training_bindings.append(
            ForcedExpertBinding(layer_index, "code:0", expert)
        )

    with TemporaryDirectory(prefix="anchored-moe-nb003-") as temporary:
        root = Path(temporary)
        trainer = ForcedPathExpertTrainer(
            model=model,
            backend=backend,
            expert_bindings=training_bindings,
            args=TrainingArguments(
                output_dir=root / "trainer",
                max_steps=1,
                per_device_train_batch_size=1,
                learning_rate=1e-2,
                optim="adamw_torch",
                save_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                dataloader_pin_memory=False,
                seed=41,
            ),
            train_dataset=dataset,
        )
        result = trainer.train()

        package = export_full_mlp_expert_package(
            root / "code-expert",
            bindings=[
                ExpertPackageExportBinding(
                    binding.layer_index,
                    binding.expert_id,
                    binding.expert,
                )
                for binding in training_bindings
            ],
            config=ExpertPackageExportConfig(
                specialization_name="code",
                specialization_version="1.0.0",
                base_model_repository="local-random-qwen3",
                base_model_revision="synthetic-seed-13",
                base_config_sha256=sha256_json(model.config.to_dict()),
                base_weight_sha256={
                    "in-memory-anchor-state": base_weight_sha256
                },
                tokenizer_repository="local-random-qwen3",
                tokenizer_revision="synthetic-seed-13",
                base_model_dtype="float32",
                runtime_versions={
                    "accelerate": _package_version("accelerate"),
                    "kernels": "torch-default-cpu",
                    "peft": _package_version("peft"),
                    "python": platform.python_version(),
                    "safetensors": _package_version("safetensors"),
                    "torch": torch.__version__,
                    "transformers": _package_version("transformers"),
                },
                training_dataset={
                    "dataset_id": "synthetic-local-code",
                    "revision": "1",
                    "split": "train",
                    "filters": [],
                },
                router_feature_specification={
                    "feature_type": "identity",
                    "input": "pre_mlp_hidden_state",
                    "output_size": model.config.hidden_size,
                    "version": 1,
                },
                expert_parameterization_config={
                    "initialization": "anchor_copy",
                    "training_strategy": "forced_path",
                    "optimizer": "adamw_torch",
                    "max_steps": result.global_step,
                },
            ),
        )

        experts_changed = all(
            any(
                not torch.equal(
                    tensor,
                    expert_initial_state[binding.layer_index][name],
                )
                for name, tensor in binding.expert.state_dict().items()
            )
            for binding in training_bindings
        )
        anchor_exact = all(
            fingerprint_module(
                model.model.layers[layer_index].mlp.anchor_mlp
            )
            == anchor_fingerprints[layer_index]
            for layer_index in (2, 3)
        )
        assert experts_changed
        assert anchor_exact
        assert [binding.key for binding in package.payload_bindings] == [
            (2, "code:0"),
            (3, "code:0"),
        ]
        assert package.manifest.router_architecture == "none"
        package.validate_live_files()

        print(
            "NB-003 validation passed: "
            f"steps={result.global_step}, "
            f"layers={package.manifest.converted_layers}, "
            f"payloads={len(package.manifest.expert_weight_files)}, "
            f"anchor_exact={anchor_exact}, "
            f"experts_changed={experts_changed}, "
            f"manifest_sha256={package.identity.manifest_sha256}"
        )


if __name__ == "__main__":
    main()
