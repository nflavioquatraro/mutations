"""Train, checkpoint, export, reload, and prompt-route adapters without downloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from anchored_moe import (
    AdapterBaseIdentity,
    AdapterRoute,
    AdapterSpecializationModel,
    AdapterSpecializationTrainer,
    AdapterTargetConfig,
    FixedAdapterRouter,
    PromptMessage,
    PromptRoutedAdapterGenerator,
)


def main() -> None:
    """Exercise the public model-level workflow using a tiny deterministic LM."""

    torch.manual_seed(13)
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
        )
    ).eval()
    pristine = deepcopy(base)
    identity = AdapterBaseIdentity("local-qwen3", "seed-13", "local-tokenizer", "v1")
    specialization = AdapterSpecializationModel.from_model(
        base,
        adapter_id="domain",
        identity=identity,
        config=AdapterTargetConfig(
            last_n=2, components=("attention", "mlp"), rank=2, alpha=4
        ),
    )
    tokens = torch.tensor([[1, 7, 11, 3]])
    dataset = [{"input_ids": tokens[0], "labels": torch.tensor([-100, -100, 11, 3])}]
    with TemporaryDirectory(prefix="prompt-adapter-") as directory:
        root = Path(directory)
        trainer = AdapterSpecializationTrainer(
            specialization=specialization,
            adapter_id="domain",
            train_dataset=dataset,
            training_provenance={
                "dataset": "synthetic-smoke",
                "revision": "1",
                "seed": 13,
            },
            args=TrainingArguments(
                output_dir=str(root / "training"),
                max_steps=2,
                per_device_train_batch_size=1,
                learning_rate=0.01,
                save_strategy="steps",
                save_steps=1,
                use_cpu=True,
                dataloader_pin_memory=False,
                logging_strategy="no",
                report_to="none",
                disable_tqdm=True,
                seed=13,
            ),
        )
        trainer.train()
        specialization.save_adapter(root / "deployment", "domain")
        restored = AdapterSpecializationModel.from_adapter(
            pristine,
            root / "deployment",
            identity=identity,
        )
        with specialization.activate("domain") as model, torch.no_grad():
            expected = model(tokens).logits
        with restored.activate("domain") as model, torch.no_grad():
            assert torch.equal(model(tokens).logits, expected)
        generator = PromptRoutedAdapterGenerator(
            restored,
            FixedAdapterRouter(AdapterRoute("domain", 1.0)),
        )
        result = generator.generate(
            [PromptMessage("user", "Demonstrate the synthetic specialist.")],
            input_ids=tokens,
            max_new_tokens=2,
            do_sample=False,
        )
        assert result.effective_route.adapter_id == "domain"
        assert result.output.shape == (1, 6)
        assert all(not p.requires_grad for p in restored.model.parameters())
        print(
            "Prompt adapter validation passed: attention+MLP, adapter-only training, exact reload, fixed response route"
        )


if __name__ == "__main__":
    main()
