"""Isolated model-level adapter training with resumable Trainer checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from anchored_moe.adapters.model import AdapterSpecializationModel
from anchored_moe.training.objectives import (
    ExpertObjectiveConfig,
    ExpertObjectiveReport,
    ExpertWeightDrift,
    ReplayKLPreservation,
)
from anchored_moe.utils import canonical_json_bytes


class AdapterSpecializationTrainer(Trainer):
    """Train one named adapter without running a prompt classifier.

    Args:
        specialization: Owner of a materialized PEFT causal language model.
        adapter_id: Installed adapter to train; the base is never trainable.
        args: Standard Trainer arguments, including checkpoint save strategy.
        objective_config: Optional precomputed replay KL and initialization drift.
        training_provenance: JSON-compatible dataset, revision, split, seed, or
            preprocessing identifiers retained in artifacts and checked on resume.
        trainer_kwargs: Caller datasets, collator, callbacks, metrics, etc.

    Every loss computation verifies active adapter and optimizer ownership.
    Standard Trainer controls scheduling, accumulation, and checkpoint state.
    Checkpoints save only the selected adapter, optional drift snapshots, and
    native Trainer optimizer/scheduler/RNG state. They are trusted local training
    artifacts, not portable untrusted deployment packages. Resume validates
    base identity, targets, and objective settings before loading weights.

    Dataset generation, chat formatting, loss-label construction, classifier
    training, FSDP/DeepSpeed, and quantized training are outside this API. CPU
    single-process behavior is tested; accelerator training requires validation.
    """

    def __init__(
        self,
        *,
        specialization: AdapterSpecializationModel,
        adapter_id: str,
        args: TrainingArguments,
        objective_config: ExpertObjectiveConfig | None = None,
        training_provenance: dict | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if adapter_id not in specialization.adapter_ids:
            raise ValueError("Unknown training adapter.")
        if any(key in trainer_kwargs for key in ("model", "model_init")):
            raise ValueError("The specialization owns the Trainer model.")
        if args.push_to_hub or args.deepspeed or args.fsdp:
            raise ValueError(
                "Hub publication and sharded training are not supported here."
            )
        if args.save_only_model:
            raise ValueError(
                "Use save_adapter for deployment; Trainer checkpoints need optimizer state."
            )
        self.specialization = specialization
        self.adapter_id = adapter_id
        self.objective_config = objective_config or ExpertObjectiveConfig()
        if training_provenance is not None and not isinstance(
            training_provenance, dict
        ):
            raise TypeError("training_provenance must be a JSON object.")
        self.training_provenance = json.loads(
            canonical_json_bytes(training_provenance or {})
        )
        specialization.provenance[adapter_id] = self.training_provenance
        self._training_active = False
        self._expected_ids = {
            id(p) for _, p in specialization.adapter_parameters(adapter_id)
        }
        self._drift = (
            ExpertWeightDrift(specialization.adapter_parameters(adapter_id))
            if self.objective_config.drift_enabled
            else None
        )
        self._kl = (
            ReplayKLPreservation(self.objective_config.replay_temperature)
            if self.objective_config.replay_kl_enabled
            else None
        )
        self.last_objective_report: ExpertObjectiveReport | None = None
        # Trainer/Accelerate sees registered adapters before device placement.
        super().__init__(model=specialization.model, args=args, **trainer_kwargs)

    def create_optimizer(self) -> Any:
        """Build the ordinary optimizer and reject any nonselected parameters."""

        result = super().create_optimizer()
        self._assert_optimizer()
        return result

    def _assert_optimizer(self) -> None:
        if self.optimizer is not None:
            actual = {
                id(p) for group in self.optimizer.param_groups for p in group["params"]
            }
            if actual != self._expected_ids:
                raise RuntimeError(
                    "Optimizer must contain exactly the selected adapter."
                )

    def train(
        self, resume_from_checkpoint: str | bool | None = None, **kwargs: Any
    ) -> Any:
        """Force the adapter for the entire run; freeze all weights on every exit."""

        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
            if checkpoint is None:
                raise ValueError("No checkpoint available to resume.")
        if checkpoint:
            self._validate_checkpoint(Path(checkpoint))
            rng_name = (
                f"rng_state_{self.args.process_index}.pth"
                if self.args.world_size > 1
                else "rng_state.pth"
            )
            required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", rng_name)
            if any(not (Path(checkpoint) / name).is_file() for name in required):
                raise ValueError(
                    "Incomplete training checkpoint; optimizer/scheduler/RNG state is required."
                )
        with self.specialization.activate(self.adapter_id, training=True):
            self.specialization.assert_trainable(self.adapter_id)
            self._assert_optimizer()
            self._training_active = True
            try:
                return super().train(resume_from_checkpoint=checkpoint, **kwargs)
            finally:
                self._training_active = False
                self.model.zero_grad(set_to_none=True)

    def evaluate(self, *args: Any, **kwargs: Any) -> dict:
        """Evaluate the selected specialist, including outside an active run."""

        if self._training_active:
            return super().evaluate(*args, **kwargs)
        with self.specialization.activate(self.adapter_id):
            return super().evaluate(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run prediction under the selected adapter, never an incidental route."""

        if self._training_active:
            return super().predict(*args, **kwargs)
        with self.specialization.activate(self.adapter_id):
            return super().predict(*args, **kwargs)

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        extra = [
            self.objective_config.teacher_logits_key,
            self.objective_config.replay_token_mask_key,
        ]
        self._signature_columns = list(
            dict.fromkeys([*self._signature_columns, *(k for k in extra if k)])
        )

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Add optional masked replay/drift to the caller's causal-LM labels."""

        if self.model.active_adapter != self.adapter_id:
            raise RuntimeError("The active adapter changed during training/evaluation.")
        if self._training_active:
            self.specialization.assert_trainable(self.adapter_id)
            self._assert_optimizer()
        batch = dict(inputs)
        teacher = batch.pop(self.objective_config.teacher_logits_key, None)
        mask = batch.pop(self.objective_config.replay_token_mask_key, None)
        if mask is None:
            mask = batch.get("attention_mask")
        loss, outputs = super().compute_loss(
            model,
            batch,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        replay = loss.new_zeros(())
        if self._kl is not None:
            if teacher is None:
                raise ValueError("Replay KL requires precomputed anchor logits.")
            replay = self._kl(outputs.logits, teacher, token_mask=mask)
        drift = self._drift() if self._drift is not None else loss.new_zeros(())
        total = (
            loss
            + self.objective_config.replay_kl_weight * replay
            + self.objective_config.drift_weight * drift
        )
        self.last_objective_report = ExpertObjectiveReport(
            *(float(x.detach().float().cpu()) for x in (loss, replay, drift, total))
        )
        return (total, outputs) if return_outputs else total

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:
        """Save adapter and objective state without serializing the base model.

        Native Trainer independently writes its optimizer, scheduler, RNG, and
        trainer_state files around this method when checkpointing is enabled.
        """

        if not self.args.should_save:
            return
        if self._training_active:
            self._save_adapter_checkpoint(Path(output_dir or self.args.output_dir))
        else:
            with self.specialization.exclusive():
                self._save_adapter_checkpoint(Path(output_dir or self.args.output_dir))

    def _save_adapter_checkpoint(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.specialization._save_adapter(root / "specialist", self.adapter_id)
        metadata = {
            "format_version": 1,
            "adapter_id": self.adapter_id,
            "objectives": asdict(self.objective_config),
            "training_provenance": self.training_provenance,
        }
        (root / "adapter_training.json").write_bytes(canonical_json_bytes(metadata))
        if self._drift is not None:
            save_file(self._drift.snapshot_state(), str(root / "drift.safetensors"))

    def _validate_checkpoint(self, root: Path) -> tuple:
        metadata = json.loads(
            (root / "adapter_training.json").read_text(encoding="utf-8")
        )
        expected = {
            "format_version": 1,
            "adapter_id": self.adapter_id,
            "objectives": asdict(self.objective_config),
            "training_provenance": self.training_provenance,
        }
        if metadata != expected:
            raise ValueError("Checkpoint adapter/objective configuration mismatch.")
        manifest, state, _config, _report = self.specialization._read_adapter(
            root / "specialist"
        )
        if (
            manifest["adapter_id"] != self.adapter_id
            or manifest["training_provenance"] != self.training_provenance
        ):
            raise ValueError("Checkpoint package and training metadata disagree.")
        if canonical_json_bytes(manifest["config"]) != canonical_json_bytes(
            self.specialization.configs[self.adapter_id].to_dict()
        ):
            raise ValueError("Checkpoint adapter targets or hyperparameters mismatch.")
        return state, load_file(
            str(root / "drift.safetensors")
        ) if self._drift is not None else None

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: Any = None
    ) -> None:
        # Optimizer/scheduler/RNG restoration remains in native Trainer. Only
        # model restoration is replaced, preventing accidental base loading.
        state, drift = self._validate_checkpoint(Path(resume_from_checkpoint))
        if self._drift is not None:
            self._drift.load_snapshot_state(drift)
        self.specialization._restore_state(self.adapter_id, state)

    def _load_best_model(self) -> None:
        self._load_from_checkpoint(self.state.best_model_checkpoint)
