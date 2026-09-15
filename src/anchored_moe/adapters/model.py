"""Model-level LoRA ownership, frozen-base execution, and portable artifacts.

References:
    Hu et al., LoRA: https://arxiv.org/abs/2106.09685
    PEFT: https://huggingface.co/docs/peft/en/package_reference/peft_model

Differences:
    LoRA is attached to an explicitly resolved subset of decoder projections.
    Independent adapters share one immutable base; this is not token-level MoE
    or the repository's private-copy MLP-expert package format.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn

from anchored_moe.adapters.targets import (
    AdapterTargetBackend,
    AdapterTargetConfig,
    AdapterTargetReport,
)
from anchored_moe.evaluation.equivalence import fingerprint_module
from anchored_moe.utils import canonical_json_bytes, sha256_file, sha256_json


@dataclass(frozen=True)
class AdapterBaseIdentity:
    """Pinned base/tokenizer provenance; local synthetic revisions are allowed.

    All fields are nonempty identifiers supplied by the caller. Weight and
    configuration hashes are independently computed on model attachment.
    Tokenizer contents are not loaded or validated by this model-only API.
    """

    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        """Reject missing or whitespace-padded provenance identifiers."""

        for value in asdict(self).values():
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(
                    "Base identity fields must be nonempty normalized strings."
                )


class AdapterSpecializationModel:
    """Own a PEFT causal LM with independently selectable decoder adapters.

    Use :meth:`from_model` on a materialized, unconverted model. Attachment
    mutates the supplied model in place, freezes its complete original state,
    and records its identity. Model access outside this owner's contexts is
    caller-controlled and must not mutate weights or routes. Operations are
    serialized; concurrent or nested operations fail instead of racing.

    Deployment artifacts contain only adapter tensors and checked provenance.
    They are separate from ExpertRegistry packages and Trainer checkpoints.
    """

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        adapter_id: str,
        config: AdapterTargetConfig,
        identity: AdapterBaseIdentity,
        backend: AdapterTargetBackend | None = None,
    ) -> AdapterSpecializationModel:
        """Resolve and attach one adapter, freezing all weights after creation.

        Raises ValueError for already adapted, meta, or unsupported models.
        PEFT allocation errors may leave a partially wrapped caller-owned model;
        construct a fresh base after such an error, rather than retrying it.
        """

        _validate_id(adapter_id)
        if not isinstance(identity, AdapterBaseIdentity):
            raise TypeError("identity must be AdapterBaseIdentity.")
        if hasattr(model, "peft_config"):
            raise ValueError(
                "Use a fresh base; existing PEFT adapters are not adopted."
            )
        if any(p.device.type == "meta" for p in model.parameters()):
            raise ValueError("Materialize the base before attaching adapters.")
        owner = cls()
        owner.backend = backend or AdapterTargetBackend()
        report = owner.backend.resolve(model, config)
        owner.identity = identity
        owner.base_config_sha256 = sha256_json(model.config.to_dict())
        owner.base_weight_sha256 = fingerprint_module(model)
        # Parameter identities survive injection; their immutable set is the
        # optimizer exclusion boundary, independent of PEFT name rewriting.
        owner._base_parameters = tuple(model.parameters())
        owner._lock = Lock()
        owner.configs = {adapter_id: config}
        owner.reports = {adapter_id: report}
        owner.provenance = {adapter_id: {}}
        model.requires_grad_(False)
        owner.model = get_peft_model(
            model, _peft_config(config, report), adapter_name=adapter_id
        )
        owner.model.requires_grad_(False)
        owner.model.eval()
        return owner

    @classmethod
    def from_adapter(
        cls,
        model: nn.Module,
        directory: str | Path,
        *,
        identity: AdapterBaseIdentity,
        adapter_id: str | None = None,
        backend: AdapterTargetBackend | None = None,
    ) -> AdapterSpecializationModel:
        """Load a deployment package onto a fresh, exactly matching base.

        Manifest, base hashes, and tensor topology are validated before adapter
        injection. No unused bootstrap adapter is installed. As with from_model,
        PEFT allocation failure requires discarding the caller's model instance.
        """

        if hasattr(model, "peft_config"):
            raise ValueError("Use a fresh base to load a deployment package.")
        if not isinstance(identity, AdapterBaseIdentity):
            raise TypeError("identity must be AdapterBaseIdentity.")
        owner = cls()
        owner.model = model
        owner.identity = identity
        owner.backend = backend or AdapterTargetBackend()
        owner.base_config_sha256 = sha256_json(model.config.to_dict())
        owner.base_weight_sha256 = fingerprint_module(model)
        manifest, state, config, report = owner._read_adapter(directory)
        name = adapter_id or manifest["adapter_id"]
        _validate_id(name)
        owner._base_parameters = tuple(model.parameters())
        owner._lock = Lock()
        owner.configs = {name: config}
        owner.reports = {name: report}
        owner.provenance = {name: manifest["training_provenance"]}
        model.requires_grad_(False)
        owner.model = get_peft_model(
            model, _peft_config(config, report), adapter_name=name
        )
        owner._restore_state(name, state)
        owner.model.requires_grad_(False)
        owner.model.eval()
        return owner

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        """Return installed identifiers in deterministic lexical order."""

        return tuple(sorted(self.configs))

    def add_adapter(
        self, adapter_id: str, config: AdapterTargetConfig
    ) -> AdapterTargetReport:
        """Attach an independently initialized adapter; freeze it and incumbents."""

        _validate_id(adapter_id)
        with self.exclusive():
            return self._add_adapter(adapter_id, config)

    def _add_adapter(
        self, adapter_id: str, config: AdapterTargetConfig
    ) -> AdapterTargetReport:
        if adapter_id in self.configs:
            raise ValueError("Adapter identifier already installed.")
        report = self.backend.resolve(self.model.get_base_model(), config)
        previous = self.model.active_adapter
        try:
            self.model.add_adapter(adapter_id, _peft_config(config, report))
        finally:
            self.model.set_adapter(previous)
            self.model.requires_grad_(False)
        self.configs[adapter_id] = config
        self.reports[adapter_id] = report
        self.provenance[adapter_id] = {}
        return report

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Reject concurrent or nested owner operations, including route switches."""

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Adapter model is already executing another operation.")
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def activate(
        self, adapter_id: str | None, *, training: bool = False
    ) -> Iterator[PeftModel]:
        """Hold one route for a complete operation and restore state on failure.

        None executes the base through PEFT's disabled-adapter context. Training
        requires a named adapter. Only that adapter's parameters are unfrozen;
        all parameters are frozen on exit. Previous module modes are restored.
        """

        if adapter_id is not None and adapter_id not in self.configs:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        if training and adapter_id is None:
            raise ValueError("The base route cannot be trained.")
        with self.exclusive():
            previous = self.model.active_adapter
            modes = tuple((m, m.training) for m in self.model.modules())
            try:
                if adapter_id is not None:
                    self.model.set_adapter(adapter_id)
                if not training:
                    self.model.requires_grad_(False)
                self.model.train(training)
                if adapter_id is None:
                    with self.model.disable_adapter():
                        yield self.model
                else:
                    yield self.model
            finally:
                self.model.set_adapter(previous)
                self.model.requires_grad_(False)
                for module, mode in modes:
                    module.training = mode

    def adapter_parameters(
        self, adapter_id: str
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return exact named LoRA tensors belonging to one installed adapter."""

        if adapter_id not in self.configs:
            raise ValueError("Unknown adapter.")
        markers = (f".lora_A.{adapter_id}.", f".lora_B.{adapter_id}.")
        result = tuple(
            (n, p)
            for n, p in self.model.named_parameters()
            if any(m in n for m in markers)
        )
        if not result:
            raise RuntimeError("Adapter contains no optimizable LoRA matrices.")
        return result

    def assert_trainable(self, adapter_id: str) -> None:
        """Assert adapter-only gradient flags against original parameter identities."""

        expected = {id(p) for _, p in self.adapter_parameters(adapter_id)}
        actual = {id(p) for p in self.model.parameters() if p.requires_grad}
        if actual != expected or any(p.requires_grad for p in self._base_parameters):
            raise RuntimeError("Trainable parameters differ from the selected adapter.")

    def save_adapter(self, directory: str | Path, adapter_id: str) -> Path:
        """Atomically export one adapter to a new directory; never overwrite.

        No base weights, executable pickle, optimizer state, or router are
        included. A manifest binds tensor bytes, topology, and base provenance.
        """

        with self.exclusive():
            return self._save_adapter(directory, adapter_id)

    def _manifest(self, adapter_id: str) -> dict:
        return {
            "format_version": 1,
            "adapter_id": adapter_id,
            "identity": asdict(self.identity),
            "base_config_sha256": self.base_config_sha256,
            "base_weight_sha256": self.base_weight_sha256,
            "config": self.configs[adapter_id].to_dict(),
            "backend": asdict(self.backend),
            "targets": asdict(self.reports[adapter_id]),
            "training_provenance": self.provenance[adapter_id],
        }

    def _save_adapter(self, directory: str | Path, adapter_id: str) -> Path:
        if adapter_id not in self.configs:
            raise ValueError("Unknown adapter.")
        target = Path(directory).absolute()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".adapter-", dir=target.parent))
        try:
            tensors = {
                n: t.detach().cpu().contiguous().clone()
                for n, t in sorted(
                    get_peft_model_state_dict(
                        self.model, adapter_name=adapter_id, save_embedding_layers=False
                    ).items()
                )
            }
            if any(not torch.isfinite(t).all() for t in tensors.values()):
                raise ValueError("Adapter tensors must be finite.")
            save_file(tensors, str(stage / "adapter.safetensors"))
            manifest = self._manifest(adapter_id)
            manifest["weights_sha256"] = sha256_file(stage / "adapter.safetensors")
            (stage / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return target

    def load_adapter(
        self, directory: str | Path, *, adapter_id: str | None = None
    ) -> str:
        """Validate and install a frozen artifact, optionally under a new identifier.

        Rejects incompatible base identity, topology, digests, shapes, dtypes,
        nonfinite tensors, extra files, and overwriting an installed adapter.
        """

        with self.exclusive():
            manifest, state, config, _report = self._read_adapter(directory)
            name = adapter_id or manifest["adapter_id"]
            _validate_id(name)
            self._add_adapter(name, config)
            try:
                self._restore_state(name, state)
                self.provenance[name] = manifest["training_provenance"]
            except Exception:
                self.model.delete_adapter(name)
                self.configs.pop(name)
                self.reports.pop(name)
                self.provenance.pop(name)
                raise
        return name

    def _read_adapter(self, directory: str | Path) -> tuple:
        root = Path(directory)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Adapter package must be a regular directory.")
        if {p.name for p in root.iterdir()} != {"manifest.json", "adapter.safetensors"}:
            raise ValueError("Unexpected adapter package contents.")
        if any(p.is_symlink() or not p.is_file() for p in root.iterdir()):
            raise ValueError("Adapter package files must be regular files.")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected_keys = {
            "format_version",
            "adapter_id",
            "identity",
            "base_config_sha256",
            "base_weight_sha256",
            "config",
            "backend",
            "targets",
            "weights_sha256",
            "training_provenance",
        }
        if set(manifest) != expected_keys or manifest["format_version"] != 1:
            raise ValueError("Unsupported adapter manifest schema/version.")
        _validate_id(manifest["adapter_id"])
        if not isinstance(manifest["training_provenance"], dict):
            raise TypeError("Training provenance must be a JSON object.")
        for key, expected in (
            ("identity", asdict(self.identity)),
            ("base_config_sha256", self.base_config_sha256),
            ("base_weight_sha256", self.base_weight_sha256),
            ("backend", asdict(self.backend)),
        ):
            if canonical_json_bytes(manifest[key]) != canonical_json_bytes(expected):
                raise ValueError(f"Adapter package mismatch: {key}")
        if sha256_file(root / "adapter.safetensors") != manifest["weights_sha256"]:
            raise ValueError("Adapter tensor digest mismatch.")
        values = dict(manifest["config"])
        for key in ("components", "module_paths", "rank_pattern", "alpha_pattern"):
            values[key] = tuple(
                tuple(v) if isinstance(v, list) else v for v in values[key]
            )
        if isinstance(values["layers"], list):
            values["layers"] = tuple(values["layers"])
        config = AdapterTargetConfig(**values)
        native = (
            self.model.get_base_model()
            if isinstance(self.model, PeftModel)
            else self.model
        )
        report = self.backend.resolve(native, config)
        if canonical_json_bytes(asdict(report)) != canonical_json_bytes(
            manifest["targets"]
        ):
            raise ValueError("Resolved adapter topology mismatch.")
        state = load_file(str(root / "adapter.safetensors"))
        expected_shapes = {}
        for path in report.module_names:
            module = native.get_submodule(path)
            if hasattr(module, "get_base_layer"):
                module = module.get_base_layer()
            rank = dict(config.rank_pattern).get(path, config.rank)
            for matrix, shape in (
                ("A", (rank, module.in_features)),
                ("B", (module.out_features, rank)),
            ):
                expected_shapes[f"base_model.model.{path}.lora_{matrix}.weight"] = shape
        if set(state) != set(expected_shapes):
            raise ValueError("Adapter tensor keys do not match topology.")
        for key, tensor in state.items():
            if (
                tuple(tensor.shape) != expected_shapes[key]
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Invalid adapter tensor shape, dtype, or values.")
        return manifest, state, config, report

    def _restore_state(self, adapter_id: str, state: dict) -> None:
        expected = get_peft_model_state_dict(
            self.model, adapter_name=adapter_id, save_embedding_layers=False
        )
        if set(expected) != set(state) or any(
            expected[n].shape != state[n].shape or expected[n].dtype != state[n].dtype
            for n in expected
        ):
            raise ValueError(
                "Adapter payload schema/dtype differs from installed adapter."
            )
        set_peft_model_state_dict(self.model, state, adapter_name=adapter_id)


def _validate_id(adapter_id: str) -> None:
    if (
        not isinstance(adapter_id, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", adapter_id) is None
    ):
        raise ValueError(
            "Adapter IDs must start with a letter and use letters, digits, _ or -."
        )


def _peft_config(
    config: AdapterTargetConfig, report: AdapterTargetReport
) -> LoraConfig:
    # Regex selection is anchored at the entire model path, unlike PEFT suffix
    # matching. Layer indices therefore cannot accidentally include earlier blocks.
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        init_lora_weights=True,
        target_modules="^(?:"
        + "|".join(re.escape(n) for n in report.module_names)
        + ")$",
        rank_pattern={"^" + re.escape(n): v for n, v in config.rank_pattern},
        alpha_pattern={"^" + re.escape(n): v for n, v in config.alpha_pattern},
    )
