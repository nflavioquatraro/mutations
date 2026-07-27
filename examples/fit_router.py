"""Fit, calibrate, package, and materialize one anchored top-1 router."""

from __future__ import annotations

import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import accelerate
import safetensors
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from anchored_moe import (
    AnchoredMoEConfig,
    AnchoredTop1Selection,
    CalibratedRouterPackageConfig,
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ExpertRegistry,
    FullMLPExpert,
    RouterFeatureBank,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    derive_calibrated_router_package,
    export_full_mlp_expert_package,
    fit_independent_ridge,
    synchronize_anchored_top1_registry,
)
from anchored_moe.backends import Qwen3Backend
from anchored_moe.evaluation import fingerprint_module
from anchored_moe.utils import sha256_json


def _tiny_model() -> Qwen3ForCausalLM:
    """Return a deterministic local Qwen3 fixture."""

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


def _runtime_versions() -> dict[str, str]:
    """Capture the package-derivation runtime without optional imports."""

    return {
        "accelerate": accelerate.__version__,
        "kernels": "torch-default-cpu",
        "peft": "not-installed",
        "python": platform.python_version(),
        "safetensors": safetensors.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _feature_specification() -> dict[str, object]:
    """Return the frozen identity feature contract."""

    return {
        "feature_type": "identity",
        "input": "pre_mlp_hidden_state",
        "output_size": 16,
        "version": 1,
    }


def _feature_bank(
    scalar_features: torch.Tensor,
    *,
    row_ids: torch.Tensor,
    split: str,
) -> RouterFeatureBank:
    """Build aligned two-layer features with one informative coordinate."""

    features = torch.zeros(len(scalar_features), 16)
    features[:, 0] = scalar_features
    return RouterFeatureBank(
        {2: features, 3: features.clone()},
        feature_specification=_feature_specification(),
        dataset_provenance={
            "dataset_id": "synthetic-local-router",
            "revision": "1",
            "split": split,
            "filters": [],
        },
        seed=73,
        row_ids=row_ids,
    )


def _utility_labels(
    bank: RouterFeatureBank,
    utilities: torch.Tensor,
) -> UtilityLabelBank:
    """Bind one expert utility column to both converted layers."""

    values = utilities.reshape(-1, 1)
    return UtilityLabelBank(
        {2: values, 3: values.clone()},
        expert_ids_by_layer={2: ("code:0",), 3: ("code:0",)},
        row_ids=bank.row_ids,
        feature_bank_metadata_sha256=bank.metadata_sha256,
        loss_name="cross_entropy",
        incumbent_route="anchor",
    )


def main() -> None:
    """Exercise the complete NB-003 to NB-004 artifact boundary."""

    model = _tiny_model()
    backend = Qwen3Backend()
    feature_specification = _feature_specification()
    runtime_versions = _runtime_versions()

    with TemporaryDirectory(prefix="anchored-moe-nb004-") as temporary:
        workspace = Path(temporary)
        source = export_full_mlp_expert_package(
            workspace / "code-expert-only",
            bindings=[
                ExpertPackageExportBinding(
                    layer_index,
                    "code:0",
                    FullMLPExpert.from_anchor(
                        model.model.layers[layer_index].mlp
                    ),
                )
                for layer_index in (2, 3)
            ],
            config=ExpertPackageExportConfig(
                specialization_name="code",
                specialization_version="1.0.0",
                base_model_repository="local-random-qwen3",
                base_model_revision="synthetic-seed-13",
                base_config_sha256=sha256_json(model.config.to_dict()),
                base_weight_sha256={
                    "in-memory-anchor-state": fingerprint_module(model)
                },
                tokenizer_repository="local-random-qwen3",
                tokenizer_revision="synthetic-seed-13",
                base_model_dtype="float32",
                runtime_versions=runtime_versions,
                training_dataset={
                    "dataset_id": "synthetic-local-code",
                    "revision": "1",
                    "split": "train",
                    "filters": [],
                },
                router_feature_specification=feature_specification,
            ),
        )
        source_manifest_bytes = (
            source.package_directory / "manifest.json"
        ).read_bytes()

        fitting_scalars = torch.tensor(
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        )
        fitting_bank = _feature_bank(
            fitting_scalars,
            row_ids=torch.arange(6),
            split="train",
        )
        fitting_labels = _utility_labels(fitting_bank, fitting_scalars)
        fitted = fit_independent_ridge(
            fitting_bank,
            fitting_labels,
            regularization=1e-6,
        )

        calibration_scalars = torch.tensor(
            [-2.0, -1.0, 1.0, 2.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]
        )
        calibration_utilities = torch.tensor(
            [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        )
        calibration_bank = _feature_bank(
            calibration_scalars,
            row_ids=torch.arange(10, 20),
            split="validation",
        )
        calibration_labels = _utility_labels(
            calibration_bank,
            calibration_utilities,
        )
        calibrated, report = calibrate_temperature_and_thresholds(
            fitted,
            calibration_bank,
            calibration_labels,
            temperature_row_ids=[10, 11, 12, 13],
            threshold_row_ids=[14, 15, 16, 17, 18, 19],
            false_activation_rate=1.0 / 3.0,
        )
        derived = derive_calibrated_router_package(
            source,
            workspace / "code-routed",
            calibrated_router=calibrated,
            calibration_report=report,
            fitting_feature_bank=fitting_bank,
            fitting_utility_labels=fitting_labels,
            calibration_feature_bank=calibration_bank,
            calibration_utility_labels=calibration_labels,
            config=CalibratedRouterPackageConfig(
                specialization_version="1.0.0+router.1",
                calibration_dataset_version="synthetic-router-v1",
                ridge_regularization=1e-6,
                runtime_versions=runtime_versions,
            ),
        )

        scores = calibrated(
            calibration_bank.features_for_layer(2),
            layer_index=2,
        )
        decision = AnchoredTop1Selection()(
            scores,
            expert_ids=calibrated.expert_ids_for_layer(2),
        )
        route_ids = decision.selected_indices.tolist()
        assert 0 in route_ids and 1 in route_ids
        assert (
            source.package_directory / "manifest.json"
        ).read_bytes() == source_manifest_bytes
        assert derived.manifest.router_architecture == (
            "independent_affine_heads"
        )

        backend.convert(
            model,
            AnchoredMoEConfig(converted_layers=[2, 3]),
        )
        registry = ExpertRegistry()
        registry.install(derived)
        composition = synchronize_anchored_top1_registry(
            model,
            backend=backend,
            registry=registry,
        )
        assert composition.configured_layers == (2, 3)

        print(
            "NB-004 validation passed: "
            f"heads={calibrated.head_keys}, "
            f"routes={route_ids}, "
            f"source_immutable=True, "
            f"derived_version={derived.identity.specialization_version}, "
            f"registry_sha256={composition.registry_sha256}"
        )


if __name__ == "__main__":
    main()
