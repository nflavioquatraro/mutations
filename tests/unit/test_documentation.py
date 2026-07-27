"""Contract tests for foundation documentation."""

from __future__ import annotations

import inspect
from pathlib import Path

import anchored_moe
from anchored_moe.backends import ConversionReport, MLPReference, ModelBackend, Qwen3Backend
from anchored_moe.calibration import (
    DomainLabelBank,
    FeatureBankSerializationError,
    HeadCalibrationReport,
    LabelBankSerializationError,
    RidgeFittingError,
    RouterFeatureBank,
    TemperatureThresholdCalibrationError,
    TemperatureThresholdCalibrationReport,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    fit_independent_ridge,
)
from anchored_moe.evaluation import (
    RegistryLifecycleEvaluation,
    RouteAgreement,
    TensorEquivalence,
    assert_route_decisions_equal,
    assert_tensors_equivalent,
    compare_route_decisions,
    compare_tensors,
    evaluate_registry_lifecycle,
    fingerprint_module,
)
from anchored_moe.experts import Expert, ExpertSerializationError, FullMLPExpert
from anchored_moe.inference import (
    AnchoredTop1Dispatcher,
    AnchoredTop2Dispatcher,
    PackageMaterializationError,
    RegistryCompositionReport,
    RouteRecord,
    RouteSummary,
    RouteTelemetry,
    Top1DispatchResult,
    Top2DispatchResult,
    synchronize_anchored_top1_registry,
)
from anchored_moe.policies import (
    ActivationCombinationResult,
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CapacityDecision,
    CombinationPolicy,
    OverflowToAnchorCapacity,
    RouteDecision,
    ScoreSoftmaxCombination,
    SelectionPolicy,
    Top2RouteDecision,
)
from anchored_moe.registry import (
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    CalibratedRouterPackageConfig,
    ExpertPackage,
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ExpertPackageExportError,
    ExpertPackageManifest,
    ExpertPayloadBinding,
    ExpertRegistry,
    PackageIdentity,
    PackageValidationError,
    PackageValidationReport,
    RegistryChange,
    RegistryLifecycleError,
    RegistrySnapshot,
    derive_calibrated_router_package,
    export_full_mlp_expert_package,
    parse_expert_payload_bindings,
)
from anchored_moe.routers import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    RouterFeatureExtractor,
    RouterHead,
    RouterSerializationError,
)
from anchored_moe.training import (
    ExpertObjectiveConfig,
    ExpertObjectiveError,
    ExpertObjectiveReport,
    ExpertWeightDrift,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    ForcedPathTrainingError,
    ForcedPathTrainingReport,
    ReplayKLPreservation,
    prepare_forced_path_training,
)
from anchored_moe.utils import canonical_json_bytes, sha256_file, sha256_json


def test_feature_bearing_subpackages_have_theory_readmes() -> None:
    source_root = Path(__file__).parents[2] / "src" / "anchored_moe"
    feature_subpackages = {
        path.parent
        for path in source_root.glob("*/*.py")
        if path.name != "__init__.py"
    }

    assert feature_subpackages
    assert all((subpackage / "README.md").is_file() for subpackage in feature_subpackages)


def test_current_public_symbols_have_docstrings() -> None:
    public_symbols = [
        anchored_moe.AnchoredMoEConfig,
        anchored_moe.AnchoredModelError,
        anchored_moe.AnchoredMoEForCausalLM,
        anchored_moe.ResolvedStrategyConfig,
        anchored_moe.RoutedMLP,
        ConversionReport,
        MLPReference,
        ModelBackend,
        Qwen3Backend,
        RegistryLifecycleEvaluation,
        RouteAgreement,
        TensorEquivalence,
        assert_route_decisions_equal,
        assert_tensors_equivalent,
        compare_route_decisions,
        compare_tensors,
        evaluate_registry_lifecycle,
        fingerprint_module,
        Expert,
        ExpertSerializationError,
        FullMLPExpert,
        RouterFeatureBank,
        FeatureBankSerializationError,
        DomainLabelBank,
        UtilityLabelBank,
        LabelBankSerializationError,
        RidgeFittingError,
        fit_independent_ridge,
        HeadCalibrationReport,
        TemperatureThresholdCalibrationReport,
        TemperatureThresholdCalibrationError,
        calibrate_temperature_and_thresholds,
        SelectionPolicy,
        AnchoredTop1Selection,
        AnchorVetoTop2Selection,
        CapacityDecision,
        CombinationPolicy,
        ScoreSoftmaxCombination,
        ActivationCombinationResult,
        OverflowToAnchorCapacity,
        RouteDecision,
        Top2RouteDecision,
        AnchoredTop1Dispatcher,
        AnchoredTop2Dispatcher,
        Top1DispatchResult,
        Top2DispatchResult,
        RouteRecord,
        RouteSummary,
        RouteTelemetry,
        PackageMaterializationError,
        RegistryCompositionReport,
        synchronize_anchored_top1_registry,
        CalibratedRouterPackageConfig,
        ExpertPackage,
        ExpertPackageExportBinding,
        ExpertPackageExportConfig,
        ExpertPackageExportError,
        ExpertPackageManifest,
        ExpertPayloadBinding,
        ExpertRegistry,
        PackageIdentity,
        PackageValidationError,
        PackageValidationReport,
        RegistryChange,
        RegistryLifecycleError,
        RegistrySnapshot,
        derive_calibrated_router_package,
        export_full_mlp_expert_package,
        parse_expert_payload_bindings,
        RouterFeatureExtractor,
        IdentityRouterFeatureExtractor,
        RouterHead,
        AffineRouterHead,
        AppendOnlyRouterHeads,
        RouterSerializationError,
        ForcedExpertBinding,
        ForcedPathExpertTrainer,
        ForcedPathTrainingError,
        ForcedPathTrainingReport,
        ExpertObjectiveConfig,
        ExpertObjectiveError,
        ExpertObjectiveReport,
        ExpertWeightDrift,
        ReplayKLPreservation,
        prepare_forced_path_training,
        canonical_json_bytes,
        sha256_file,
        sha256_json,
    ]

    assert EXPERT_PAYLOAD_BINDING_FORMAT_VERSION == 1
    assert all(inspect.getdoc(symbol) for symbol in public_symbols)
