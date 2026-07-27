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
    RouteRecord,
    RouteSummary,
    RouteTelemetry,
    Top1DispatchResult,
)
from anchored_moe.policies import (
    AnchoredTop1Selection,
    CapacityDecision,
    OverflowToAnchorCapacity,
    RouteDecision,
    SelectionPolicy,
)
from anchored_moe.registry import (
    ExpertPackage,
    ExpertPackageManifest,
    ExpertRegistry,
    PackageIdentity,
    PackageValidationError,
    PackageValidationReport,
    RegistryChange,
    RegistryLifecycleError,
    RegistrySnapshot,
)
from anchored_moe.routers import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
    RouterFeatureExtractor,
    RouterHead,
    RouterSerializationError,
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
        CapacityDecision,
        OverflowToAnchorCapacity,
        RouteDecision,
        AnchoredTop1Dispatcher,
        Top1DispatchResult,
        RouteRecord,
        RouteSummary,
        RouteTelemetry,
        ExpertPackage,
        ExpertPackageManifest,
        ExpertRegistry,
        PackageIdentity,
        PackageValidationError,
        PackageValidationReport,
        RegistryChange,
        RegistryLifecycleError,
        RegistrySnapshot,
        RouterFeatureExtractor,
        IdentityRouterFeatureExtractor,
        RouterHead,
        AffineRouterHead,
        AppendOnlyRouterHeads,
        RouterSerializationError,
        canonical_json_bytes,
        sha256_file,
        sha256_json,
    ]

    assert all(inspect.getdoc(symbol) for symbol in public_symbols)
