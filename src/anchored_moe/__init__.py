"""Public API for the Anchored Plug-in MoE framework."""

from anchored_moe.calibration import (
    ROUTER_FEATURE_BANK_FORMAT_VERSION,
    ROUTER_LABEL_BANK_FORMAT_VERSION,
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
from anchored_moe.configuration import AnchoredMoEConfig, ResolvedStrategyConfig
from anchored_moe.experts import Expert, ExpertSerializationError, FullMLPExpert
from anchored_moe.inference import (
    AnchoredTop1Dispatcher,
    RouteRecord,
    RouteSummary,
    RouteTelemetry,
    Top1DispatchResult,
)
from anchored_moe.modeling import RoutedMLP
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

__all__ = [
    "AffineRouterHead",
    "AnchoredMoEConfig",
    "AnchoredTop1Dispatcher",
    "AnchoredTop1Selection",
    "AppendOnlyRouterHeads",
    "CapacityDecision",
    "DomainLabelBank",
    "Expert",
    "ExpertPackage",
    "ExpertPackageManifest",
    "ExpertRegistry",
    "ExpertSerializationError",
    "FeatureBankSerializationError",
    "FullMLPExpert",
    "HeadCalibrationReport",
    "IdentityRouterFeatureExtractor",
    "PackageValidationError",
    "PackageValidationReport",
    "LabelBankSerializationError",
    "OverflowToAnchorCapacity",
    "PackageIdentity",
    "ResolvedStrategyConfig",
    "ROUTER_FEATURE_BANK_FORMAT_VERSION",
    "ROUTER_LABEL_BANK_FORMAT_VERSION",
    "RidgeFittingError",
    "RegistryChange",
    "RegistryLifecycleError",
    "RegistrySnapshot",
    "RouteRecord",
    "RouteDecision",
    "RouteSummary",
    "RouteTelemetry",
    "RouterFeatureBank",
    "RouterFeatureExtractor",
    "RouterHead",
    "RouterSerializationError",
    "RoutedMLP",
    "SelectionPolicy",
    "TemperatureThresholdCalibrationError",
    "TemperatureThresholdCalibrationReport",
    "Top1DispatchResult",
    "UtilityLabelBank",
    "calibrate_temperature_and_thresholds",
    "fit_independent_ridge",
]

__version__ = "0.1.0.dev0"
