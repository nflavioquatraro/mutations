"""Public API for the Anchored Plug-in MoE framework."""

from anchored_moe.configuration import AnchoredMoEConfig, ResolvedStrategyConfig
from anchored_moe.experts import Expert, ExpertSerializationError, FullMLPExpert
from anchored_moe.modeling import RoutedMLP
from anchored_moe.policies import (
    AnchoredTop1Selection,
    RouteDecision,
    SelectionPolicy,
)
from anchored_moe.registry import (
    ExpertPackageManifest,
    PackageValidationError,
    PackageValidationReport,
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
    "AnchoredTop1Selection",
    "AppendOnlyRouterHeads",
    "Expert",
    "ExpertPackageManifest",
    "ExpertSerializationError",
    "FullMLPExpert",
    "IdentityRouterFeatureExtractor",
    "PackageValidationError",
    "PackageValidationReport",
    "ResolvedStrategyConfig",
    "RouteDecision",
    "RouterFeatureExtractor",
    "RouterHead",
    "RouterSerializationError",
    "RoutedMLP",
    "SelectionPolicy",
]

__version__ = "0.1.0.dev0"
