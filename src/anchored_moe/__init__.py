"""Public API for the Anchored Plug-in MoE framework."""

from anchored_moe.configuration import AnchoredMoEConfig, ResolvedStrategyConfig
from anchored_moe.modeling import RoutedMLP
from anchored_moe.registry import (
    ExpertPackageManifest,
    PackageValidationError,
    PackageValidationReport,
)

__all__ = [
    "AnchoredMoEConfig",
    "ExpertPackageManifest",
    "PackageValidationError",
    "PackageValidationReport",
    "ResolvedStrategyConfig",
    "RoutedMLP",
]

__version__ = "0.1.0.dev0"
