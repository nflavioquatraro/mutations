"""Composable shared-prefix response branches with an optional Needle backend.

Importing this package does not import JAX, Flax, Optax or Needle. Those optional
dependencies are loaded only by backend construction and native training calls.
"""

from .backend import BranchBackend, PrefixState
from .configuration import BranchConfig, BranchedModelConfig
from .evaluation import (
    BranchLatencyReport,
    BranchUtilityReport,
    benchmark_branch_generation,
    benchmark_branch_request,
    evaluate_branch_utilities,
)
from .model import BranchedForwardResult, BranchedModel, BranchGenerationResult
from .native_export import (
    CactExportConfig,
    CactExportReport,
    NativeBranchBundleReport,
    export_needle_branch,
    export_needle_bundle,
)
from .needle import NeedleBranchBackend
from .needle_cache import NeedleDecodeState
from .package import (
    BRANCH_PACKAGE_VERSION,
    BranchPackageError,
    load_branch_package,
    save_branch_package,
)
from .parameters import BranchParameterization, CopiedSuffix, FrozenSuffix, LoRASuffix
from .routing import (
    ArgmaxSelection,
    BranchRoute,
    FixedBranchSelection,
    LinearRequestScorer,
    RequestScorer,
    RequestScores,
    RequestSelectionPolicy,
    ThresholdSelection,
    calibrate_utility_selection,
)
from .training import (
    BranchObjective,
    BranchTrainer,
    BranchTrainingReport,
    BranchTrainingStrategy,
    CategoricalRouterObjective,
    JaxAdamStrategy,
    ResponseObjective,
    ResponseTrainingExample,
    RouterTrainingExample,
    UtilityRegressionObjective,
)

__all__ = [
    "BRANCH_PACKAGE_VERSION",
    "ArgmaxSelection",
    "BranchBackend",
    "BranchConfig",
    "BranchGenerationResult",
    "BranchLatencyReport",
    "BranchObjective",
    "BranchPackageError",
    "BranchParameterization",
    "BranchRoute",
    "BranchTrainer",
    "BranchTrainingReport",
    "BranchTrainingStrategy",
    "BranchUtilityReport",
    "BranchedForwardResult",
    "BranchedModel",
    "BranchedModelConfig",
    "CactExportConfig",
    "CactExportReport",
    "CategoricalRouterObjective",
    "CopiedSuffix",
    "FixedBranchSelection",
    "FrozenSuffix",
    "JaxAdamStrategy",
    "LinearRequestScorer",
    "LoRASuffix",
    "NativeBranchBundleReport",
    "NeedleBranchBackend",
    "NeedleDecodeState",
    "PrefixState",
    "RequestScorer",
    "RequestScores",
    "RequestSelectionPolicy",
    "ResponseObjective",
    "ResponseTrainingExample",
    "RouterTrainingExample",
    "ThresholdSelection",
    "UtilityRegressionObjective",
    "benchmark_branch_generation",
    "benchmark_branch_request",
    "calibrate_utility_selection",
    "evaluate_branch_utilities",
    "export_needle_branch",
    "export_needle_bundle",
    "load_branch_package",
    "save_branch_package",
]
