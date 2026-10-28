"""Composable shared-prefix response branches with an optional Needle backend.

Importing this package does not import JAX, Flax, Optax or Needle. Those optional
dependencies are loaded only by backend construction and native training calls.
"""

from .backend import BranchBackend, PrefixState
from .browser_export import (
    BrowserBranchBundleReport,
    apply_cact_patch,
    create_cact_patch,
    export_needle_browser_bundle,
)
from .configuration import BranchConfig, BranchedModelConfig
from .evaluation import (
    BranchLatencyReport,
    BranchUtilityReport,
    benchmark_branch_generation,
    benchmark_branch_request,
    evaluate_branch_utilities,
)
from .generative import (
    GenerativeRouterConfig,
    RoutingMessage,
    parse_generative_route,
    prepare_generative_router_example,
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
    "BrowserBranchBundleReport",
    "CactExportConfig",
    "CactExportReport",
    "CategoricalRouterObjective",
    "CopiedSuffix",
    "FixedBranchSelection",
    "FrozenSuffix",
    "GenerativeRouterConfig",
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
    "RoutingMessage",
    "ThresholdSelection",
    "UtilityRegressionObjective",
    "apply_cact_patch",
    "benchmark_branch_generation",
    "benchmark_branch_request",
    "calibrate_utility_selection",
    "create_cact_patch",
    "evaluate_branch_utilities",
    "export_needle_branch",
    "export_needle_browser_bundle",
    "export_needle_bundle",
    "load_branch_package",
    "parse_generative_route",
    "prepare_generative_router_example",
    "save_branch_package",
]
