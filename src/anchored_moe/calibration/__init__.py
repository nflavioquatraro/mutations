"""Router calibration data contracts and reference-bank serialization."""

from anchored_moe.calibration.features import (
    ROUTER_FEATURE_BANK_FORMAT_VERSION,
    FeatureBankSerializationError,
    RouterFeatureBank,
)
from anchored_moe.calibration.global_recalibration import (
    GlobalRouterRecalibrationError,
    GlobalRouterRecalibrationReport,
    LayerGlobalRouterRecalibrationReport,
    recalibrate_global_router,
)
from anchored_moe.calibration.labels import (
    ROUTER_LABEL_BANK_FORMAT_VERSION,
    DomainLabelBank,
    LabelBankSerializationError,
    UtilityLabelBank,
)
from anchored_moe.calibration.margins import (
    HeadStabilityMarginReport,
    StabilityMarginRefinementError,
    StabilityMarginRefinementReport,
    refine_stability_margins,
)
from anchored_moe.calibration.mixture import (
    MixtureWeightFittingConfig,
    MixtureWeightFittingError,
    MixtureWeightFittingReport,
    fit_learned_mixture_head,
)
from anchored_moe.calibration.oracle import (
    CounterfactualOracleResult,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    evaluate_counterfactual_oracle,
)
from anchored_moe.calibration.pairs import (
    PairSamplingPlan,
    PairUtilityResult,
    compute_pair_utilities,
    plan_expert_pairs,
)
from anchored_moe.calibration.ridge import (
    RidgeFittingError,
    fit_independent_ridge,
)
from anchored_moe.calibration.temperature import (
    HeadCalibrationReport,
    TemperatureThresholdCalibrationError,
    TemperatureThresholdCalibrationReport,
    calibrate_temperature_and_thresholds,
)

__all__ = [
    "DomainLabelBank",
    "CounterfactualOracleResult",
    "CounterfactualRoute",
    "CounterfactualRouteLosses",
    "FeatureBankSerializationError",
    "GlobalRouterRecalibrationError",
    "GlobalRouterRecalibrationReport",
    "HeadCalibrationReport",
    "HeadStabilityMarginReport",
    "LabelBankSerializationError",
    "LayerGlobalRouterRecalibrationReport",
    "MixtureWeightFittingConfig",
    "MixtureWeightFittingError",
    "MixtureWeightFittingReport",
    "PairSamplingPlan",
    "PairUtilityResult",
    "ROUTER_FEATURE_BANK_FORMAT_VERSION",
    "ROUTER_LABEL_BANK_FORMAT_VERSION",
    "RidgeFittingError",
    "RouterFeatureBank",
    "StabilityMarginRefinementError",
    "StabilityMarginRefinementReport",
    "TemperatureThresholdCalibrationError",
    "TemperatureThresholdCalibrationReport",
    "UtilityLabelBank",
    "calibrate_temperature_and_thresholds",
    "compute_pair_utilities",
    "evaluate_counterfactual_oracle",
    "fit_independent_ridge",
    "fit_learned_mixture_head",
    "plan_expert_pairs",
    "recalibrate_global_router",
    "refine_stability_margins",
]
