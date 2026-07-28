"""Contract tests for foundation documentation."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import anchored_moe
from anchored_moe.backends import ConversionReport, MLPReference, ModelBackend, Qwen3Backend
from anchored_moe.calibration import (
    CounterfactualOracleResult,
    CounterfactualRoute,
    CounterfactualRouteLosses,
    DomainLabelBank,
    FeatureBankSerializationError,
    GlobalRouterRecalibrationError,
    GlobalRouterRecalibrationReport,
    HeadCalibrationReport,
    HeadStabilityMarginReport,
    LabelBankSerializationError,
    LayerGlobalRouterRecalibrationReport,
    MixtureWeightFittingConfig,
    MixtureWeightFittingError,
    MixtureWeightFittingReport,
    PairSamplingPlan,
    PairUtilityResult,
    RidgeFittingError,
    RouterFeatureBank,
    StabilityMarginRefinementError,
    StabilityMarginRefinementReport,
    TemperatureThresholdCalibrationError,
    TemperatureThresholdCalibrationReport,
    UtilityLabelBank,
    calibrate_temperature_and_thresholds,
    compute_pair_utilities,
    evaluate_counterfactual_oracle,
    fit_independent_ridge,
    fit_learned_mixture_head,
    plan_expert_pairs,
    recalibrate_global_router,
    refine_stability_margins,
)
from anchored_moe.evaluation import (
    ActiveMLPMetrics,
    EfficiencyBenchmarkConfig,
    EfficiencyBenchmarkReport,
    ModuleStateFootprint,
    RegistryCompositionEvaluation,
    RegistryCompositionEvaluationError,
    RegistryCompositionTrial,
    RegistryLifecycleEvaluation,
    RegistrySubsetEvaluation,
    RouteAgreement,
    TensorEquivalence,
    Top1Top2OracleComparison,
    assert_route_decisions_equal,
    assert_tensors_equivalent,
    benchmark_operation,
    compare_route_decisions,
    compare_top1_top2_to_oracle,
    compare_tensors,
    evaluate_registry_lifecycle,
    evaluate_registry_compositions,
    fingerprint_module,
    measure_active_mlp_metrics,
    measure_module_state_footprint,
)
from anchored_moe.experts import (
    Expert,
    ExpertSerializationError,
    FullMLPExpert,
    LoRAExpertConfig,
    PEFTLoRAExpert,
)
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
    AdaptiveRouteDecision,
    AdaptiveTop1Top2Selection,
    AnchorVetoTop2Selection,
    AnchoredTop1Selection,
    CapacityDecision,
    CombinationPolicy,
    CompatibilityConstrainedTop2Selection,
    DynamicKRouteDecision,
    DynamicKSelection,
    EqualWeightCombination,
    LearnedMixtureSerializationError,
    LearnedMixtureWeightCombination,
    LearnedMixtureWeightHead,
    NormalizedPositiveEvidenceCombination,
    OverflowToAnchorCapacity,
    RouteDecision,
    ScoreSoftmaxCombination,
    SelectionPolicy,
    Top2CapacityDecision,
    Top2SelectionPolicy,
    Top2RouteDecision,
    Top2ToTop1Capacity,
)
from anchored_moe.registry import (
    COMPOSITION_FORMAT_VERSION,
    AnchoredCompositionManifest,
    EXPERT_PAYLOAD_BINDING_FORMAT_VERSION,
    CalibratedRouterPackageConfig,
    CompositionCheckpointError,
    CompositionPackageEntry,
    ExpertPackage,
    ExpertPackageExportBinding,
    ExpertPackageExportConfig,
    ExpertPackageExportError,
    ExpertPackageManifest,
    ExpertPayloadBinding,
    ExpertRegistry,
    MixtureCompatibilityError,
    MixtureCompatibilityIndex,
    PackageIdentity,
    PackageValidationError,
    PackageValidationReport,
    RegistryChange,
    RegistryLifecycleError,
    RegistryRouterVersion,
    RegistryRouterVersionError,
    REGISTRY_ROUTER_VERSION_FORMAT_VERSION,
    RegistrySnapshot,
    RegistryUpgradeChange,
    derive_calibrated_router_package,
    export_full_mlp_expert_package,
    parse_expert_payload_bindings,
)
from anchored_moe.routers import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    HardRouteDistillationConfig,
    IdentityRouterFeatureExtractor,
    LayerRouterDistillationReport,
    LogicalAppendOnlyRouter,
    RouterDistillationError,
    RouterDistillationReport,
    RouterFeatureExtractor,
    RouterHead,
    RouterScoreBank,
    RouterSerializationError,
    compile_logical_router,
    distill_frozen_router,
)
from anchored_moe.training import (
    AuxiliaryLoadBalancingError,
    AuxiliaryLoadBalancingLoss,
    ExpertObjectiveConfig,
    ExpertObjectiveError,
    ExpertObjectiveReport,
    ExpertWeightDrift,
    ForcedExpertBinding,
    ForcedPathExpertTrainer,
    ForcedPathTrainingError,
    ForcedPathTrainingReport,
    IndependentExpertSetTrainingReport,
    IndependentExpertTrainingReport,
    IndependentExpertTrainingRun,
    LocalMiniMoEExpertBinding,
    LocalMiniMoEObjectiveReport,
    LocalMiniMoETrainer,
    LocalMiniMoETrainingConfig,
    LocalMiniMoETrainingReport,
    LocalMiniMoETrainingTopology,
    ReplayKLPreservation,
    prepare_forced_path_training,
    prepare_local_mini_moe_training,
    train_independent_expert_set,
)
from anchored_moe.utils import canonical_json_bytes, sha256_file, sha256_json


_REPOSITORY_ROOT = Path(__file__).parents[2]
_FEATURE_ID = re.compile(r"[A-Z]+-\d{3}")
_VALID_STATUSES = {
    "Not started",
    "In progress",
    "Blocked",
    "Implemented",
    "Validated",
    "Deferred",
}
_THEORY_SECTIONS = (
    "Scope",
    "Implemented strategies",
    "Algorithms",
    "Assumptions and invariants",
    "Hugging Face integration",
    "Configuration and examples",
    "References",
    "Tests and known limitations",
)


def _tracker_rows() -> dict[str, tuple[str, ...]]:
    """Return tracker feature rows keyed by their stable identifiers."""

    rows: dict[str, tuple[str, ...]] = {}
    tracker = _REPOSITORY_ROOT / "feature-implementation-plan.md"
    for line in tracker.read_text(encoding="utf-8").splitlines():
        cells = tuple(
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        )
        if len(cells) != 8 or _FEATURE_ID.fullmatch(cells[0]) is None:
            continue
        assert cells[0] not in rows, f"duplicate feature ID {cells[0]}"
        rows[cells[0]] = cells
    return rows


def test_feature_bearing_subpackages_have_theory_readmes() -> None:
    source_root = _REPOSITORY_ROOT / "src" / "anchored_moe"
    feature_subpackages = {
        path.parent
        for path in source_root.glob("*/*.py")
        if path.name != "__init__.py"
    }

    assert feature_subpackages
    assert all((subpackage / "README.md").is_file() for subpackage in feature_subpackages)


def test_theory_readmes_follow_the_complete_section_contract() -> None:
    """Keep every feature subpackage aligned with the DOC-002 template."""

    source_root = _REPOSITORY_ROOT / "src" / "anchored_moe"
    for readme in source_root.glob("*/README.md"):
        headings = {
            line.removeprefix("## ").strip()
            for line in readme.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        }
        assert set(_THEORY_SECTIONS) <= headings
        assert (
            "Mathematical formulation" in headings
            or "Behavioral contracts" in headings
        )


def test_tracker_ids_dependencies_statuses_and_notes_are_consistent() -> None:
    """Validate the machine-checkable DOC-003 tracker contract."""

    rows = _tracker_rows()
    assert rows
    for feature_id, cells in rows.items():
        priority, status, last_update, notes = (
            cells[1],
            cells[5],
            cells[6],
            cells[7],
        )
        assert re.fullmatch(r"P[0-3]", priority)
        assert status in _VALID_STATUSES
        dependencies = set(_FEATURE_ID.findall(cells[4]))
        assert dependencies <= rows.keys()
        assert feature_id not in dependencies
        if status == "Not started":
            assert last_update == "—"
        else:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_update)
            assert last_update in notes
        if status == "Validated":
            assert all(
                label in notes
                for label in ("Tests:", "Theory:", "References:")
            )


def test_documented_strategy_rows_match_tracker_status_and_shape() -> None:
    """Synchronize README feature IDs, public names, modules, and statuses."""

    rows = _tracker_rows()
    source_root = _REPOSITORY_ROOT / "src" / "anchored_moe"
    documented: set[str] = set()
    for readme in source_root.glob("*/README.md"):
        for line in readme.read_text(encoding="utf-8").splitlines():
            feature_ids = _FEATURE_ID.findall(line)
            if not line.startswith("|") or not feature_ids:
                continue
            cells = tuple(
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            )
            assert len(cells) == 4, f"invalid strategy row in {readme}: {line}"
            assert "`" in cells[1], f"missing public name in {readme}: {line}"
            assert ".py" in cells[2], f"missing module mapping in {readme}: {line}"
            for feature_id in feature_ids:
                assert feature_id in rows
                assert cells[3] == rows[feature_id][5]
                documented.add(feature_id)

    documentation_corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            _REPOSITORY_ROOT / "README.md",
            *_REPOSITORY_ROOT.glob("src/anchored_moe/*/README.md"),
            *_REPOSITORY_ROOT.glob("docs/*.md"),
        )
    )
    required = {
        feature_id
        for feature_id, cells in rows.items()
        if cells[5] in {"In progress", "Implemented", "Validated"}
    }
    assert all(feature_id in documentation_corpus for feature_id in required)
    assert documented


def test_paper_derived_source_documents_references_and_differences() -> None:
    """Require citations and deviations beside paper-derived source code."""

    source_root = _REPOSITORY_ROOT / "src" / "anchored_moe"
    paper_modules = tuple(
        path
        for path in source_root.rglob("*.py")
        if "arxiv.org/" in path.read_text(encoding="utf-8")
    )
    assert paper_modules
    for module in paper_modules:
        source = module.read_text(encoding="utf-8")
        assert "References:" in source, module
        assert "Difference" in source, module


def test_root_public_exports_are_unique_resolvable_and_documented() -> None:
    """Make DOC-003 public-API coverage automatic instead of list-only."""

    assert len(anchored_moe.__all__) == len(set(anchored_moe.__all__))
    for name in anchored_moe.__all__:
        assert hasattr(anchored_moe, name)
        symbol = getattr(anchored_moe, name)
        if inspect.isclass(symbol) or inspect.isfunction(symbol):
            assert inspect.getdoc(symbol), name


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
        ActiveMLPMetrics,
        EfficiencyBenchmarkConfig,
        EfficiencyBenchmarkReport,
        ModuleStateFootprint,
        RegistryLifecycleEvaluation,
        RegistryCompositionEvaluation,
        RegistryCompositionEvaluationError,
        RegistryCompositionTrial,
        RegistrySubsetEvaluation,
        RouteAgreement,
        TensorEquivalence,
        assert_route_decisions_equal,
        assert_tensors_equivalent,
        benchmark_operation,
        compare_route_decisions,
        compare_tensors,
        evaluate_registry_lifecycle,
        evaluate_registry_compositions,
        fingerprint_module,
        measure_active_mlp_metrics,
        measure_module_state_footprint,
        Expert,
        ExpertSerializationError,
        FullMLPExpert,
        LoRAExpertConfig,
        PEFTLoRAExpert,
        CounterfactualOracleResult,
        CounterfactualRoute,
        CounterfactualRouteLosses,
        RouterFeatureBank,
        FeatureBankSerializationError,
        GlobalRouterRecalibrationError,
        GlobalRouterRecalibrationReport,
        DomainLabelBank,
        UtilityLabelBank,
        LabelBankSerializationError,
        LayerGlobalRouterRecalibrationReport,
        MixtureWeightFittingConfig,
        MixtureWeightFittingError,
        MixtureWeightFittingReport,
        PairSamplingPlan,
        PairUtilityResult,
        RidgeFittingError,
        fit_independent_ridge,
        fit_learned_mixture_head,
        HeadCalibrationReport,
        HeadStabilityMarginReport,
        TemperatureThresholdCalibrationReport,
        TemperatureThresholdCalibrationError,
        StabilityMarginRefinementReport,
        StabilityMarginRefinementError,
        calibrate_temperature_and_thresholds,
        compute_pair_utilities,
        evaluate_counterfactual_oracle,
        Top1Top2OracleComparison,
        compare_top1_top2_to_oracle,
        plan_expert_pairs,
        recalibrate_global_router,
        refine_stability_margins,
        SelectionPolicy,
        Top2SelectionPolicy,
        AdaptiveRouteDecision,
        AdaptiveTop1Top2Selection,
        AnchoredTop1Selection,
        AnchorVetoTop2Selection,
        CapacityDecision,
        CombinationPolicy,
        CompatibilityConstrainedTop2Selection,
        DynamicKRouteDecision,
        DynamicKSelection,
        EqualWeightCombination,
        LearnedMixtureSerializationError,
        LearnedMixtureWeightCombination,
        LearnedMixtureWeightHead,
        NormalizedPositiveEvidenceCombination,
        ScoreSoftmaxCombination,
        ActivationCombinationResult,
        OverflowToAnchorCapacity,
        Top2CapacityDecision,
        Top2ToTop1Capacity,
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
        AnchoredCompositionManifest,
        CompositionCheckpointError,
        CompositionPackageEntry,
        ExpertPackage,
        ExpertPackageExportBinding,
        ExpertPackageExportConfig,
        ExpertPackageExportError,
        ExpertPackageManifest,
        ExpertPayloadBinding,
        ExpertRegistry,
        MixtureCompatibilityError,
        MixtureCompatibilityIndex,
        PackageIdentity,
        PackageValidationError,
        PackageValidationReport,
        RegistryChange,
        RegistryLifecycleError,
        RegistryRouterVersion,
        RegistryRouterVersionError,
        RegistrySnapshot,
        RegistryUpgradeChange,
        derive_calibrated_router_package,
        export_full_mlp_expert_package,
        parse_expert_payload_bindings,
        RouterFeatureExtractor,
        IdentityRouterFeatureExtractor,
        RouterHead,
        AffineRouterHead,
        AppendOnlyRouterHeads,
        HardRouteDistillationConfig,
        LayerRouterDistillationReport,
        LogicalAppendOnlyRouter,
        RouterDistillationError,
        RouterDistillationReport,
        RouterScoreBank,
        compile_logical_router,
        distill_frozen_router,
        RouterSerializationError,
        AuxiliaryLoadBalancingError,
        AuxiliaryLoadBalancingLoss,
        ForcedExpertBinding,
        ForcedPathExpertTrainer,
        ForcedPathTrainingError,
        ForcedPathTrainingReport,
        IndependentExpertSetTrainingReport,
        IndependentExpertTrainingReport,
        IndependentExpertTrainingRun,
        LocalMiniMoEExpertBinding,
        LocalMiniMoEObjectiveReport,
        LocalMiniMoETrainer,
        LocalMiniMoETrainingConfig,
        LocalMiniMoETrainingReport,
        LocalMiniMoETrainingTopology,
        ExpertObjectiveConfig,
        ExpertObjectiveError,
        ExpertObjectiveReport,
        ExpertWeightDrift,
        ReplayKLPreservation,
        prepare_forced_path_training,
        prepare_local_mini_moe_training,
        train_independent_expert_set,
        canonical_json_bytes,
        sha256_file,
        sha256_json,
    ]

    assert EXPERT_PAYLOAD_BINDING_FORMAT_VERSION == 1
    assert COMPOSITION_FORMAT_VERSION == 1
    assert REGISTRY_ROUTER_VERSION_FORMAT_VERSION == 1
    assert all(inspect.getdoc(symbol) for symbol in public_symbols)
