"""Framework contracts independent of the optional Needle/JAX dependencies."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from anchored_moe.branching import (
    ArgmaxSelection,
    BranchBackend,
    BranchConfig,
    BranchedModel,
    BranchedModelConfig,
    BranchPackageError,
    FixedBranchSelection,
    LinearRequestScorer,
    PrefixState,
    RequestScores,
    ThresholdSelection,
    benchmark_branch_generation,
    benchmark_branch_request,
    calibrate_utility_selection,
    evaluate_branch_utilities,
    load_branch_package,
    save_branch_package,
)
from anchored_moe.branching.model import tensor_digest
from anchored_moe.branching.parameters import checked_parameters
from anchored_moe.utils.hashing import sha256_file


class ArrayBackend(BranchBackend):
    """Small deterministic split model exposing execution counts for contracts."""

    def __init__(self):
        self.prefills, self.suffixes = 0, 0
        w = np.eye(2, dtype=np.float32)
        w.setflags(write=False)
        self._suffix = MappingProxyType({"layers/1/block/self_attn/q_proj/kernel": w})

    @property
    def identity(self):
        return {"backend": "array-test", "shared_prefix_layers": 1}

    @property
    def suffix_parameters(self):
        return self._suffix

    @property
    def feature_size(self):
        return 2

    def prefill(self, input_ids):
        self.prefills += 1
        ids = np.asarray(input_ids)
        x = np.stack((ids, 1 + ids), axis=-1).astype(np.float32)
        x.setflags(write=False)
        return PrefixState("array-test", tensor_digest({"ids": ids}), x)

    def features(self, state, suffix=None):
        x = state.payload
        if suffix is not None:
            self.suffixes += 1
            x = x @ next(iter(suffix.values()))
        return x.mean(axis=(0, 1))

    def logits(self, state, suffix):
        self.suffixes += 1
        return state.payload @ next(iter(suffix.values()))

    def native_parameters(self, parameters):
        return dict(parameters)


def make_model(*, router=False, kind="utility", selection=None):
    branches = (
        BranchConfig("base", "frozen"),
        BranchConfig("specialist", "lora", rank=1, targets=("q_proj",)),
    )
    config = BranchedModelConfig(
        1, branches, BranchConfig("router", "copy") if router else None
    )
    ids = ("specialist",) if kind == "utility" else ("base", "specialist")
    scorer = LinearRequestScorer(
        ids, np.zeros((2, len(ids))), np.ones(len(ids)), kind=kind
    )
    return BranchedModel(ArrayBackend(), config, scorer, selection or ArgmaxSelection())


def test_configuration_and_parameter_publication_are_strict():
    with pytest.raises(ValueError, match="unique"):
        BranchedModelConfig(1, (BranchConfig("base", "frozen"),), BranchConfig("base"))
    with pytest.raises(ValueError, match="frozen"):
        BranchedModelConfig(1, (BranchConfig("base"),))
    with pytest.raises(ValueError):
        BranchConfig("../escape")
    model = make_model()
    before = model.composition_key
    params = model.branch_parameters("specialist")
    params[next(iter(params))] = np.full((7,), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        model.replace_branch_parameters("specialist", params)
    assert model.composition_key == before
    with pytest.raises(ValueError, match="frozen"):
        model.replace_branch_parameters("base", {})
    with pytest.raises(ValueError):
        checked_parameters(
            {"x": np.zeros(2, np.float64)}, {"x": np.zeros(2, np.float32)}
        )


def test_request_reuses_prefix_and_runs_only_selected_response():
    model = make_model(router=True)
    result = model.forward(np.array([[2, 3]]))
    assert result.route.branch_id == "specialist"
    assert model.backend.prefills == 1
    assert model.backend.suffixes == 2  # router + one response, never both responses
    original = model.forward_branch(np.array([[2, 3]]), "base")
    np.testing.assert_array_equal(result.logits, original)
    model.set_selection(FixedBranchSelection("base"))
    previous = model.backend.suffixes
    model.forward(np.array([[2, 3]]))
    assert model.backend.suffixes == previous + 1  # fixed policy bypasses router


def test_mutation_is_isolated_and_does_not_change_frozen_base():
    model = make_model()
    ids = np.array([[2, 3]])
    base = model.forward_branch(ids, "base").copy()
    params = model.branch_parameters("specialist")
    key = next(k for k in params if k.endswith("/B"))
    params[key][:] = 2
    # Caller mutation has no effect until explicitly published.
    np.testing.assert_array_equal(model.forward_branch(ids, "specialist"), base)
    model.replace_branch_parameters("specialist", params)
    assert not np.array_equal(model.forward_branch(ids, "specialist"), base)
    np.testing.assert_array_equal(model.forward_branch(ids, "base"), base)
    params[key][:] = 9
    assert not np.array_equal(model.branch_parameters("specialist")[key], params[key])


def test_policy_semantics_anchor_ties_and_thresholds():
    assert (
        ArgmaxSelection()
        .select(RequestScores(("a", "b"), (0.0, 0.0)), "base")
        .branch_id
        == "base"
    )
    assert (
        ArgmaxSelection()
        .select(RequestScores(("a", "b"), (1.0, 1.0)), "base")
        .branch_id
        == "a"
    )
    score = RequestScores(("a", "base"), (0.5, 0.5), "probability")
    assert ArgmaxSelection().select(score, "base").branch_id == "base"
    with pytest.raises(ValueError, match="sum"):
        RequestScores(("a", "b"), (0.6, 0.6), "probability")
    with pytest.raises(ValueError, match="kinds"):
        ThresholdSelection((("a", 0.5),), "utility").select(score, "base")
    score = RequestScores(("a", "b"), (2.0, 1.0))
    route = ThresholdSelection((("a", 2.0), ("b", 0.0))).select(score, "base")
    assert route.branch_id == "base"  # reject winner, do not rerank


def test_ridge_fit_and_calibration_staleness():
    x = np.array([[0, 1], [1, 0], [2, 1], [1, 2]], np.float32)
    y = x[:, :1] - 0.5
    scorer = LinearRequestScorer.fit_ridge(x, y, ("specialist",), regularization=1e-8)
    np.testing.assert_allclose([scorer.score(row).values for row in x], y, atol=1e-5)
    model = make_model()
    model.set_scorer(scorer)
    scores = np.array([[-1.0], [0.2], [1.0]])
    utilities = np.array([[-1.0], [-1.0], [1.0]])
    policy = calibrate_utility_selection(
        scores,
        utilities,
        ("specialist",),
        composition_key=model.composition_key,
        data_id="calibration",
        max_false_activation_rate=0,
    )
    assert policy.thresholds == (("specialist", 0.2),)
    model.set_selection(policy)
    assert not model.calibration_stale
    params = model.branch_parameters("specialist")
    params[next(k for k in params if k.endswith("/B"))][:] = 1
    model.replace_branch_parameters("specialist", params)
    assert model.calibration_stale
    with pytest.raises(RuntimeError, match="stale"):
        model.forward(np.array([[2]]))
    model.set_selection(FixedBranchSelection("base"))
    model.forward(np.array([[2]]))


def test_generation_locks_route_and_cleans_up_failures():
    model = make_model(router=True)
    result = model.generate_tokens(np.array([[2]]), max_new_tokens=3)
    assert result.token_ids == (1, 1, 1)
    assert result.prefix_evaluations == 3
    assert model.backend.suffixes == 4  # router once, response three times
    with pytest.raises(RuntimeError, match="nested"), model.operation():
        model.forward(np.array([[2]]))

    def fail(*args):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        model.generate_tokens(np.array([[2]]), max_new_tokens=1, logits_processor=fail)
    assert model.forward(np.array([[2]])).route.branch_id == "specialist"
    with pytest.raises(ValueError, match="finite"):
        model.generate_tokens(
            np.array([[2]]),
            max_new_tokens=1,
            logits_processor=lambda ids, logits: np.full_like(logits, -np.inf),
        )


def test_package_roundtrip_tamper_and_overwrite(tmp_path):
    model = make_model(router=True)
    params = model.branch_parameters("specialist")
    params[next(k for k in params if k.endswith("/B"))][:] = 1
    model.replace_branch_parameters("specialist", params)
    policy = calibrate_utility_selection(
        [[0], [1]],
        [[-1], [1]],
        ("specialist",),
        composition_key=model.composition_key,
        data_id="heldout",
        max_false_activation_rate=0,
    )
    model.set_selection(policy)
    path = save_branch_package(model, tmp_path / "package")
    reloaded = load_branch_package(path, ArrayBackend())
    assert reloaded.composition_key == model.composition_key
    assert reloaded.selection == model.selection
    for branch in model.response_ids:
        np.testing.assert_array_equal(
            reloaded.forward_branch([[2]], branch), model.forward_branch([[2]], branch)
        )
    with pytest.raises(FileExistsError):
        save_branch_package(model, path)
    payload = path / "branches.safetensors"
    payload.write_bytes(payload.read_bytes() + b"bad")
    with pytest.raises(BranchPackageError, match="digest"):
        load_branch_package(path, ArrayBackend())


def test_reload_rejects_unexpected_tensors_and_changed_base(tmp_path):
    from safetensors.numpy import load_file, save_file

    path = save_branch_package(make_model(), tmp_path / "package")
    tensor_path = path / "branches.safetensors"
    tensors = load_file(str(tensor_path))
    tensors["unknown"] = np.zeros(1, np.float32)
    save_file(tensors, str(tensor_path))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["tensor_sha256"] = sha256_file(tensor_path)
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BranchPackageError, match="Unexpected"):
        load_branch_package(path, ArrayBackend())
    manifest["backend"]["backend"] = "other"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BranchPackageError, match="base"):
        load_branch_package(path, ArrayBackend())


def test_empty_calibration_negatives_and_missing_targets_rejected():
    with pytest.raises(ValueError, match="nonbeneficial"):
        calibrate_utility_selection(
            [[1]], [[1]], ("a",), composition_key="x", data_id="c"
        )
    with pytest.raises(ValueError, match="Unresolved"):
        BranchedModel(
            ArrayBackend(),
            BranchedModelConfig(
                1, (BranchConfig("base", "frozen"), BranchConfig("specialist"))
            ),
            LinearRequestScorer(("specialist",), np.zeros((2, 1)), [0]),
            ArgmaxSelection(),
        )


def test_scorer_is_immutable_and_policy_changes_are_validated():
    model = make_model()
    with pytest.raises(FrozenInstanceError):
        model.scorer.temperature = 2
    copy = model.scorer.parameters
    copy["bias"][:] = -10
    assert model.forward([[2]]).route.branch_id == "specialist"
    for policy in (
        FixedBranchSelection("unknown"),
        ThresholdSelection((("unknown", 0.0),)),
        ThresholdSelection((("specialist", 0.5),), "probability"),
    ):
        with pytest.raises(ValueError):
            model.set_selection(policy)
    assert isinstance(model.selection, ArgmaxSelection)


def test_offline_utilities_reuse_prefix_and_benchmark_synchronizes():
    model = make_model(router=True)
    params = model.branch_parameters("specialist")
    params[next(k for k in params if k.endswith("/B"))][:] = 1
    model.replace_branch_parameters("specialist", params)
    requests = [[[2, 3]], [[4, 5]]]
    report = evaluate_branch_utilities(
        model, requests, lambda index, logits: float(np.mean(logits)), data_id="heldout"
    )
    assert model.backend.prefills == 2
    assert model.backend.suffixes == 4  # all responses, never the router
    assert report.specialist_ids == ("specialist",)
    expected = []
    for ids in requests:
        base = np.mean(model.forward_branch(ids, "base"))
        specialist = np.mean(model.forward_branch(ids, "specialist"))
        expected.append([base - specialist])
    np.testing.assert_allclose(report.utilities, expected)
    assert report.composition_key == model.composition_key
    before = model.backend.prefills
    timing = benchmark_branch_request(model, requests[0], warmup=1, repeats=2)
    assert model.backend.prefills == before + 3
    assert len(timing.samples_seconds) == 2
    assert timing.p95_seconds >= timing.median_seconds > 0
    with pytest.raises(ValueError, match="finite"):
        evaluate_branch_utilities(
            model, requests, lambda *args: np.nan, data_id="invalid"
        )
    model.forward(requests[0])  # error released the owner


def test_package_import_does_not_require_optional_needle_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import anchored_moe; "
                "assert not {'jax', 'flax', 'needle', 'optax'}.intersection(sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_two_specialists_can_use_custom_request_scorer_and_policy():
    class Scorer:
        def score(self, features):
            return RequestScores(("first", "second"), (0.1, float(features[0])))

        def identity(self):
            return {"type": "second-by-first-feature-v1"}

    class Selector:
        def select(self, scores, fallback):
            return ArgmaxSelection().select(scores, fallback)

    config = BranchedModelConfig(
        1,
        (
            BranchConfig("base", "frozen"),
            BranchConfig("first", "copy"),
            BranchConfig("second", "copy"),
        ),
        BranchConfig("router", "copy"),
    )
    model = BranchedModel(ArrayBackend(), config, Scorer(), Selector())
    result = model.forward([[2, 3]])
    assert result.route.branch_id == "second"
    assert model.backend.prefills == 1
    assert model.backend.suffixes == 2
    assert isinstance(model.composition_key, str)


def test_cache_capability_is_explicit_and_generation_benchmark_uses_full_path():
    model = make_model(router=True)
    with pytest.raises(NotImplementedError, match="cached"):
        model.generate_tokens([[2]], max_new_tokens=2, use_cache=True)
    for invalid in ("true", 1):
        with pytest.raises(ValueError, match="use_cache"):
            model.generate_tokens([[2]], max_new_tokens=2, use_cache=invalid)
    result = model.generate_tokens([[2]], max_new_tokens=2)
    assert not result.cached and result.prefix_evaluations == 2
    start = model.backend.prefills
    report = benchmark_branch_generation(
        model, [[2]], max_new_tokens=2, warmup=1, repeats=2
    )
    assert model.backend.prefills == start + 6
    assert len(report.samples_seconds) == 2
    assert report.p95_seconds >= report.median_seconds > 0


def test_external_request_routing_does_not_execute_response_suffix():
    model = make_model(router=True)
    route = model.route_request([[2, 3]])
    assert route.branch_id == "specialist"
    assert model.backend.prefills == 1 and model.backend.suffixes == 1
    model.set_selection(FixedBranchSelection("base"))
    route = model.route_request([[2, 3]])
    assert route.branch_id == "base"
    assert model.backend.suffixes == 1
