"""Offline acceptance tests against actual Needle layers, using tiny random weights."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("needle.model.architecture")
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

from anchored_moe.branching import (
    ArgmaxSelection,
    BranchConfig,
    BranchedModel,
    BranchedModelConfig,
    BranchTrainer,
    FixedBranchSelection,
    JaxAdamStrategy,
    LinearRequestScorer,
    NeedleBranchBackend,
    ResponseObjective,
    ResponseTrainingExample,
    RouterTrainingExample,
    calibrate_utility_selection,
    export_needle_branch,
    export_needle_bundle,
    load_branch_package,
    save_branch_package,
)
from anchored_moe.branching.model import tensor_digest


@pytest.fixture(scope="module")
def native():
    config = TransformerConfig(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        num_kv_heads=1,
        qk_head_dim=8,
        v_head_dim=8,
        num_layers=4,
        mhc_lanes=2,
        engram_layers=(1, 3),
        engram_slots=16,
        engram_heads=1,
        max_seq_len=32,
        sliding_window=3,
        global_layers=(3,),
        flash=False,
        remat=False,
        dtype="float32",
        qkv_conv_taps=3,
    )
    model = SimpleAttentionNetwork(config)
    params = model.init(jax.random.PRNGKey(5), jnp.array([[2, 5, 6, 7]]))["params"]
    return config, params


def backend(native, split=2):
    config, params = native
    return NeedleBranchBackend(
        params,
        config,
        shared_prefix_layers=split,
        model_id="tiny-needle",
        revision="seed-5",
        tokenizer_id="synthetic-v1",
    )


def owner(native, *, router=True, copied=False, probability=False):
    b = backend(native)
    cfg = BranchedModelConfig(
        2,
        (
            BranchConfig("base", "frozen"),
            BranchConfig("specialist", "copy" if copied else "lora", rank=2),
        ),
        BranchConfig("router", "lora", rank=2) if router else None,
    )
    ids = ("base", "specialist") if probability else ("specialist",)
    scorer = LinearRequestScorer(
        ids,
        np.ones((b.feature_size, len(ids)), np.float32) * 0.01,
        np.zeros(len(ids), np.float32),
        kind="probability" if probability else "utility",
    )
    return BranchedModel(b, cfg, scorer, ArgmaxSelection())


@pytest.mark.parametrize("split", [1, 2, 3])
def test_native_equivalence_including_non_lane_aligned_split(native, split):
    b = backend(native, split)
    ids = np.array([[2, 5, 6, 7, 8]], np.int32)
    state = b.prefill(ids)
    actual = b.logits(state, b.suffix_parameters)
    np.testing.assert_allclose(actual, b.original_logits(ids), rtol=2e-5, atol=2e-6)
    snapshot = np.array(state.payload.stream)
    b.features(state, b.suffix_parameters)
    np.testing.assert_array_equal(state.payload.stream, snapshot)
    other = backend(native, 1 if split != 1 else 2)
    with pytest.raises(ValueError, match="incompatible"):
        other.logits(state, other.suffix_parameters)


def test_zero_lora_and_full_copy_are_exact_split_base(native):
    for copy in (False, True):
        model = owner(native, copied=copy)
        ids = np.array([[2, 4, 7]], np.int32)
        np.testing.assert_array_equal(
            model.forward_branch(ids, "specialist"), model.forward_branch(ids, "base")
        )
    b = backend(native)
    for bad in ([[0, 2]], [[2.5]], [[-1]], [[32]], [[2], [3]]):
        with pytest.raises(ValueError):
            b.prefill(bad)


@pytest.mark.parametrize("copied", [False, True])
def test_specialist_training_isolated_and_roundtrips(native, copied, tmp_path):
    model = owner(native, copied=copied)
    ids = np.array([[2, 5, 6, 7]], np.int32)
    labels = np.array([[-100, -100, 6, 7]], np.int32)
    base = np.array(model.forward_branch(ids, "base"))
    router_before = tensor_digest(model.branch_parameters("router"))
    specialist_before = tensor_digest(model.branch_parameters("specialist"))
    report = BranchTrainer(
        JaxAdamStrategy(steps=2, learning_rate=0.005)
    ).train_specialist(
        model,
        "specialist",
        [ResponseTrainingExample(ids, labels, base)],
        data_id="specialist-train",
        objective=ResponseObjective(replay_kl_weight=0.1),
        drift_weight=0.01,
    )
    assert report.final_loss < report.initial_loss
    assert report.trainable_parameters == sum(
        v.size for v in model.branch_parameters("specialist").values()
    )
    assert tensor_digest(model.branch_parameters("specialist")) != specialist_before
    assert tensor_digest(model.branch_parameters("router")) == router_before
    np.testing.assert_array_equal(model.forward_branch(ids, "base"), base)
    path = save_branch_package(model, tmp_path / "trained")
    reload = load_branch_package(path, backend(native))
    np.testing.assert_array_equal(
        reload.forward_branch(ids, "specialist"),
        model.forward_branch(ids, "specialist"),
    )


@pytest.mark.parametrize(
    "router,probability", [(False, False), (True, False), (True, True)]
)
def test_router_training_updates_only_head_and_optional_suffix(
    native, router, probability
):
    model = owner(native, router=router, probability=probability)
    specialist = tensor_digest(model.branch_parameters("specialist"))
    head = tensor_digest(model.scorer.parameters)
    router_before = tensor_digest(model.branch_parameters("router")) if router else None
    examples = [
        RouterTrainingExample(
            np.array([[2, 5, 6]], np.int32),
            1 if probability else np.array([1.0], np.float32),
        )
    ]
    report = BranchTrainer(JaxAdamStrategy(steps=2, learning_rate=0.01)).train_router(
        model, examples, data_id="router-fit"
    )
    assert report.final_loss < report.initial_loss
    assert tensor_digest(model.scorer.parameters) != head
    assert tensor_digest(model.branch_parameters("specialist")) == specialist
    if router:
        assert tensor_digest(model.branch_parameters("router")) != router_before
    scores = model.scorer.score(np.zeros(16))
    assert scores.kind == ("probability" if probability else "utility")
    if not probability:
        policy = calibrate_utility_selection(
            [[0], [1]],
            [[-1], [1]],
            ("specialist",),
            composition_key=model.composition_key,
            data_id="router-fit",
        )
        with pytest.raises(ValueError, match="disjoint"):
            model.set_selection(policy)


def test_failed_training_publishes_nothing_and_owner_recovers(native):
    model = owner(native)
    before = model.composition_key

    class FailingOptimizer:
        def optimize(self, parameters, loss_fn):
            # Materialize a real loss before simulating a failed optimizer.
            float(loss_fn(parameters))
            raise RuntimeError("injected optimizer failure")

    examples = [ResponseTrainingExample([[2, 5, 6]], [[-100, 5, 6]])]
    with pytest.raises(RuntimeError, match="injected"):
        BranchTrainer(FailingOptimizer()).train_specialist(
            model, "specialist", examples, data_id="bad-run"
        )
    assert model.composition_key == before
    assert "bad-run" not in model._training_data_ids
    model.forward([[2, 5]])
    with pytest.raises(ValueError, match="valid"):
        BranchTrainer().train_specialist(
            model,
            "specialist",
            [ResponseTrainingExample([[2, 5]], [[-100, -100]])],
            data_id="empty-target",
        )


def test_greedy_reference_generation_locks_original_branch(native):
    model = owner(native)
    model.set_selection(FixedBranchSelection("base"))
    ids = np.array([[2, 5, 6]], np.int32)
    expected, sequence = [], ids
    for _ in range(3):
        logits = np.asarray(model.backend.original_logits(sequence))[0, -1].copy()
        logits[0] = -np.inf
        token = int(logits.argmax())
        expected.append(token)
        sequence = np.concatenate((sequence, [[token]]), axis=1)
    result = model.generate_tokens(ids, max_new_tokens=3)
    assert result.token_ids == tuple(expected)
    assert result.route.branch_id == "base"


def test_master_checkpoint_loading_roundtrips_and_rejects_pickle(native, tmp_path):
    from needle.model.checkpoints import write_checkpoint

    config, params = native
    path = tmp_path / "master.safetensors"
    write_checkpoint(
        path, {"format_version": 2, "config": asdict(config), "params": params}
    )
    identity = {
        "shared_prefix_layers": 2,
        "model_id": "tiny-needle",
        "revision": "seed-5",
        "tokenizer_id": "synthetic-v1",
    }
    loaded = NeedleBranchBackend.from_checkpoint(path, **identity)
    original = backend(native)
    assert loaded.identity == original.identity
    np.testing.assert_array_equal(
        loaded.logits(loaded.prefill([[2, 3]]), loaded.suffix_parameters),
        original.logits(original.prefill([[2, 3]]), original.suffix_parameters),
    )
    bad = tmp_path / "untrusted.pkl"
    bad.write_bytes(b"not read")
    with pytest.raises(ValueError, match="safetensors"):
        NeedleBranchBackend.from_checkpoint(bad, **identity)
    write_checkpoint(path, {"format_version": 1, "params": params})
    with pytest.raises(ValueError, match="format-v2"):
        NeedleBranchBackend.from_checkpoint(path, **identity)


def test_head_only_training_preserves_router_suffix_and_temperature(native):
    model = owner(native, probability=True)
    params = model.scorer.parameters
    model.set_scorer(
        LinearRequestScorer(
            model.response_ids,
            params["weight"],
            params["bias"],
            kind="probability",
            temperature=2.5,
        )
    )
    suffix_before = tensor_digest(model.branch_parameters("router"))
    examples = [RouterTrainingExample([[2, 5, 6]], 1)]
    report = BranchTrainer(JaxAdamStrategy(steps=2, learning_rate=0.01)).train_router(
        model, examples, data_id="head-fit", train_suffix=False
    )
    assert report.final_loss < report.initial_loss
    assert report.trainable_parameters == 34
    assert tensor_digest(model.branch_parameters("router")) == suffix_before
    assert model.scorer.temperature == 2.5
    features = model.backend.features(
        model.backend.prefill(examples[0].input_ids), model.effective_suffix("router")
    )
    assert report.final_loss == pytest.approx(
        -np.log(model.scorer.score(features).values[1]), rel=1e-5
    )


def test_generation_masks_padding_and_respects_context_capacity(native):
    model = owner(native)
    model.set_selection(FixedBranchSelection("base"))

    def prefer_padding(ids, logits):
        logits[:] = 0
        logits[0] = 100
        logits[3] = 10
        return logits

    assert model.generate_tokens(
        [[2, 5]], max_new_tokens=2, logits_processor=prefer_padding
    ).token_ids == (3, 3)
    with pytest.raises(ValueError, match="context limit"):
        model.generate_tokens([[2, 5]], max_new_tokens=31)
    model.forward([[2, 5]])


def test_native_bfloat16_flash_without_engram_and_with_asymmetric_heads(native):
    config = replace(
        native[0],
        dtype="bfloat16",
        flash=True,
        engram_layers=(),
        sliding_window=0,
        qkv_conv_taps=0,
        v_head_dim=16,
    )
    ids = jnp.array([[2, 5, 6]])
    params = SimpleAttentionNetwork(config).init(jax.random.PRNGKey(7), ids)["params"]
    b = backend((config, params), split=1)
    np.testing.assert_allclose(
        b.logits(b.prefill(ids), b.suffix_parameters),
        b.original_logits(ids),
        rtol=0.02,
        atol=0.002,
    )
    _, state = b.start_decode(b.prefill_cached(ids, capacity=8), b.suffix_parameters)
    logits, _ = b.decode_step(state, 7)
    np.testing.assert_allclose(
        logits[:, -1],
        b.original_logits(jnp.array([[2, 5, 6, 7]]))[:, -1],
        rtol=0.02,
        atol=0.002,
    )


@pytest.mark.parametrize("split,prompt_length", [(1, 1), (2, 5), (3, 14)])
def test_cached_logits_match_reference_with_real_convolution_history(
    native, split, prompt_length
):
    from flax.core import unfreeze

    config, original = native
    params = unfreeze(original)
    rng = np.random.default_rng(44)
    for name in ("q", "k", "v"):
        values = params["stack"]["layers"]["block"]["self_attn"][name + "_taps"]
        params["stack"]["layers"]["block"]["self_attn"][name + "_taps"] = (
            values + jnp.asarray(rng.normal(0, 0.15, values.shape), jnp.float32)
        )
    for site in (0, 1):
        values = params[f"engrams_{site}"]["taps"]
        params[f"engrams_{site}"]["taps"] = values + jnp.asarray(
            rng.normal(0, 0.2, values.shape), jnp.float32
        )
    b = backend((config, params), split)
    sequence = np.array([[2 + i % 20 for i in range(prompt_length)]], np.int32)
    prefix = b.prefill_cached(sequence, capacity=prompt_length + 4)
    logits, state = b.start_decode(prefix, b.suffix_parameters)
    np.testing.assert_allclose(
        logits, b.original_logits(sequence), rtol=2e-5, atol=2e-6
    )
    assert state.cache_lengths == (3, 3, 3, prompt_length + 4)
    initial_state = state
    for token in (21, 22, 23, 24):
        logits, state = b.decode_step(state, token)
        sequence = np.concatenate((sequence, [[token]]), axis=1)
        np.testing.assert_allclose(
            logits[:, -1], b.original_logits(sequence)[:, -1], rtol=2e-5, atol=2e-6
        )
    assert initial_state.token_count == prompt_length
    with pytest.raises(ValueError, match="exhausted"):
        b.decode_step(state, 2)
    fork, _ = b.decode_step(initial_state, 21)
    expected = b.original_logits(sequence[:, : prompt_length + 1])
    np.testing.assert_allclose(fork[:, -1], expected[:, -1], rtol=2e-5, atol=2e-6)


def test_cached_generation_reuses_router_and_keeps_branches_isolated(
    native, monkeypatch
):
    model = owner(native, copied=True)
    original = model.backend.prefill_cached
    calls = []

    def prefill(ids, *, capacity):
        calls.append(np.shape(ids))
        return original(ids, capacity=capacity)

    monkeypatch.setattr(model.backend, "prefill_cached", prefill)
    params = model.branch_parameters("specialist")
    key = next(k for k in params if k.endswith("q_proj/kernel"))
    params[key] += 0.1
    model.replace_branch_parameters("specialist", params)
    cached = model.generate_tokens([[2, 5, 6]], max_new_tokens=4)
    reference = model.generate_tokens([[2, 5, 6]], max_new_tokens=4, use_cache=False)
    assert cached.route == reference.route
    assert cached.token_ids == reference.token_ids
    assert cached.cached and cached.prefix_evaluations == 1
    assert not reference.cached and reference.prefix_evaluations == 4
    assert calls == [(1, 3)]
    prefix = model.backend.prefill_cached([[2, 5]], capacity=6)
    _, base = model.backend.start_decode(prefix, model.effective_suffix("base"))
    _, specialist = model.backend.start_decode(
        prefix, model.effective_suffix("specialist")
    )
    for branch, state in (("base", base), ("specialist", specialist)):
        logits, _ = model.backend.decode_step(state, 7)
        np.testing.assert_allclose(
            logits[:, -1],
            model.forward_branch([[2, 5, 7]], branch)[:, -1],
            rtol=2e-5,
            atol=2e-6,
        )
    with pytest.raises(ValueError, match="incompatible"):
        backend(native, split=1).decode_step(base, 7)


def _tokenizer_blob():
    from needle.model.tokenizer import CHAT_MARKERS

    pieces = ["<pad>", "</s>", "<s>", "<unk>", *CHAT_MARKERS] + [
        chr(i) for i in range(65, 77)
    ]
    header = struct.pack("<IIIIIBBH", len(pieces), 0, 1, 2, 3, 0, 0, 0)
    return header + b"".join(
        struct.pack(
            "<fBH",
            0,
            2 if i < 3 else 1 if i == 3 else 3 if i < 20 else 0,
            len(p.encode()),
        )
        + p.encode()
        for i, p in enumerate(pieces)
    )


@pytest.mark.parametrize("copied", [False, True])
def test_native_export_merges_only_selected_suffix_and_publishes_standard_archive(
    native, tmp_path, copied
):
    from needle.model.export import read_export, read_tokenizer_blob

    model = owner(native, copied=copied)
    original = model.composition_key
    params = model.branch_parameters("specialist")
    key = next(k for k in params if k.endswith("q_proj/kernel" if copied else "/B"))
    params[key] += 0.2
    model.replace_branch_parameters("specialist", params)
    changed = model.composition_key
    assert changed != original
    report = export_needle_bundle(
        model, tmp_path / "native", tokenizer=_tokenizer_blob()
    )
    assert model.composition_key == changed
    base_geometry, base = read_export(report.branches[0].path)
    geometry, tuned = read_export(report.branches[1].path)
    assert geometry["num_layers"] == base_geometry["num_layers"] == 4
    assert read_tokenizer_blob(report.branches[1].path) == _tokenizer_blob()
    # The unchanged prefix's native records must survive export byte-for-byte.
    # Count actual records through layer 1 from upstream's canonical tensor walk.
    from needle.model.export import _tensors

    from anchored_moe.branching.native_export import _merged_parameters

    named = _tensors(_merged_parameters(model, "base")[0], native[0], 4, 128)
    different = []
    for entry, original_value, new_value in zip(named, base, tuned):
        if entry.name.startswith(("embedding", "layer00.", "layer01.")):
            np.testing.assert_array_equal(original_value, new_value)
        if not np.array_equal(original_value, new_value):
            different.append(entry.name)
    assert different and all(
        name.startswith(("layer02.", "layer03.")) for name in different
    )
    manifest = json.loads((report.path / "manifest.json").read_text())
    assert manifest["routing"]["native_prefix_sharing"] is False
    assert manifest["routing"]["native_calibration"] == "requires_evaluation"
    with pytest.raises(FileExistsError):
        export_needle_bundle(model, report.path, tokenizer=_tokenizer_blob())
    single = export_needle_branch(
        model, "specialist", tmp_path / "single.cact", tokenizer=report.branches[0].path
    )
    assert single.sha256 == report.branches[1].sha256
    with pytest.raises(FileExistsError):
        export_needle_branch(
            model, "specialist", single.path, tokenizer=_tokenizer_blob()
        )


def test_native_export_failure_is_atomic_and_rejects_malformed_tokenizer(
    native, tmp_path, monkeypatch
):
    from needle.model import export

    model = owner(native)
    target = tmp_path / "native"
    with pytest.raises(ValueError, match="Malformed"):
        export_needle_bundle(model, target, tokenizer=b"bad")
    assert not target.exists()

    original_reader = export.read_export

    def mismatched(path):
        geometry, tensors = original_reader(path)
        geometry["engram_orders"] = (2,)
        return geometry, tensors

    with monkeypatch.context() as changed:
        changed.setattr(export, "read_export", mismatched)
        with pytest.raises(ValueError, match="roundtrip"):
            export_needle_bundle(model, target, tokenizer=_tokenizer_blob())
    assert list(tmp_path.iterdir()) == []

    def fail(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(export, "write_export", fail)
    with pytest.raises(RuntimeError, match="injected"):
        export_needle_bundle(model, target, tokenizer=_tokenizer_blob())
    assert list(tmp_path.iterdir()) == []
    model.forward([[2, 5]])


def test_native_engine_loads_export_when_local_library_supplied(native, tmp_path):
    library = os.environ.get("ANCHORED_MOE_NEEDLE_LIBRARY")
    if not library:
        pytest.skip(
            "Set ANCHORED_MOE_NEEDLE_LIBRARY to test an installed native engine offline."
        )
    # The native SIMD kernels reject the width-16 toy geometry used by JAX
    # tests. Exercise a supported width/head/lane geometry without real weights.
    config = replace(
        native[0],
        d_model=128,
        qk_head_dim=64,
        v_head_dim=64,
        mhc_lanes=4,
        max_seq_len=512,
        sliding_window=16,
    )
    params = SimpleAttentionNetwork(config).init(
        jax.random.PRNGKey(9), jnp.array([[2, 5]])
    )["params"]
    report = export_needle_branch(
        owner((config, params)),
        "specialist",
        tmp_path / "native.cact",
        tokenizer=_tokenizer_blob(),
    )
    script = (
        "import ctypes, json, pathlib, sys; "
        "lib = ctypes.CDLL(sys.argv[1]); "
        "lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]; "
        "lib.needle_load.restype = ctypes.c_int; "
        "lib.needle_last_error.restype = ctypes.c_char_p; "
        "data = pathlib.Path(sys.argv[2]).read_bytes(); "
        "result = lib.needle_load(data, len(data)); "
        "assert result >= 0, lib.needle_last_error(); "
        "lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]; "
        "lib.needle_init.restype = ctypes.c_int; "
        "tools = json.dumps([{'name': 'A', 'description': 'A', 'parameters': {'type': 'object', 'properties': {}}}]).encode(); "
        "assert lib.needle_init(b'', tools, None) >= 0, lib.needle_last_error(); "
        "lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]; "
        "lib.needle_complete.restype = ctypes.c_int; "
        "buffer = ctypes.create_string_buffer(8192); "
        "assert lib.needle_complete(b'A', 2, buffer, len(buffer)) >= 0, lib.needle_last_error(); "
        "assert isinstance(json.loads(buffer.value), dict); "
        "print('native load, prefill and decode passed')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, library, str(report.path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
