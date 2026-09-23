"""Offline acceptance tests against actual Needle layers, using tiny random weights."""

from __future__ import annotations

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
        np.ones((16, len(ids)), np.float32) * 0.01,
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
