"""Offline shared-prefix Needle workflow using tiny random native Needle weights.

Install the optional dependency set: python -m pip install -e ".[needle]".
Runs without model downloads. Temporary deployment artifacts are removed on exit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from anchored_moe.branching import (
    ArgmaxSelection,
    BranchConfig,
    BranchedModel,
    BranchedModelConfig,
    BranchTrainer,
    JaxAdamStrategy,
    LinearRequestScorer,
    NeedleBranchBackend,
    ResponseTrainingExample,
    RouterTrainingExample,
    benchmark_branch_request,
    calibrate_utility_selection,
    load_branch_package,
    save_branch_package,
)


def main() -> None:
    """Check native equivalence, train isolated components, calibrate and reload."""
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig

    native_config = TransformerConfig(
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
    )
    ids = np.array([[2, 5, 6, 7]], np.int32)
    native = SimpleAttentionNetwork(native_config)
    params = native.init(jax.random.PRNGKey(5), jnp.asarray(ids))["params"]
    backend = NeedleBranchBackend(
        params,
        native_config,
        shared_prefix_layers=2,
        model_id="tiny-needle",
        revision="seed-5",
        tokenizer_id="synthetic-v1",
    )
    config = BranchedModelConfig(
        shared_prefix_layers=2,
        response_branches=(
            BranchConfig("base", "frozen"),
            BranchConfig("specialist", rank=2),
        ),
        router_branch=BranchConfig("router", rank=2),
    )
    scorer = LinearRequestScorer(
        ("specialist",), np.ones((16, 1), np.float32) * 0.01, [0]
    )
    model = BranchedModel(backend, config, scorer, ArgmaxSelection())
    base = np.asarray(model.forward_branch(ids, "base"))
    error = float(np.max(np.abs(base - np.asarray(backend.original_logits(ids)))))
    np.testing.assert_allclose(base, backend.original_logits(ids), atol=2e-6, rtol=2e-5)
    trainer = BranchTrainer(JaxAdamStrategy(steps=2, learning_rate=0.005))
    specialist_report = trainer.train_specialist(
        model,
        "specialist",
        [ResponseTrainingExample(ids, [[-100, -100, 6, 7]])],
        data_id="synthetic-specialist-fit",
    )
    # This target is explicitly synthetic. Real routing targets should come from
    # held-out task outcomes of base and specialist, with response-free inputs.
    router_report = trainer.train_router(
        model,
        [RouterTrainingExample([[2, 5]], np.array([1.0], np.float32))],
        data_id="synthetic-router-fit",
    )
    policy = calibrate_utility_selection(
        [[0], [1]],
        [[-1], [1]],
        ("specialist",),
        composition_key=model.composition_key,
        data_id="synthetic-calibration",
        max_false_activation_rate=0,
    )
    model.set_selection(policy)
    np.testing.assert_array_equal(model.forward_branch(ids, "base"), base)
    with TemporaryDirectory(prefix="needle-branches-") as directory:
        package = save_branch_package(model, Path(directory) / "composition")
        restored = load_branch_package(package, backend)
        np.testing.assert_array_equal(
            model.forward(ids).logits, restored.forward(ids).logits
        )
        assert model.forward(ids).route == restored.forward(ids).route
    latency = benchmark_branch_request(model, ids, repeats=3)
    print(
        json.dumps(
            {
                "native_max_absolute_error": error,
                "specialist": asdict(specialist_report),
                "router": asdict(router_report),
                "latency": asdict(latency),
                "validation": "BRN-001 through BRN-006 passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
