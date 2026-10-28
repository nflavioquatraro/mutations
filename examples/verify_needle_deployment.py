"""Offline .cact export, KV-cache parity and warm generation timing.

Install anchored-moe[needle]. This uses random weights and a synthetic tokenizer;
it measures implementation behavior, not Needle quality. No downloads occur.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from anchored_moe.branching import (
    ArgmaxSelection,
    BranchConfig,
    BranchedModel,
    BranchedModelConfig,
    LinearRequestScorer,
    NeedleBranchBackend,
    benchmark_branch_generation,
    export_needle_bundle,
)


def main() -> None:
    """Verify cached/reference parity, export two native models and compare latency."""
    import jax
    import jax.numpy as jnp
    from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
    from needle.model.tokenizer import CHAT_MARKERS

    config = TransformerConfig(
        vocab_size=32,
        d_model=128,
        num_heads=2,
        num_kv_heads=1,
        qk_head_dim=64,
        v_head_dim=64,
        num_layers=2,
        mhc_lanes=4,
        engram_layers=(1,),
        engram_slots=32,
        engram_heads=1,
        max_seq_len=512,
        sliding_window=16,
        global_layers=(1,),
        flash=False,
        remat=False,
        dtype="float32",
        qkv_conv_taps=3,
    )
    ids = np.array([[2, 20, 21, 22]], np.int32)
    parameters = SimpleAttentionNetwork(config).init(
        jax.random.PRNGKey(2), jnp.asarray(ids)
    )["params"]
    backend = NeedleBranchBackend(
        parameters,
        config,
        shared_prefix_layers=1,
        model_id="synthetic-native-geometry",
        revision="seed-2",
        tokenizer_id="synthetic-32",
    )
    model = BranchedModel(
        backend,
        BranchedModelConfig(
            1,
            (BranchConfig("base", "frozen"), BranchConfig("specialist", rank=2)),
            router_branch=BranchConfig("router", rank=2),
        ),
        LinearRequestScorer(("specialist",), np.zeros((128, 1)), [1]),
        ArgmaxSelection(),
    )
    learned = model.branch_parameters("specialist")
    for name in learned:
        if name.endswith("/B"):
            learned[name] += 0.02
    model.replace_branch_parameters("specialist", learned)
    cached = model.generate_tokens(ids, max_new_tokens=6, use_cache=True)
    reference = model.generate_tokens(ids, max_new_tokens=6, use_cache=False)
    assert cached.route == reference.route and cached.token_ids == reference.token_ids
    assert cached.prefix_evaluations == 1 and reference.prefix_evaluations == 6
    # A production export receives its tokenizer from the original .cact archive.
    # This synthetic fixture includes the normal chat marker IDs for engine loading.
    pieces = ["<pad>", "</s>", "<s>", "<unk>", *CHAT_MARKERS] + [
        chr(i) for i in range(65, 77)
    ]
    blob = struct.pack("<IIIIIBBH", len(pieces), 0, 1, 2, 3, 0, 0, 0)
    for index, piece in enumerate(pieces):
        kind = 2 if index < 3 else 1 if index == 3 else 3 if index < 20 else 0
        encoded = piece.encode("utf-8")
        blob += struct.pack("<fBH", 0, kind, len(encoded)) + encoded
    with TemporaryDirectory(prefix="needle-native-") as directory:
        bundle = export_needle_bundle(
            model, Path(directory) / "deployment", tokenizer=blob
        )
        native_bytes = {r.branch_id: r.size_bytes for r in bundle.branches}
        manifest = json.loads((bundle.path / "manifest.json").read_text())
        assert manifest["routing"]["native_prefix_sharing"] is False
    timing = {
        name: benchmark_branch_generation(
            model, ids, max_new_tokens=6, use_cache=enabled, repeats=2
        )
        for name, enabled in (("cached", True), ("uncached", False))
    }
    print(
        json.dumps(
            {
                "tokens": cached.token_ids,
                "prefix_evaluations": {
                    "cached": cached.prefix_evaluations,
                    "uncached": reference.prefix_evaluations,
                },
                "native_bytes": native_bytes,
                "warm_generation_seconds": {
                    name: asdict(report) for name, report in timing.items()
                },
                "observed_speedup": timing["uncached"].median_seconds
                / timing["cached"].median_seconds,
                "scope": "Synthetic CPU validation; .cact responses use external routing, not native prefix sharing.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
