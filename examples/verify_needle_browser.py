"""Write a synthetic base-plus-patch bundle for offline browser runtime checks.

Usage: python examples/verify_needle_browser.py --output .cache/browser-demo
Random weights test transport/execution, not router quality. No downloads occur.
Copy pinned official needle.js/needle.wasm into OUTPUT/engine, then serve OUTPUT
over localhost and open index.html. Real usage trains labeled router examples first.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anchored_moe.branching import (
    BranchConfig,
    BranchedModel,
    BranchedModelConfig,
    FixedBranchSelection,
    GenerativeRouterConfig,
    LinearRequestScorer,
    NeedleBranchBackend,
    export_needle_browser_bundle,
)


def build_demo(output: Path):
    """Export tiny native-supported geometry with independent nonzero suffixes."""
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
        max_seq_len=2048,
        sliding_window=16,
        global_layers=(1,),
        flash=False,
        remat=False,
        dtype="float32",
        qkv_conv_taps=3,
    )
    params = SimpleAttentionNetwork(config).init(
        jax.random.PRNGKey(2), jnp.array([[2, 20, 21]])
    )["params"]
    backend = NeedleBranchBackend(
        params,
        config,
        shared_prefix_layers=1,
        model_id="synthetic-browser",
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
        LinearRequestScorer(("specialist",), np.zeros((128, 1)), [0]),
        FixedBranchSelection("base"),
    )
    for branch, delta in (("specialist", 0.03), ("router", 0.02)):
        tensors = model.branch_parameters(branch)
        for name in tensors:
            if name.endswith("/B"):
                tensors[name] += delta
        model.replace_branch_parameters(branch, tensors)
    pieces = ["<pad>", "</s>", "<s>", "<unk>", *CHAT_MARKERS] + [
        chr(i) for i in range(65, 77)
    ]
    blob = struct.pack("<IIIIIBBH", len(pieces), 0, 1, 2, 3, 0, 0, 0)
    for index, piece in enumerate(pieces):
        kind = 2 if index < 3 else 1 if index == 3 else 3 if index < 20 else 0
        encoded = piece.encode()
        blob += struct.pack("<fBH", 0.0, kind, len(encoded)) + encoded
    report = export_needle_browser_bundle(
        model,
        output,
        router=GenerativeRouterConfig(model.response_ids, max_new_tokens=8),
        tokenizer=blob,
    )
    shutil.copyfile(
        Path(__file__).with_name("needle_browser.html"), report.path / "index.html"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asdict(build_demo(args.output)), default=str, indent=2))
