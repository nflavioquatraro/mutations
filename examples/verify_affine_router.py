"""Verify frozen identity features and append-only affine router heads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from anchored_moe import (
    AffineRouterHead,
    AppendOnlyRouterHeads,
    IdentityRouterFeatureExtractor,
)


def main() -> None:
    """Score deterministic pre-MLP features, append a head, and reload."""

    torch.manual_seed(37)
    hidden_states = torch.randn(2, 3, 4)
    extractor = IdentityRouterFeatureExtractor(feature_size=4)
    features = extractor(hidden_states)

    router = AppendOnlyRouterHeads(feature_size=extractor.output_size)
    code_head = AffineRouterHead(
        4,
        layer_index=2,
        expert_id="code:0",
        temperature=1.5,
        threshold=0.1,
    )
    with torch.no_grad():
        code_head.linear.weight.copy_(
            torch.tensor([[0.2, -0.1, 0.3, 0.4]])
        )
        code_head.linear.bias.fill_(0.05)
    router.append(code_head)
    original_code_scores = router(features, layer_index=2)

    # Appending a second independently owned head preserves the first column.
    router.append(
        AffineRouterHead(
            4,
            layer_index=2,
            expert_id="math:0",
            threshold=0.2,
        )
    )
    appended_scores = router(features, layer_index=2)
    assert torch.equal(appended_scores[..., :1], original_code_scores)

    with tempfile.TemporaryDirectory(
        prefix="anchored-moe-affine-router-"
    ) as directory:
        payload_path = Path(directory) / "router.safetensors"
        router.save_safetensors(payload_path)
        restored = AppendOnlyRouterHeads.from_safetensors(payload_path)
        restored_scores = restored(features, layer_index=2)
        assert torch.equal(restored_scores, appended_scores)

    print(
        "Affine router validation passed: "
        f"feature_shape={tuple(features.shape)}, "
        f"score_shape={tuple(appended_scores.shape)}, "
        f"head_order={router.expert_ids_for_layer(2)}, "
        "exact_reload=True"
    )


if __name__ == "__main__":
    main()
