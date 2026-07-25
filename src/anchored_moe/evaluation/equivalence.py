"""Dense-equivalence and immutable-anchor validation utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TensorEquivalence:
    """Numerical comparison summary for two tensors with identical shapes."""

    exact: bool
    max_absolute_difference: float
    max_relative_difference: float
    element_count: int


def compare_tensors(actual: Tensor, expected: Tensor) -> TensorEquivalence:
    """Measure exact and numerical differences without changing either tensor."""

    if actual.shape != expected.shape:
        raise ValueError(
            f"Tensor shapes differ: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}."
        )

    actual_detached = actual.detach()
    expected_detached = expected.detach()
    exact = torch.equal(actual_detached, expected_detached)
    if actual_detached.numel() == 0:
        return TensorEquivalence(
            exact=exact,
            max_absolute_difference=0.0,
            max_relative_difference=0.0,
            element_count=0,
        )

    # Float64 avoids losing diagnostic precision when the model uses a lower
    # precision dtype. The comparison itself remains on detached copies.
    actual_float = actual_detached.to(dtype=torch.float64)
    expected_float = expected_detached.to(dtype=torch.float64)
    absolute = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(torch.finfo(torch.float64).tiny)
    relative = absolute / denominator
    return TensorEquivalence(
        exact=exact,
        max_absolute_difference=absolute.max().item(),
        max_relative_difference=relative.max().item(),
        element_count=actual_detached.numel(),
    )


def assert_tensors_equivalent(
    actual: Tensor,
    expected: Tensor,
    *,
    name: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> TensorEquivalence:
    """Assert declared tolerances and return diagnostics for reporting.

    Args:
        actual: Tensor emitted by the converted model.
        expected: Tensor emitted by the original dense model.
        name: Human-readable value name included in assertion errors.
        atol: Absolute tolerance passed to ``torch.testing.assert_close``.
        rtol: Relative tolerance passed to ``torch.testing.assert_close``.
    """

    result = compare_tensors(actual, expected)
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=(
            f"{name} violates dense equivalence: "
            f"max_abs={result.max_absolute_difference}, "
            f"max_rel={result.max_relative_difference}."
        ),
    )
    return result


def fingerprint_module(module: nn.Module) -> str:
    """Return a deterministic SHA-256 digest of module tensors and metadata.

    State-dict keys, tensor shapes, dtypes, and raw tensor bytes are included.
    The function is suitable for proving that an anchor MLP's parameters and
    buffers did not change during conversion.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        # Viewing as bytes also works for dtypes such as bfloat16 that NumPy
        # cannot always represent directly.
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
