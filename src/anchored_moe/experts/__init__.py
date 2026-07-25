"""Expert representations for immutable-anchor specialization."""

from anchored_moe.experts.base import Expert
from anchored_moe.experts.full_mlp import (
    FullMLPExpert,
    ExpertSerializationError,
)

__all__ = [
    "Expert",
    "ExpertSerializationError",
    "FullMLPExpert",
]
