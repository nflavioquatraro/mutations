"""One prompt decision per response with explicit immutable-base fallback."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from anchored_moe.adapters.model import AdapterSpecializationModel


@dataclass(frozen=True)
class PromptMessage:
    """Immutable role/text message supplied to an external prompt classifier."""

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validate textual conversation messages without tokenizer assumptions."""

        if self.role not in {"system", "user", "assistant", "tool"} or not isinstance(
            self.content, str
        ):
            raise ValueError("Messages require a supported role and string content.")


@dataclass(frozen=True)
class AdapterRoute:
    """Selected adapter (None means base), with optional confidence in [0, 1]."""

    adapter_id: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed route IDs and nonfinite classifier confidence."""

        if self.adapter_id is not None and (
            not isinstance(self.adapter_id, str) or not self.adapter_id
        ):
            raise ValueError("adapter_id must be a nonempty string or None.")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and in [0, 1].")


class PromptAdapterRouter(Protocol):
    """Extension point for classifiers/retrievers; never sees the target answer.

    Implement select(messages) returning AdapterRoute. Training this component
    is independent of adapter training. Inputs are immutable history snapshots.
    """

    def select(self, messages: tuple[PromptMessage, ...]) -> AdapterRoute:
        """Choose base or one adapter from the provided pre-response history."""
        ...


@dataclass(frozen=True)
class FixedAdapterRouter:
    """Explicit base/specialist routing for oracle evaluation and debugging."""

    route: AdapterRoute = AdapterRoute()

    def select(self, messages: tuple[PromptMessage, ...]) -> AdapterRoute:
        """Return the configured route without examining conversation content."""

        return self.route


@dataclass(frozen=True)
class PromptRoutingConfig:
    """Configure classifier history and confidence fallback.

    min_confidence: Named routes below this threshold use base. A positive
        threshold requires a confidence value; unknown adapters always fail.
    history_messages: Optional count of latest messages passed to the router;
        None includes the complete supplied history. Generation inputs are
        caller-tokenized separately and are never truncated by this setting.
    """

    min_confidence: float = 0.0
    history_messages: int | None = None

    def __post_init__(self) -> None:
        """Validate finite thresholds and positive optional history bounds."""

        if self.min_confidence is None:
            raise ValueError("min_confidence must be a finite threshold, not None.")
        AdapterRoute(confidence=self.min_confidence)
        if self.history_messages is not None and (
            isinstance(self.history_messages, bool)
            or not isinstance(self.history_messages, int)
            or self.history_messages <= 0
        ):
            raise ValueError("history_messages must be a positive integer or None.")


@dataclass(frozen=True)
class PromptGenerationResult:
    """Requested/effective routes and native Hugging Face generation output."""

    requested_route: AdapterRoute
    effective_route: AdapterRoute
    output: Any


class PromptRoutedAdapterGenerator:
    """Generate a whole response under one selected adapter or the frozen base.

    One call represents one response, not a session. Call again with updated
    history for subsequent turns. The wrapper never reuses caches across calls
    and rejects externally supplied caches. Dynamic caching inside generation
    is allowed. Persistent/static cache implementations are rejected explicitly.
    Requests sharing an owner cannot execute concurrently. Tokenization and
    ensuring messages match the encoded prompt are caller responsibilities.
    """

    def __init__(
        self,
        specialization: AdapterSpecializationModel,
        router: PromptAdapterRouter,
        config: PromptRoutingConfig | None = None,
    ) -> None:
        """Bind model ownership, external routing strategy, and fallback policy."""

        self.specialization = specialization
        self.router = router
        self.config = config or PromptRoutingConfig()

    def generate(
        self, messages: Sequence[PromptMessage], **generation_kwargs: Any
    ) -> PromptGenerationResult:
        """Classify once, generate with fresh cache, and restore adapter state.

        Raises ValueError for empty/invalid history, unknown routes, missing
        required confidence, or reusable cache inputs. Exceptions release the
        owner and restore frozen parameters and previous execution mode.
        """

        history = tuple(messages)
        if not history or any(not isinstance(m, PromptMessage) for m in history):
            raise ValueError("Provide a nonempty sequence of PromptMessage objects.")
        if any(
            generation_kwargs.get(k) is not None
            for k in ("past_key_values", "cache_position", "past")
        ):
            raise ValueError(
                "Cross-call caches are forbidden; re-encode the conversation."
            )
        generation_config = generation_kwargs.get(
            "generation_config", self.specialization.model.generation_config
        )
        cache = generation_kwargs.get(
            "cache_implementation",
            getattr(generation_config, "cache_implementation", None),
        )
        if cache not in (None, "dynamic"):
            raise ValueError("Only fresh dynamic generation caches are supported.")
        if self.config.history_messages is not None:
            history = history[-self.config.history_messages :]
        requested = self.router.select(history)
        if not isinstance(requested, AdapterRoute):
            raise TypeError("The router must return AdapterRoute.")
        if (
            requested.adapter_id is not None
            and requested.adapter_id not in self.specialization.adapter_ids
        ):
            raise ValueError("Router selected an unknown adapter.")
        effective = requested
        if requested.adapter_id is not None and self.config.min_confidence > 0:
            if requested.confidence is None:
                raise ValueError("Confidence is required by the fallback policy.")
            if requested.confidence < self.config.min_confidence:
                effective = AdapterRoute(None, requested.confidence)
        with (
            self.specialization.activate(effective.adapter_id) as model,
            torch.inference_mode(),
        ):
            # Explicitly override any native config to keep cache ownership
            # inside this response even after a previous caller changed defaults.
            generation_kwargs["cache_implementation"] = "dynamic"
            output = model.generate(**generation_kwargs)
        return PromptGenerationResult(requested, effective, output)
