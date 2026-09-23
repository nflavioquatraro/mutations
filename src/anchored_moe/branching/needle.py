"""Optional Needle 3 JAX reference backend with a real residual-lane boundary.

References:
    https://github.com/cactus-compute/needle/blob/main/needle/model/architecture.py

Differences:
    Calls the upstream scan body for explicit contiguous layer ranges, retaining
    original layer indices and hyper-connection offsets. Uses floating reference
    numerics only; no native .cact export, quantization or incremental KV cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from anchored_moe.utils.hashing import sha256_file, sha256_json

from .backend import BranchBackend, PrefixState
from .model import tensor_digest


@dataclass(frozen=True)
class _NeedleState:
    stream: Any
    mask: Any
    local_mask: Any
    rope: Any
    engram_kv: Any


class NeedleBranchBackend(BranchBackend):
    """Split the pinned cactus-needle 3.0.4 reference model without porting it.

    parameters: Native unquantized Flax ``params`` tree (not the variables dict).
    model_config: Needle TransformerConfig. Every layer keeps its original index.
    shared_prefix_layers: Count in [1, num_layers - 1].
    model_id, revision, tokenizer_id: Nonempty caller-pinned provenance strings;
        tensor/config digests independently bind actual loaded contents.

    Embeddings, engram tables/projections, final norm and tied output head remain
    shared and frozen. Copied suffixes own block and layer-specific mHC tensors.
    Input is one unpadded request, [1, sequence]; padding is explicitly rejected.
    JAX/Flax/Needle imports happen only when constructing this backend.
    """

    @classmethod
    def from_checkpoint(
        cls,
        path,
        *,
        shared_prefix_layers: int,
        model_id: str,
        revision: str,
        tokenizer_id: str,
    ):
        """Load an existing Needle 3 format-v2 .safetensors master checkpoint.

        This path never downloads files or accepts pickle/.cact input. Identity
        pins are supplied by the caller and checked against actual tensor hashes
        when reloading a composition package.
        """
        path = Path(path)
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("Provide an existing .safetensors master checkpoint.")
        from needle.model.architecture import TransformerConfig
        from needle.model.checkpoints import read_checkpoint

        checkpoint = read_checkpoint(path)
        if checkpoint.get("format_version") != 2 or not checkpoint.get("config"):
            raise ValueError(
                "Expected a Needle 3 format-v2 checkpoint with configuration."
            )
        config = checkpoint["config"]
        if "attn_dim" in config and "qk_head_dim" not in config:
            raise ValueError("Needle 2 checkpoints are unsupported.")
        parameters = {
            k: v for k, v in checkpoint["params"].items() if not k.startswith("mtp_")
        }
        return cls(
            parameters,
            TransformerConfig.from_saved(config),
            shared_prefix_layers=shared_prefix_layers,
            model_id=model_id,
            revision=revision,
            tokenizer_id=tokenizer_id,
        )

    def __init__(
        self,
        parameters,
        model_config,
        *,
        shared_prefix_layers: int,
        model_id: str,
        revision: str,
        tokenizer_id: str,
    ):
        """Snapshot the native base and reject incompatible runtime/topology."""
        try:
            import jax
            import jax.numpy as jnp
            from flax.core import freeze, unfreeze
            from flax.traverse_util import flatten_dict
            from needle.model import architecture as arch
        except ImportError as exc:
            raise ImportError(
                "Install anchored-moe[needle] to use the Needle reference backend."
            ) from exc
        if version("cactus-needle") != "3.0.4":
            raise ValueError(
                "Needle backend currently supports cactus-needle==3.0.4 only."
            )
        cfg = arch.TransformerConfig(**asdict(model_config))
        if (
            isinstance(shared_prefix_layers, bool)
            or not isinstance(shared_prefix_layers, int)
            or not 0 < shared_prefix_layers < cfg.num_layers
        ):
            raise ValueError("The split must leave nonempty prefix and suffix ranges.")
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (model_id, revision, tokenizer_id)
        ):
            raise ValueError(
                "Pinned model, revision and tokenizer identities are required."
            )
        raw = unfreeze(parameters)
        flat = flatten_dict(raw)
        if not flat or any(not np.isfinite(np.asarray(v)).all() for v in flat.values()):
            raise ValueError("Expected a finite native Needle params tree.")
        self._jax, self._jnp, self._arch = jax, jnp, arch
        self._cfg, self._split = cfg, shared_prefix_layers
        self._params = freeze(jax.tree.map(lambda v: jnp.array(v, copy=True), raw))
        self._native_model = arch.SimpleAttentionNetwork(cfg)
        stack = self._params["stack"]
        if stack["mhc_phi_pre"].shape != (
            cfg.num_layers,
            cfg.mhc_lanes * cfg.d_model,
            cfg.mhc_lanes,
        ):
            raise ValueError(
                "Needle hyper-connection topology does not match configuration."
            )
        self._layer_parameters = []
        for layer in range(cfg.num_layers):
            block = {
                "block/" + "/".join(path): value[layer]
                for path, value in flatten_dict(stack["layers"]["block"]).items()
            }
            block.update(
                {
                    "hc/" + key.removeprefix("mhc_"): value[layer]
                    for key, value in stack.items()
                    if key.startswith("mhc_")
                }
            )
            self._layer_parameters.append(MappingProxyType(block))
        self._suffix = MappingProxyType(
            {
                f"layers/{i}/{key}": value
                for i in range(shared_prefix_layers, cfg.num_layers)
                for key, value in self._layer_parameters[i].items()
            }
        )
        self._identity = {
            "backend": "needle-jax-reference-v1",
            "model_id": model_id,
            "revision": revision,
            "tokenizer_id": tokenizer_id,
            "config": asdict(cfg),
            "shared_prefix_layers": shared_prefix_layers,
            "base_sha256": tensor_digest({"/".join(k): v for k, v in flat.items()}),
            "runtime": {
                name: version(name) for name in ("cactus-needle", "jax", "flax")
            },
            "architecture_sha256": sha256_file(arch.__file__),
            "numerics": "unquantized-reference",
        }
        self._key = sha256_json(self._identity)
        self._body = arch._ScanBody(
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.d_model,
            cfg.num_layers,
            cfg.jax_dtype,
            cfg.flash,
            False,
            qk_head_dim=cfg.qk_head_dim,
            v_head_dim=cfg.v_head_dim,
            qkv_conv_taps=cfg.qkv_conv_taps,
            hada_split=bool(cfg.ladder_widths),
        )

    @property
    def forbidden_generation_ids(self) -> tuple[int, ...]:
        """Exclude padding, which is invalid in the unpadded reference input path."""
        return (self._cfg.pad_token_id,)

    @property
    def max_sequence_length(self) -> int:
        """Return the pinned checkpoint's declared context capacity."""
        return self._cfg.max_seq_len

    @property
    def identity(self) -> dict:
        """Return a defensive JSON-compatible copy of the pinned runtime identity."""
        import json

        return json.loads(json.dumps(self._identity))

    @property
    def suffix_parameters(self):
        """Expose immutable JAX leaves through a read-only flat mapping."""
        return self._suffix

    @property
    def feature_size(self) -> int:
        """Return Needle's residual width after lane and sequence pooling."""
        return self._cfg.d_model

    def native_parameters(self, parameters):
        """Convert host arrays to immutable JAX arrays without host roundtrips."""
        return self._jax.tree.map(self._jnp.asarray, dict(parameters))

    def _advance(self, state, start, stop, suffix=None):
        from flax.traverse_util import unflatten_dict

        jnp, cfg = self._jnp, self._cfg
        stream = state.stream
        for layer in range(start, stop):
            if suffix is None:
                parameters = self._layer_parameters[layer]
            else:
                prefix = f"layers/{layer}/"
                parameters = {
                    key[len(prefix) :]: value
                    for key, value in suffix.items()
                    if key.startswith(prefix)
                }
            block = unflatten_dict(
                {
                    tuple(key.split("/")): value
                    for key, value in parameters.items()
                    if key.startswith("block/")
                }
            )
            hc = {
                key.removeprefix("hc/"): jnp.asarray(value)
                for key, value in parameters.items()
                if key.startswith("hc/")
            }
            # Preserve global layer positions; restarting the lane cycle at the
            # split would change the original model even with frozen branches.
            lane = jnp.eye(cfg.mhc_lanes, dtype=jnp.float32)[layer % cfg.mhc_lanes]
            hc.update(pre_off=8 * lane - 4, post_off=-4 * (1 - lane))
            flags = (
                jnp.asarray([float(site == layer) for site in cfg.engram_layers])
                if cfg.engram_layers
                else None
            )
            global_flag = jnp.float32(layer in cfg.global_layers)
            stream, _ = self._body.apply(
                {"params": block},
                stream,
                (flags, global_flag, hc, jnp.int32(layer)),
                state.mask,
                state.local_mask,
                state.rope,
                False,
                state.engram_kv,
                jnp.int32(0),
            )
        return stream

    def prefill(self, input_ids) -> PrefixState:
        """Compute one full prefix; retain lanes, causal context and engram values."""
        jnp, cfg = self._jnp, self._cfg
        ids = np.asarray(input_ids)
        if (
            ids.ndim != 2
            or ids.shape[0] != 1
            or not 0 < ids.shape[1] <= cfg.max_seq_len
            or ids.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Needle expects integer unpadded [1, sequence] within max_seq_len."
            )
        if (
            (ids < 0).any()
            or (ids >= cfg.vocab_size).any()
            or (ids == cfg.pad_token_id).any()
        ):
            raise ValueError("Token IDs must be in vocabulary and contain no padding.")
        tokens = jnp.asarray(ids, dtype=jnp.int32)
        request_key = tensor_digest({"input_ids": np.asarray(tokens)})
        mask = self._arch.make_causal_mask(tokens.shape[1])
        positions = jnp.arange(tokens.shape[1])
        local = (
            mask
            & ((positions[:, None] - positions[None, :]) < cfg.sliding_window)[
                None, None
            ]
            if cfg.sliding_window
            else None
        )
        rope = self._arch.precompute_rope_freqs(
            cfg.qk_head_dim or cfg.d_model // cfg.num_heads,
            tokens.shape[1],
            cfg.rope_theta,
        )
        # Engram values depend on tokens and frozen memory, not the chosen suffix.
        engram = self._native_model.apply(
            {"params": self._params},
            tokens,
            mask,
            False,
            method=self._native_model._engram_kv,
        )
        x = (
            self._params["embedding"]["embedding"][tokens] * np.sqrt(cfg.d_model)
        ).astype(cfg.jax_dtype)
        stream = jnp.broadcast_to(
            x[:, :, None, :], (*x.shape[:2], cfg.mhc_lanes, cfg.d_model)
        )
        initial = _NeedleState(stream, mask, local, rope, engram)
        stream = self._jax.lax.stop_gradient(self._advance(initial, 0, self._split))
        return PrefixState(
            self._key, request_key, _NeedleState(stream, mask, local, rope, engram)
        )

    def _check_state(self, state):
        if (
            not isinstance(state, PrefixState)
            or state.backend_key != self._key
            or not isinstance(state.payload, _NeedleState)
        ):
            raise ValueError(
                "Prefix state belongs to an incompatible backend or topology."
            )

    def _hidden(self, state, suffix):
        self._check_state(state)
        if set(suffix) != set(self._suffix):
            raise ValueError("Suffix topology does not match the backend.")
        stream = self._advance(state.payload, self._split, self._cfg.num_layers, suffix)
        hidden = self._jnp.mean(stream, axis=2)
        return self._arch.ZCRMSNorm(dtype=self._cfg.jax_dtype).apply(
            {"params": self._params["stack"]["final_norm"]}, hidden
        )

    def features(self, state, suffix=None):
        """Mean-pool [sequence, lanes] or suffix [sequence] into one feature vector."""
        self._check_state(state)
        if suffix is None:
            return self._jnp.mean(
                state.payload.stream.astype(self._jnp.float32), axis=(0, 1, 2)
            )
        return self._jnp.mean(
            self._hidden(state, suffix).astype(self._jnp.float32), axis=(0, 1)
        )

    def logits(self, state, suffix):
        """Resume only the suffix and apply the original frozen tied output head."""
        hidden = self._hidden(state, suffix)
        head = self._params["embedding"]["embedding"]
        if self._cfg.out_vocab:
            head = head[: self._cfg.out_vocab]
        return hidden.astype(self._jnp.float32) @ head.T

    def original_logits(self, input_ids):
        """Evaluate the unsplit upstream model for reference equivalence checks."""
        return self._native_model.apply(
            {"params": self._params}, self._jnp.asarray(input_ids), quant=False
        )
