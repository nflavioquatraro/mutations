"""Incremental Needle attention with immutable, request-local cache snapshots.

References:
    https://github.com/cactus-compute/needle/blob/main/needle/model/architecture.py

Differences:
    Intercepts only native attention; the upstream mHC, engram injection, norms
    and Hadamard blocks still execute. Global layers retain the reserved context;
    local layers use bounded rings. QKV convolution history and token n-gram
    context are retained separately. No optional runtime imports occur at import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

from .backend import PrefixState


class _LayerCache(NamedTuple):
    key: Any
    value: Any
    positions: Any
    q_history: Any
    k_history: Any
    v_history: Any


class _DecodeArrays(NamedTuple):
    position: Any
    layers: tuple
    token_history: Any


@dataclass(frozen=True)
class NeedleDecodeState:
    """Immutable decode snapshot, bound to a backend and selected suffix weights.

    token_count is the number already consumed; capacity is the reserved maximum.
    Snapshots can be forked safely. No cache crosses response branches or owner
    calls implicitly. cache_lengths reports per-layer allocated KV token slots.
    Direct construction/private-field mutation is unsupported.
    """

    backend_key: str
    token_count: int
    capacity: int
    _suffix: Any
    _arrays: Any

    @property
    def cache_lengths(self) -> tuple[int, ...]:
        """Report bounded local and full-context global KV allocations."""
        return tuple(layer.key.shape[2] for layer in self._arrays.layers)


class NeedleKVDecoder:
    """Backend-owned cached executor; weights are arguments to the JIT update.

    This companion is constructed by NeedleBranchBackend, not by callers. Cache
    snapshots are immutable JAX pytrees. The compiled function is shared between
    requests with identical shapes; request data and branch weights are not.
    """

    def __init__(self, backend):
        """Bind one pinned backend and lazily compile the single-token update."""
        self.backend = backend
        self.jax, self.jnp, self.arch = backend._jax, backend._jnp, backend._arch
        self.cfg = backend._cfg
        self._compiled_step = self.jax.jit(self._step_arrays)

    def _dense(self, x, parameters):
        return x.astype(self.cfg.jax_dtype) @ parameters["kernel"].astype(
            self.cfg.jax_dtype
        )

    def _norm(self, x, parameters):
        return self.arch.ZCRMSNorm(dtype=self.cfg.jax_dtype).apply(
            {"params": parameters}, x
        )

    def _history(self, values, length):
        jnp = self.jnp
        return (
            jnp.pad(values, ((0, 0), (max(0, length - values.shape[1]), 0), (0, 0)))[
                :, -length:
            ]
            if length
            else values[:, :0]
        )

    def _attention(
        self, x, p, mask, rope, *, layer, capacity, cache=None, position=None
    ):
        jnp, cfg = self.jnp, self.cfg
        qk = cfg.qk_head_dim or cfg.d_model // cfg.num_heads
        vd = cfg.v_head_dim or cfg.d_model // cfg.num_heads
        raw = tuple(self._dense(x, p[name + "_proj"]) for name in ("q", "k", "v"))
        taps = cfg.qkv_conv_taps
        history_length = max(0, taps - 1)
        window = cfg.sliding_window if layer not in cfg.global_layers else 0
        old_history = (
            None
            if cache is None
            else (cache.q_history, cache.k_history, cache.v_history)
        )
        projected, history = [], []
        for index, name in enumerate(("q", "k", "v")):
            values = raw[index]
            all_values = (
                values
                if cache is None
                else jnp.concatenate((old_history[index], values), axis=1)
            )
            history.append(self._history(all_values, history_length))
            if taps:
                weights = p[name + "_taps"].astype(cfg.jax_dtype)
                if cache is None:
                    values = sum(
                        weights[j]
                        * self.arch._shift_right(values, j)
                        * self.arch._mask_diag(mask, j)[..., None].astype(values.dtype)
                        for j in range(taps)
                    )
                else:
                    values = sum(
                        weights[j]
                        * all_values[:, -1 - j : -j or None]
                        * ((position >= j) & (j < window if window else True))
                        for j in range(taps)
                    )
            projected.append(values)
        q, k, v = (
            projected[0].reshape(1, -1, cfg.num_heads, qk).transpose(0, 2, 1, 3),
            projected[1].reshape(1, -1, cfg.num_kv_heads, qk).transpose(0, 2, 1, 3),
            projected[2].reshape(1, -1, cfg.num_kv_heads, vd).transpose(0, 2, 1, 3),
        )
        q, k = self._norm(q, p["q_norm"]), self._norm(k, p["k_norm"])
        q, k = (self.arch.apply_rope(t, *rope) for t in (q, k))
        if cache is None:
            size = min(window, capacity) if window else capacity
            kept = min(x.shape[1], size)
            positions = jnp.arange(x.shape[1] - kept, x.shape[1], dtype=jnp.int32)
            slots = positions % size
            keys = (
                jnp.zeros((1, cfg.num_kv_heads, size, qk), k.dtype)
                .at[:, :, slots, :]
                .set(k[:, :, -kept:, :])
            )
            values = (
                jnp.zeros((1, cfg.num_kv_heads, size, vd), v.dtype)
                .at[:, :, slots, :]
                .set(v[:, :, -kept:, :])
            )
            indices = jnp.full((size,), -1, jnp.int32).at[slots].set(positions)
        else:
            slot = position % cache.key.shape[2]
            keys = cache.key.at[:, :, slot, :].set(k[:, :, 0, :])
            values = cache.value.at[:, :, slot, :].set(v[:, :, 0, :])
            indices = cache.positions.at[slot].set(position)
            k, v = keys, values
            valid = (indices >= 0) & (indices <= position)
            if window:
                valid &= position - indices < window
            mask = valid[None, None, None, :]
        new_cache = _LayerCache(keys, values, indices, *history)
        if cfg.flash:
            fused = max(qk, vd)
            qf = q * jnp.asarray(math.sqrt(fused / qk), q.dtype)
            qf = jnp.pad(qf, ((0, 0), (0, 0), (0, 0), (0, fused - qk)))
            kf = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, fused - qk)))
            vf = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, fused - vd)))
            impl = (
                "cudnn"
                if self.jax.default_backend() == "gpu"
                and q.dtype in (jnp.bfloat16, jnp.float16)
                else None
            )
            out = self.jax.nn.dot_product_attention(
                qf.transpose(0, 2, 1, 3),
                kf.transpose(0, 2, 1, 3),
                vf.transpose(0, 2, 1, 3),
                mask=mask,
                implementation=impl,
            )[..., :vd].reshape(1, -1, cfg.num_heads * vd)
        else:
            repeats = cfg.num_heads // cfg.num_kv_heads
            k, v = jnp.repeat(k, repeats, axis=1), jnp.repeat(v, repeats, axis=1)
            scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(qk))
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
            out = (
                (self.jax.nn.softmax(scores, axis=-1) @ v)
                .transpose(0, 2, 1, 3)
                .reshape(1, -1, cfg.num_heads * vd)
            )
        out *= self.jax.nn.sigmoid(self._dense(x, p["gate_proj"]))
        return self._dense(out, p["out_proj"]), new_cache

    def _advance(
        self, state, start, stop, suffix, capacity, caches=None, position=None
    ):
        from flax import linen as nn

        captured = []

        def intercept(next_fun, args, kwargs, context):
            if (
                isinstance(context.module, self.arch.MultiHeadAttention)
                and context.method_name == "__call__"
            ):
                layer = start + len(captured)
                result, cache = self._attention(
                    args[0],
                    context.module.variables["params"],
                    kwargs["mask"],
                    kwargs["rope"],
                    layer=layer,
                    capacity=capacity,
                    cache=None if caches is None else caches[layer],
                    position=position,
                )
                captured.append(cache)
                return result
            return next_fun(*args, **kwargs)

        with nn.intercept_methods(intercept):
            stream = self.backend._advance(state, start, stop, suffix)
        if len(captured) != stop - start:
            raise RuntimeError("Native attention interception topology changed.")
        return stream, tuple(captured)

    def prefill(self, input_ids, *, capacity):
        """Encode one prefix and retain only the selected cache capacity."""
        initial, request_key, tokens = self.backend._prepare_input(input_ids)
        if (
            type(capacity) is not int
            or not tokens.shape[1] <= capacity <= self.cfg.max_seq_len
        ):
            raise ValueError("Cache capacity must cover the prompt within max_seq_len.")
        stream, caches = self._advance(initial, 0, self.backend._split, None, capacity)
        return PrefixState(
            self.backend._key,
            request_key,
            replace(
                initial,
                stream=self.jax.lax.stop_gradient(stream),
                decode_prefix=caches,
                tokens=tokens,
                capacity=capacity,
            ),
        )

    def _logits(self, stream, params):
        hidden = self._norm(
            self.jnp.mean(stream, axis=2), params["stack"]["final_norm"]
        )
        head = params["embedding"]["embedding"]
        if self.cfg.out_vocab:
            head = head[: self.cfg.out_vocab]
        return hidden.astype(self.jnp.float32) @ head.T

    def start(self, state, suffix):
        """Prefill exactly one suffix and snapshot its immutable effective weights."""
        self.backend._check_state(state)
        payload = state.payload
        if payload.decode_prefix is None or set(suffix) != set(
            self.backend.suffix_parameters
        ):
            raise ValueError("Expected a cached prefix and compatible suffix tensors.")
        suffix = {
            key: self.jnp.array(value, copy=True) for key, value in suffix.items()
        }
        stream, caches = self._advance(
            payload, self.backend._split, self.cfg.num_layers, suffix, payload.capacity
        )
        length = 4 * max(self.cfg.engram_orders) if self.cfg.engram_layers else 1
        tokens = payload.tokens
        history = self.jnp.pad(tokens, ((0, 0), (max(0, length - tokens.shape[1]), 0)))[
            :, -length:
        ]
        arrays = _DecodeArrays(
            self.jnp.int32(tokens.shape[1]), payload.decode_prefix + caches, history
        )
        snapshot = NeedleDecodeState(
            self.backend._key,
            tokens.shape[1],
            payload.capacity,
            MappingProxyType(suffix),
            arrays,
        )
        return self._logits(stream, self.backend._params), snapshot

    def step(self, state, token_id):
        """Append one valid token, rejecting overflow and incompatible snapshots."""
        if (
            not isinstance(state, NeedleDecodeState)
            or state.backend_key != self.backend._key
        ):
            raise ValueError("Decode snapshot belongs to an incompatible backend.")
        if state.token_count >= state.capacity:
            raise ValueError("Decode cache capacity exhausted.")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, (int, np.integer))
            or not 0 <= token_id < self.cfg.vocab_size
            or token_id == self.cfg.pad_token_id
        ):
            raise ValueError("Expected a non-padding vocabulary token ID.")
        logits, arrays = self._compiled_step(
            self.backend._params,
            dict(state._suffix),
            state._arrays,
            self.jnp.int32(token_id),
        )
        return logits, replace(state, token_count=state.token_count + 1, _arrays=arrays)

    def _step_arrays(self, params, suffix, arrays, token):
        from flax.traverse_util import flatten_dict

        from .needle import _NeedleState

        jnp, cfg = self.jnp, self.cfg
        position = arrays.position
        history = jnp.concatenate(
            (arrays.token_history[:, 1:], token.reshape(1, 1)), axis=1
        )
        engram = None
        if cfg.engram_layers:
            size = history.shape[1]
            valid = position - (size - 1) + jnp.arange(size) >= 0
            mask = (
                self.arch.make_causal_mask(size)
                & valid[None, None, :, None]
                & valid[None, None, None, :]
            )
            engram = self.backend._native_model.apply(
                {"params": params},
                history,
                mask,
                False,
                method=self.backend._native_model._engram_kv,
            )
            engram = tuple(v[:, :, -1:, :] for v in engram)
        x = (
            params["embedding"]["embedding"][token].reshape(1, 1, -1)
            * math.sqrt(cfg.d_model)
        ).astype(cfg.jax_dtype)
        stream = jnp.broadcast_to(x[:, :, None, :], (1, 1, cfg.mhc_lanes, cfg.d_model))
        qk = cfg.qk_head_dim or cfg.d_model // cfg.num_heads
        angles = position.astype(jnp.float32) / cfg.rope_theta ** (
            jnp.arange(0, qk, 2).astype(jnp.float32) / qk
        )
        rope = (jnp.cos(angles)[None, :], jnp.sin(angles)[None, :])
        # Supply weights as traced arguments, avoiding giant constants baked into
        # the executable and allowing every branch to reuse the compiled update.
        block = flatten_dict(params["stack"]["layers"]["block"])
        full = {
            f"layers/{layer}/block/" + "/".join(path): value[layer]
            for layer in range(cfg.num_layers)
            for path, value in block.items()
        }
        full.update(
            {
                f"layers/{layer}/hc/" + key.removeprefix("mhc_"): value[layer]
                for layer in range(cfg.num_layers)
                for key, value in params["stack"].items()
                if key.startswith("mhc_")
            }
        )
        full.update(suffix)
        state = _NeedleState(stream, None, None, rope, engram)
        stream, caches = self._advance(
            state, 0, cfg.num_layers, full, 0, arrays.layers, position
        )
        return self._logits(stream, params), _DecodeArrays(
            position + 1, caches, history
        )
