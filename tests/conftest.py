"""Deterministic test fixtures for the foundation implementation."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_qwen3_model() -> Qwen3ForCausalLM:
    """Return a deterministic Qwen3 model small enough for CPU unit tests."""

    torch.manual_seed(13)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    return Qwen3ForCausalLM(config).eval()
