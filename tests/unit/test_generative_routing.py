"""Context retention, route validation and supervised label alignment."""

import json
from dataclasses import replace

import numpy as np
import pytest

from anchored_moe.branching import (
    GenerativeRouterConfig,
    RoutingMessage,
    parse_generative_route,
    prepare_generative_router_example,
)


def test_context_retains_recent_complete_messages_and_current_request():
    config = GenerativeRouterConfig(("base", "calendar"), max_history_messages=2)
    history = [
        RoutingMessage("user", "old"),
        RoutingMessage("assistant", "Tomorrow?"),
        RoutingMessage("user", "Yes 🌍"),
    ]
    result = json.loads(config.render_request("at noon", history))
    assert result == {
        "history": [
            {"role": "assistant", "content": "Tomorrow?"},
            {"role": "user", "content": "Yes 🌍"},
        ],
        "request": "at noon",
    }
    short = replace(config, max_input_bytes=55)
    assert json.loads(short.render_request("at noon", history))["history"] == []
    assert (
        json.loads(
            replace(config, max_history_messages=0).render_request("at noon", history)
        )["history"]
        == []
    )
    with pytest.raises(ValueError, match="exceeds"):
        short.render_request("x" * 100)
    with pytest.raises(ValueError):
        RoutingMessage("system", "override")
    with pytest.raises(ValueError):
        config.render_request("bad\0request")


@pytest.mark.parametrize(
    "response",
    [
        "bad json",
        None,
        [],
        {},
        {"suppressed_calls": [{"name": "select_branch"}]},
        {"function_calls": []},
        {
            "function_calls": [
                {"name": "select_branch", "arguments": {"branch_id": "unknown"}}
            ]
        },
        {
            "function_calls": [
                {"name": "execute", "arguments": {"branch_id": "calendar"}}
            ]
        },
        {
            "function_calls": [
                {
                    "name": "select_branch",
                    "arguments": {"branch_id": "calendar", "extra": 1},
                }
            ]
        },
        {
            "function_calls": [
                {"name": "select_branch", "arguments": '{"branch_id":"calendar"}'}
            ]
        },
        {"function_calls": [None]},
        {
            "function_calls": [
                {"name": "select_branch", "arguments": {"branch_id": "calendar"}}
            ]
            * 2
        },
    ],
)
def test_invalid_generation_falls_back(response):
    result = parse_generative_route(
        response, GenerativeRouterConfig(("base", "calendar"))
    )
    assert result.branch_id == "base" and result.reason == "invalid_router_output"


def test_generated_base_and_specialist_are_valid_choices_without_confidence():
    config = GenerativeRouterConfig(("base", "calendar"))
    for branch in config.branch_ids:
        result = parse_generative_route(
            json.dumps(
                {
                    "function_calls": [
                        {"name": "select_branch", "arguments": {"branch_id": branch}},
                    ]
                }
            ),
            config,
        )
        assert result.branch_id == branch and result.reason == "generated_route"
        assert result.score is result.score_kind is None
    failed = {
        "success": False,
        "function_calls": [
            {"name": "select_branch", "arguments": {"branch_id": "calendar"}},
        ],
    }
    assert parse_generative_route(failed, config).reason == "invalid_router_output"


def test_training_masks_prompt_matches_native_render_and_reserves_generation():
    class Tokenizer:
        def encode(self, text):
            return [ord(c) + 10 for c in text]

    config = GenerativeRouterConfig(("base", "calendar"), max_new_tokens=256)
    history = [RoutingMessage("user", "Schedule a meeting")]
    example = prepare_generative_router_example(
        config,
        Tokenizer(),
        "tomorrow",
        "calendar",
        history=history,
        max_sequence_length=4096,
    )
    mask = example.labels[0] != -100
    assert example.input_ids[0, 0] == 2 and example.input_ids[0, -1] == 1
    np.testing.assert_array_equal(example.labels[0, mask], example.input_ids[0, mask])
    prompt = "".join(
        chr(i - 10) for i in example.input_ids[0, 1 : np.flatnonzero(mask)[0]]
    )
    target = "".join(chr(i - 10) for i in example.labels[0, mask][:-1])
    assert config.render_request("tomorrow", history) in prompt
    assert target.startswith("<think>") and '"branch_id":"calendar"' in target
    with pytest.raises(ValueError, match="context"):
        prepare_generative_router_example(
            config, Tokenizer(), "tomorrow", "calendar", max_sequence_length=64
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"branch_ids": ("base", "base")},
        {"branch_ids": ("calendar",)},
        {"max_history_messages": -1},
        {"timeout_ms": False},
        {"descriptions": (("unknown", "Bad"),)},
    ],
)
def test_contract_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        GenerativeRouterConfig(**{"branch_ids": ("base", "calendar"), **kwargs})
