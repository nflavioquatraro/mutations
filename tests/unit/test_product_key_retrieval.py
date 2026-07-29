"""Acceptance tests for EXP-005 product-key micro-expert retrieval."""

from __future__ import annotations

import pytest
import torch

from anchored_moe import (
    ProductKeyMicroExpertRetriever,
    ProductKeyRetrievalConfig,
    ProductKeyRetrievalError,
    ProductKeyRetrievalResult,
)


def _expert_ids(count: int) -> tuple[str, ...]:
    """Return stable row-major test expert identifiers."""

    return tuple(f"micro:{index}" for index in range(count))


def _retriever(
    first_subkeys: torch.Tensor,
    second_subkeys: torch.Tensor,
    *,
    top_k: int,
    trainable: bool = True,
) -> ProductKeyMicroExpertRetriever:
    """Construct a retriever with complete derived identifiers."""

    return ProductKeyMicroExpertRetriever(
        first_subkeys,
        second_subkeys,
        expert_ids=_expert_ids(first_subkeys.shape[0] ** 2),
        config=ProductKeyRetrievalConfig(top_k=top_k),
        trainable=trainable,
    )


def _exhaustive(
    queries: torch.Tensor,
    first_subkeys: torch.Tensor,
    second_subkeys: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the stable exhaustive Cartesian reference."""

    subkey_size = first_subkeys.shape[1]
    first_queries, second_queries = queries.split(subkey_size, dim=-1)
    first_scores = first_queries @ first_subkeys.T
    second_scores = second_queries @ second_subkeys.T
    all_scores = (
        first_scores.unsqueeze(-1) + second_scores.unsqueeze(-2)
    ).flatten(start_dim=-2)
    indices = torch.argsort(
        all_scores,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :top_k]
    return indices, torch.gather(all_scores, dim=-1, index=indices)


@pytest.mark.parametrize("top_k", [1, 2, 5, 16])
def test_bounded_product_key_search_exactly_matches_exhaustive_top_k(
    top_k: int,
) -> None:
    torch.manual_seed(379 + top_k)
    first_subkeys = torch.randn(4, 3)
    second_subkeys = torch.randn(4, 3)
    queries = torch.randn(2, 5, 6)
    retriever = _retriever(
        first_subkeys,
        second_subkeys,
        top_k=top_k,
    )

    result = retriever(queries)
    expected_indices, expected_scores = _exhaustive(
        queries,
        first_subkeys,
        second_subkeys,
        top_k=top_k,
    )

    assert torch.equal(result.selected_indices, expected_indices)
    assert torch.equal(result.selected_scores, expected_scores)
    assert torch.equal(
        result.selected_pairs[..., 0] * 4
        + result.selected_pairs[..., 1],
        result.selected_indices,
    )
    assert result.top_k == top_k
    assert result.candidate_mask.shape == (2, 5, 16)
    assert torch.all(result.candidate_mask.sum(dim=-1) == top_k)


def test_equal_scores_use_stable_row_major_expert_order() -> None:
    retriever = _retriever(
        torch.zeros(3, 2),
        torch.zeros(3, 2),
        top_k=5,
        trainable=False,
    )

    result = retriever(torch.ones(2, 4))

    assert result.selected_indices.tolist() == [[0, 1, 2, 3, 4]] * 2
    assert result.selected_pairs.tolist() == [
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],
    ]
    assert result.mask_for_expert("micro:4").tolist() == [True, True]
    assert result.mask_for_expert("micro:8").tolist() == [False, False]
    with pytest.raises(KeyError, match="Unknown"):
        result.mask_for_expert("missing")


def test_scores_retain_gradients_without_aliasing_source_keys() -> None:
    torch.manual_seed(389)
    first_subkeys = torch.randn(3, 2)
    second_subkeys = torch.randn(3, 2)
    source_first = first_subkeys.clone()
    source_second = second_subkeys.clone()
    queries = torch.randn(4, 4, requires_grad=True)
    retriever = _retriever(
        first_subkeys,
        second_subkeys,
        top_k=3,
    )
    first_subkeys.fill_(100)
    second_subkeys.fill_(-100)

    result = retriever(queries)
    expected_indices, expected_scores = _exhaustive(
        queries,
        source_first,
        source_second,
        top_k=3,
    )
    result.selected_scores.sum().backward()

    assert torch.equal(result.selected_indices, expected_indices)
    assert torch.equal(result.selected_scores, expected_scores)
    assert queries.grad is not None
    assert retriever.first_subkeys.grad is not None
    assert retriever.second_subkeys.grad is not None
    assert retriever.first_subkeys.data_ptr() != first_subkeys.data_ptr()
    assert retriever.second_subkeys.data_ptr() != second_subkeys.data_ptr()

    retriever.set_trainable(False)
    assert all(not parameter.requires_grad for parameter in retriever.parameters())
    retriever.set_trainable(True)
    assert all(parameter.requires_grad for parameter in retriever.parameters())
    with pytest.raises(TypeError, match="boolean"):
        retriever.set_trainable(1)  # type: ignore[arg-type]


def test_config_round_trip_and_constructor_metadata_are_explicit() -> None:
    config = ProductKeyRetrievalConfig(top_k=3)
    retriever = _retriever(
        torch.ones(2, 4),
        torch.ones(2, 4),
        top_k=3,
        trainable=False,
    )

    assert ProductKeyRetrievalConfig.from_dict(config.to_dict()) == config
    assert retriever.architecture == "product_key_micro_expert_retrieval"
    assert retriever.subkey_count == 2
    assert retriever.subkey_size == 4
    assert retriever.query_size == 8
    assert retriever.expert_count == 4
    assert retriever.expert_ids == _expert_ids(4)
    assert set(retriever.state_dict()) == {
        "first_subkeys",
        "second_subkeys",
    }

    with pytest.raises(ProductKeyRetrievalError, match="positive"):
        ProductKeyRetrievalConfig(top_k=0)
    with pytest.raises(ProductKeyRetrievalError, match="fields"):
        ProductKeyRetrievalConfig.from_dict({"top_k": 1, "extra": 2})


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (torch.ones(2, 3), torch.ones(3, 3), "share one shape"),
        (torch.ones(2, 3), torch.ones(2, 4), "share one shape"),
        (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64), "dtype"),
        (
            torch.tensor([[float("nan")]]),
            torch.ones(1, 1),
            "finite",
        ),
    ],
)
def test_incompatible_subkey_tables_fail_before_module_construction(
    first: torch.Tensor,
    second: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ProductKeyRetrievalError, match=message):
        ProductKeyMicroExpertRetriever(
            first,
            second,
            expert_ids=_expert_ids(first.shape[0] ** 2),
            config=ProductKeyRetrievalConfig(top_k=1),
        )


def test_invalid_expert_mapping_top_k_and_queries_are_rejected() -> None:
    keys = torch.ones(2, 3)
    config = ProductKeyRetrievalConfig(top_k=1)

    with pytest.raises(ProductKeyRetrievalError, match="one identifier"):
        ProductKeyMicroExpertRetriever(
            keys,
            keys,
            expert_ids=("a", "b"),
            config=config,
        )
    with pytest.raises(ProductKeyRetrievalError, match="unique"):
        ProductKeyMicroExpertRetriever(
            keys,
            keys,
            expert_ids=("a", "a", "b", "c"),
            config=config,
        )
    with pytest.raises(ProductKeyRetrievalError, match="cannot exceed"):
        _retriever(keys, keys, top_k=5)

    retriever = _retriever(keys, keys, top_k=2)
    with pytest.raises(ProductKeyRetrievalError, match="final dimension"):
        retriever(torch.ones(2, 5))
    with pytest.raises(TypeError, match="floating"):
        retriever(torch.ones(2, 6, dtype=torch.long))
    with pytest.raises(ProductKeyRetrievalError, match="finite"):
        retriever(
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0, float("inf")]]
            )
        )
    with pytest.raises(ProductKeyRetrievalError, match="dtype and device"):
        retriever(torch.ones(2, 6, dtype=torch.float64))


def test_result_rejects_inconsistent_pair_metadata() -> None:
    with pytest.raises(ProductKeyRetrievalError, match="do not map"):
        ProductKeyRetrievalResult(
            selected_indices=torch.tensor([[0, 1]]),
            selected_scores=torch.tensor([[2.0, 1.0]]),
            selected_pairs=torch.tensor([[[0, 0], [1, 1]]]),
            expert_ids=("a", "b", "c", "d"),
            subkey_count=2,
        )
