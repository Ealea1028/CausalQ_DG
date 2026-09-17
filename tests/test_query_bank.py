import torch

from causalq.models import GroupedCausalQueryBank, QueryCrossAttention


def test_grouped_query_bank_uses_anchor_plus_residual() -> None:
    bank = GroupedCausalQueryBank(
        16,
        num_classes=4,
        queries_per_class=3,
    )

    queries = bank(batch_size=2)

    assert queries.shape == (2, 4, 3, 16)
    assert torch.equal(queries[0], queries[1])
    assert torch.allclose(
        queries[0],
        bank.class_anchors[:, None, :] + bank.query_residuals,
    )


def test_query_cross_attention_is_one_way() -> None:
    attention = QueryCrossAttention(16, num_heads=4, num_layers=1)
    queries = torch.randn(2, 12, 16)
    image_tokens = torch.randn(2, 20, 16)
    original_image_tokens = image_tokens.clone()

    updated_queries = attention(queries, image_tokens)

    assert updated_queries.shape == queries.shape
    assert torch.equal(image_tokens, original_image_tokens)
    assert not torch.equal(updated_queries, queries)
