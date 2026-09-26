"""Grouped semantic-query branches with controlled interaction modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baseline_segmentor import BaselineDecoder
from .dinov3_wrapper import DINOv3Backbone


class GroupedCausalQueryBank(nn.Module):
    """Represent each class by a shared anchor plus diverse residual queries."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_classes: int = 19,
        queries_per_class: int = 3,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_classes < 1 or queries_per_class < 1:
            raise ValueError("Query dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.queries_per_class = int(queries_per_class)
        self.class_anchors = nn.Parameter(
            torch.empty(self.num_classes, self.hidden_size)
        )
        self.query_residuals = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.queries_per_class,
                self.hidden_size,
            )
        )
        nn.init.normal_(self.class_anchors, std=0.02)
        nn.init.normal_(self.query_residuals, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        queries = self.class_anchors[:, None, :] + self.query_residuals
        return queries.unsqueeze(0).expand(batch_size, -1, -1, -1)


class NullQueryBank(nn.Module):
    """Provide class-agnostic query slots shared by every semantic class."""

    def __init__(self, hidden_size: int, *, queries_per_class: int) -> None:
        super().__init__()
        if hidden_size < 1 or queries_per_class < 1:
            raise ValueError("Null-query dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.queries_per_class = int(queries_per_class)
        self.null_queries = nn.Parameter(
            torch.empty(self.queries_per_class, self.hidden_size)
        )
        nn.init.normal_(self.null_queries, std=0.02)

    def forward(self, batch_size: int, num_classes: int) -> Tensor:
        if batch_size < 1 or num_classes < 1:
            raise ValueError("batch_size and num_classes must be positive")
        return self.null_queries.view(
            1, 1, self.queries_per_class, self.hidden_size
        ).expand(batch_size, num_classes, -1, -1)


class _QueryCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, queries: Tensor, image_tokens: Tensor) -> Tensor:
        attended, _ = self.attention(
            queries,
            image_tokens,
            image_tokens,
            need_weights=False,
        )
        queries = self.attention_norm(queries + attended)
        return self.ffn_norm(queries + self.ffn(queries))


class QueryCrossAttention(nn.Module):
    """Update queries from image tokens without writing into image features."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            _QueryCrossAttentionLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        )

    def forward(self, queries: Tensor, image_tokens: Tensor) -> Tensor:
        if queries.ndim != 3 or image_tokens.ndim != 3:
            raise ValueError("queries and image_tokens must both be BND tensors")
        if queries.shape[0] != image_tokens.shape[0]:
            raise ValueError("queries and image_tokens must share a batch size")
        for layer in self.layers:
            queries = layer(queries, image_tokens)
        return queries


class _QueryImageSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens: Tensor) -> Tensor:
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = self.attention_norm(tokens + attended)
        return self.ffn_norm(tokens + self.ffn(tokens))


class QueryImageSelfAttention(nn.Module):
    """Jointly update query and image tokens with bidirectional attention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 8,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            _QueryImageSelfAttentionLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        )

    def forward(
        self,
        queries: Tensor,
        image_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if queries.ndim != 3 or image_tokens.ndim != 3:
            raise ValueError("queries and image_tokens must both be BND tensors")
        if queries.shape[0] != image_tokens.shape[0]:
            raise ValueError("queries and image_tokens must share a batch size")
        query_count = queries.shape[1]
        tokens = torch.cat((queries, image_tokens), dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens[:, :query_count], tokens[:, query_count:]


class QueryResidualHead(nn.Module):
    """Produce per-class dense residual logits from pixels and grouped queries."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_classes: int = 19,
        queries_per_class: int = 3,
        temperature: float = 0.07,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_classes = int(num_classes)
        self.queries_per_class = int(queries_per_class)
        self.temperature = float(temperature)
        self.feature_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def query_scores(
        self,
        image_tokens: Tensor,
        grouped_queries: Tensor,
        *,
        grid_size: tuple[int, int],
    ) -> Tensor:
        """Return per-query similarity maps before class-wise aggregation."""
        batch_size, patch_count, _ = image_tokens.shape
        expected_queries = (self.num_classes, self.queries_per_class)
        if grouped_queries.shape[1:3] != expected_queries:
            raise ValueError(
                "Unexpected grouped-query shape: "
                f"got {tuple(grouped_queries.shape)}, expected Bx"
                f"{self.num_classes}x{self.queries_per_class}xD"
            )
        if patch_count != grid_size[0] * grid_size[1]:
            raise ValueError("grid_size does not match the image-token count")

        pixel_features = F.normalize(
            self.feature_projection(image_tokens), dim=-1
        )
        query_features = F.normalize(
            self.query_projection(grouped_queries), dim=-1
        )
        scores = torch.einsum(
            "bnd,bcrd->bcrn", pixel_features, query_features
        ) / self.temperature
        return scores.reshape(
            batch_size,
            self.num_classes,
            self.queries_per_class,
            *grid_size,
        )

    def delta_logits(
        self,
        image_tokens: Tensor,
        grouped_queries: Tensor,
        *,
        grid_size: tuple[int, int],
    ) -> Tensor:
        scores = self.query_scores(
            image_tokens,
            grouped_queries,
            grid_size=grid_size,
        )
        return torch.logsumexp(scores, dim=2)

    def forward(
        self,
        image_tokens: Tensor,
        grouped_queries: Tensor,
        *,
        grid_size: tuple[int, int],
    ) -> Tensor:
        return self.alpha * self.delta_logits(
            image_tokens,
            grouped_queries,
            grid_size=grid_size,
        )


@dataclass(frozen=True)
class QuerySegmentorOutput:
    logits: Tensor
    base_logits: Tensor
    delta_logits: Tensor
    scaled_delta_logits: Tensor
    query_states: Tensor
    query_score_maps: Tensor
    null_delta_logits: Tensor | None = None
    null_scaled_delta_logits: Tensor | None = None
    null_query_states: Tensor | None = None

    def counterfactual_logits(self, class_index: int) -> Tensor:
        """Apply do(Q_c=0) by removing only class c's scaled residual."""
        if not 0 <= class_index < self.logits.shape[1]:
            raise IndexError(f"class_index out of range: {class_index}")
        result = self.logits.clone()
        # Assign the base map directly.  Subtracting the residual from
        # ``base + residual`` can leave a small floating-point cancellation
        # error, even though the counterfactual is defined to equal the base
        # path exactly for the intervened class.
        result[:, class_index] = self.base_logits[:, class_index]
        return result

    def null_counterfactual_logits(self, class_index: int) -> Tensor:
        """Replace only class ``c`` factual residual with the null residual."""
        if not 0 <= class_index < self.logits.shape[1]:
            raise IndexError(f"class_index out of range: {class_index}")
        if self.null_scaled_delta_logits is None:
            raise RuntimeError("Learned-null intervention is not enabled")
        result = self.logits.clone()
        result[:, class_index] = (
            self.base_logits[:, class_index]
            + self.null_scaled_delta_logits[:, class_index]
        )
        return result


class QuerySegmentor(nn.Module):
    """Frozen-DINOv3 baseline with an additive semantic-query residual."""

    def __init__(
        self,
        backbone: DINOv3Backbone,
        *,
        decoder_channels: int = 256,
        num_classes: int = 19,
        dropout: float = 0.1,
        queries_per_class: int = 3,
        num_heads: int = 8,
        cross_attention_layers: int = 1,
        temperature: float = 0.07,
        alpha_init: float = 0.0,
        interaction: str = "one_way",
        learned_null: bool = False,
    ) -> None:
        super().__init__()
        if not backbone.freeze:
            raise ValueError("Phase 6 requires a frozen DINOv3 backbone")
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.queries_per_class = int(queries_per_class)
        if interaction not in {"one_way", "static", "bidirectional"}:
            raise ValueError(
                "interaction must be 'one_way', 'static', or 'bidirectional'"
            )
        self.interaction = interaction
        self.learned_null = bool(learned_null)
        if self.learned_null and interaction != "static":
            raise ValueError(
                "The first learned-null control requires static interaction"
            )
        feature_count = max(1, len(backbone.intermediate_indices))
        self.decoder = BaselineDecoder(
            backbone.hidden_size,
            num_feature_maps=feature_count,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.query_bank = GroupedCausalQueryBank(
            backbone.hidden_size,
            num_classes=num_classes,
            queries_per_class=queries_per_class,
        )
        if interaction == "one_way":
            self.query_attention = QueryCrossAttention(
                backbone.hidden_size,
                num_heads=num_heads,
                num_layers=cross_attention_layers,
            )
        elif interaction == "bidirectional":
            self.query_attention = QueryImageSelfAttention(
                backbone.hidden_size,
                num_heads=num_heads,
                num_layers=cross_attention_layers,
            )
        else:
            self.query_attention = None
        self.query_head = QueryResidualHead(
            backbone.hidden_size,
            num_classes=num_classes,
            queries_per_class=queries_per_class,
            temperature=temperature,
            alpha_init=alpha_init,
        )
        # Initialize the optional null bank only after every factual-path
        # parameter. This preserves exact same-seed initialization of the
        # selected Static model, so Phase 14 changes only the null mechanism.
        self.null_query_bank = (
            NullQueryBank(
                backbone.hidden_size,
                queries_per_class=queries_per_class,
            )
            if self.learned_null
            else None
        )

    def train(self, mode: bool = True) -> "QuerySegmentor":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward_components(self, images: Tensor) -> QuerySegmentorOutput:
        features = self.backbone(images)
        feature_maps = features.intermediate_maps or (features.patch_map,)
        base_patch_logits = self.decoder(feature_maps)

        grouped_queries = self.query_bank(images.shape[0])
        residual_image_tokens = features.patch_tokens
        if self.query_attention is None:
            contextual_queries = grouped_queries
        elif self.interaction == "one_way":
            flat_queries = grouped_queries.flatten(1, 2)
            contextual_queries = self.query_attention(
                flat_queries,
                features.patch_tokens,
            ).reshape(
                images.shape[0],
                self.num_classes,
                self.queries_per_class,
                self.backbone.hidden_size,
            )
        else:
            flat_queries = grouped_queries.flatten(1, 2)
            contextual_queries, residual_image_tokens = self.query_attention(
                flat_queries,
                features.patch_tokens,
            )
            contextual_queries = contextual_queries.reshape(
                images.shape[0],
                self.num_classes,
                self.queries_per_class,
                self.backbone.hidden_size,
            )
        grid_size = features.patch_map.shape[-2:]
        query_score_maps = self.query_head.query_scores(
            residual_image_tokens,
            contextual_queries,
            grid_size=grid_size,
        )
        delta_patch_logits = torch.logsumexp(query_score_maps, dim=2)

        output_size = images.shape[-2:]
        base_logits = F.interpolate(
            base_patch_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        delta_logits = F.interpolate(
            delta_patch_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        scaled_delta_logits = self.query_head.alpha * delta_logits
        null_delta_logits = None
        null_scaled_delta_logits = None
        null_query_states = None
        if self.null_query_bank is not None:
            null_query_states = self.null_query_bank(
                images.shape[0], self.num_classes
            )
            null_score_maps = self.query_head.query_scores(
                residual_image_tokens,
                null_query_states,
                grid_size=grid_size,
            )
            null_patch_logits = torch.logsumexp(null_score_maps, dim=2)
            null_delta_logits = F.interpolate(
                null_patch_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            null_scaled_delta_logits = self.query_head.alpha * null_delta_logits
        return QuerySegmentorOutput(
            logits=base_logits + scaled_delta_logits,
            base_logits=base_logits,
            delta_logits=delta_logits,
            scaled_delta_logits=scaled_delta_logits,
            query_states=contextual_queries,
            query_score_maps=query_score_maps,
            null_delta_logits=null_delta_logits,
            null_scaled_delta_logits=null_scaled_delta_logits,
            null_query_states=null_query_states,
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_components(images).logits

    def get_query_effect(
        self,
        images: Tensor | None = None,
        *,
        output: QuerySegmentorOutput | None = None,
    ) -> Tensor:
        """Return the public logit-level effect ``alpha * delta_logits``."""
        if (images is None) == (output is None):
            raise ValueError("Provide exactly one of images or output")
        if output is None:
            output = self.forward_components(images)
        return output.scaled_delta_logits

    def get_null_query_effect(
        self,
        images: Tensor | None = None,
        *,
        output: QuerySegmentorOutput | None = None,
    ) -> Tensor:
        """Return factual-minus-null class-logit effects."""
        if (images is None) == (output is None):
            raise ValueError("Provide exactly one of images or output")
        if output is None:
            output = self.forward_components(images)
        if output.null_scaled_delta_logits is None:
            raise RuntimeError("Learned-null intervention is not enabled")
        return output.scaled_delta_logits - output.null_scaled_delta_logits

    def get_query_residuals(self) -> Tensor:
        """Return the learnable within-class residual queries for regularization."""
        return self.query_bank.query_residuals

    def get_query_score_maps(
        self,
        images: Tensor | None = None,
        *,
        output: QuerySegmentorOutput | None = None,
    ) -> Tensor:
        """Return pre-aggregation per-query score maps at patch resolution."""
        if (images is None) == (output is None):
            raise ValueError("Provide exactly one of images or output")
        if output is None:
            output = self.forward_components(images)
        return output.query_score_maps
