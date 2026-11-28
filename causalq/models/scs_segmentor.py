"""Frozen DINO with REIN feature taps and an optional SCS injector."""

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from causalq.rein_head import load_rein_mask2former_head
from .dinov2_wrapper import DINOv2Backbone
from .dinov3_wrapper import DINOv3Backbone, DINOv3Features
from .scs import SCSConfig, SCSInjector, scs_auxiliary_loss


REIN_TAPS = (8, 12, 16, 24)


def _patch_map(patches: Tensor, grid: tuple[int, int]) -> Tensor:
    return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], *grid).contiguous()


class SCSBackbone(nn.Module):
    def __init__(
        self, backbone: DINOv2Backbone | DINOv3Backbone, config: SCSConfig | None,
        *, taps: tuple[int, ...] = REIN_TAPS,
    ):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.is_dino2 = isinstance(backbone, DINOv2Backbone)
        self.prefix_tokens = 1 if self.is_dino2 else backbone.num_prefix_tokens
        layers = backbone.model.blocks if self.is_dino2 else backbone.model.layer
        if not taps or tuple(sorted(set(taps))) != taps or not 1 <= taps[0] <= taps[-1] <= len(layers):
            raise ValueError("Expected ordered, one-based backbone taps")
        if config is not None and (config.backbone_dim != backbone.hidden_size
                                   or config.selected_layers[-1] > taps[-1]):
            raise ValueError("SCS injection must match backbone width and precede the last tap")
        self.taps = taps
        self.injector = SCSInjector(config) if config is not None else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, images: Tensor, *, return_aux: bool = False, patch_valid: Tensor | None = None,
        inject: bool = True,
    ) -> tuple[DINOv3Features, Tensor | None]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected [B, 3, H, W] RGB images")
        ph, pw = self.backbone.patch_size
        h, w = images.shape[-2:]
        if h % ph or w % pw or h < ph or w < pw:
            raise ValueError("Expected a nonempty grid aligned to the backbone patch size")
        grid = (h // ph, w // pw)
        model = self.backbone.model
        if self.is_dino2:
            layers = model.blocks
            patches = model.patch_embed(images)
            cls = model.cls_token.to(patches.dtype).expand(images.shape[0], -1, -1)
            tokens = torch.cat((cls, patches), dim=1) + model.position(h, w, patches.dtype)
            layer_kwargs = {}
        else:
            layers = model.layer
            images = images.to(model.embeddings.patch_embeddings.weight.dtype)
            tokens = model.embeddings(images, bool_masked_pos=None)
            layer_kwargs = {"attention_mask": None, "position_embeddings": model.rope_embeddings(images)}

        prefix = self.prefix_tokens
        if tokens.shape[1] != prefix + grid[0] * grid[1]:
            raise RuntimeError("DINO tokens and spatial grid disagree")
        injector = self.injector if inject else None
        selected = injector.config.selected_layers if injector is not None else ()
        maps = []
        aux_map = None
        for layer, block in enumerate(layers, start=1):
            tokens = block(tokens, **layer_kwargs)
            if layer in selected:
                patches, aux = injector(layer, tokens[:, prefix:], patch_valid=patch_valid,
                                        return_aux=return_aux and layer == selected[-1])
                tokens = torch.cat((tokens[:, :prefix], patches), dim=1)
                if aux is not None:
                    aux_map = _patch_map(aux, grid)
            if layer in self.taps:
                maps.append(_patch_map(tokens[:, prefix:], grid))
        final = model.norm(tokens)
        patches = final[:, prefix:]
        return DINOv3Features(
            patch_map=_patch_map(patches, grid), patch_tokens=patches,
            intermediate_maps=tuple(maps), cls_token=final[:, 0], register_tokens=final[:, 1:prefix],
        ), aux_map


def rein_pyramid(features: DINOv3Features) -> list[Tensor]:
    if len(features.intermediate_maps) != 4:
        raise ValueError("The REIN pixel decoder requires four ordered DINO taps")
    return [feature.float() if scale == 1 else F.interpolate(
        feature.float(), scale_factor=scale, mode="bilinear", align_corners=False
    ) for feature, scale in zip(features.intermediate_maps, (4, 2, 1, 0.5))]


class SCSMask2Former(nn.Module):
    def __init__(self, backbone: SCSBackbone, head: nn.Module, *, num_queries=100, head_dim=256):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.query_embed = nn.Embedding(num_queries, head_dim)

    def loss(
        self, images: Tensor, labels: Tensor, *, auxiliary_weight: float = 0.1,
        patch_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

        features, aux = self.backbone(images, return_aux=auxiliary_weight != 0,
                                      patch_valid=patch_valid)
        size = images.shape[-2:]
        samples = []
        for label in labels:
            sample = SegDataSample(metainfo={"img_shape": size, "pad_shape": size, "ori_shape": size})
            sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
            samples.append(sample)
        # The MMCV deformable attention receives float32 input even with DINO autocast.
        with torch.autocast(images.device.type, enabled=False):
            losses = self.head.loss((rein_pyramid(features), self.query_embed.weight), samples, {})
            if aux is not None:
                losses["loss_scs_aux"] = auxiliary_weight * scs_auxiliary_loss(aux, labels)
        return losses


def build_scs_segmentor(backbone: SCSBackbone, rein_root: Path) -> SCSMask2Former:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    head_class = load_rein_mask2former_head(rein_root)
    init_default_scope("mmseg")
    config = Config.fromfile(str(rein_root / "configs/_base_/models/rein_dinov2_mask2former.py"))
    head_config = config.model.decode_head.copy()
    if head_config.pop("type") != head_class.__name__:
        raise ValueError("Unexpected head type in the pinned REIN config")
    head = head_class(**head_config)
    head.init_weights()
    return SCSMask2Former(backbone, head, num_queries=head_config.num_queries,
                          head_dim=head_config.feat_channels)
