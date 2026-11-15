"""LoRA study routing adapter; preserve historical frozen routing source bytes."""
from dataclasses import replace
import torch
from .hpi_query_route import RoutedQuerySegmentor
from .query_segmentor import QuerySegmentor


class LoRARoutedQuerySegmentor(RoutedQuerySegmentor):
    def forward_routed(self, features, output_size, *, include_anchor=False, component_protocol=None,
                       closed_classes=None):
        grid=features.patch_map.shape[-2:]
        active=None if closed_classes is None else (
            torch.arange(self.num_classes,device=closed_classes.device)[None]!=closed_classes[:,None])
        route=self.query_router(features.patch_tokens,self.get_anchors(),grid,active_classes=active)
        if len(features.intermediate_maps)!=4:
            raise ValueError('Routing expects four backbone maps')
        batch,_,width=route.tokens.shape
        routed=replace(features,
            patch_map=route.tokens.transpose(1,2).reshape(batch,width,*grid),patch_tokens=route.tokens,
            intermediate_maps=(*features.intermediate_maps[:-1],features.intermediate_maps[-1]+
                route.injected.transpose(1,2).reshape(batch,width,*grid)))
        output=QuerySegmentor.forward_features(self,routed,output_size,include_anchor=include_anchor,
                                               component_protocol=component_protocol)
        return output,route
