from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from causalq.models.dinov2_wrapper import DINOv2Backbone, DinoVisionTransformer
from causalq.models.dinov3_wrapper import DINOv3Backbone, DINOv3Features
from causalq.models.lora import LoRALinear, adapters_disabled, inject_dino_lora
from causalq.models.query_segmentor import QuerySegmentor
from causalq.losses.causal_query_effect import causal_query_effect_loss
from causalq.protocols.lora_causalq_1114 import MATRIX, require_action
from causalq.runtime.feature_pipeline import slice_features


def tiny():
    return DINOv2Backbone(DinoVisionTransformer(dim=32, depth=4, heads=4, img_size=32),
                          intermediate_indices=(0, 1, 2, 3))


class SeparateAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj = nn.Linear(32, 32), nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)

    def forward(self, x):
        a = (self.q_proj(x) @ self.k_proj(x).transpose(-1, -2) / 32**.5).softmax(-1)
        return a @ self.v_proj(x)


class FakeHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=16, hidden_size=32, num_register_tokens=2)
        self.proj = nn.Conv2d(3, 32, 16, stride=16)
        self.layer = nn.ModuleList([nn.ModuleDict({'attention': SeparateAttention()}) for _ in range(4)])

    def forward(self, pixel_values, output_hidden_states, return_dict):
        x = self.proj(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat((x.mean(1, keepdim=True).expand(-1, 3, -1), x), 1)
        states = [x]
        for layer in self.layer:
            x = x + layer.attention(x)
            states.append(x)
        return SimpleNamespace(last_hidden_state=x, hidden_states=tuple(states))


@pytest.mark.parametrize('kind', ['d2', 'd3'])
def test_zero_delta_gradient_frozen_base_cls_and_rng(kind):
    torch.manual_seed(8)
    b = tiny() if kind == 'd2' else DINOv3Backbone(FakeHF(), intermediate_indices=(1, 2, 3, 4))
    x = torch.randn(2, 3, 32, 48)
    old = b(x)
    frozen = {id(p): p.detach().clone() for p in b.parameters()}
    rng = torch.get_rng_state().clone()
    manifest = inject_dino_lora(b, blocks=(2, 3))
    assert torch.equal(rng, torch.get_rng_state())
    assert manifest['trainable_parameters'] == 2*2*2*32*4
    b.train()
    assert not b.model.training
    new = b(x)
    assert torch.equal(old.patch_tokens, new.patch_tokens)
    assert torch.equal(old.cls_token, new.cls_token)
    assert new.cls_token.shape == (2, 32) and new.patch_tokens.shape == (2, 6, 32)
    assert new.register_tokens.shape[1] == (0 if kind == 'd2' else 2)
    # Nonuniform feature objective avoids the constant energy of LayerNorm.
    optimizer = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=.01)
    probe = torch.randn_like(new.patch_tokens)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (b(x).patch_tokens * probe).mean()
        loss.backward()
        optimizer.step()
    adapters = [m for m in b.modules() if isinstance(m, LoRALinear)]
    assert any(m.q.A.grad.abs().sum() > 0 for m in adapters)
    assert any(m.q.B.grad.abs().sum() > 0 for m in adapters)
    for p in b.parameters():
        if id(p) in frozen:
            assert p.grad is None and torch.equal(p, frozen[id(p)])
    with adapters_disabled(b):
        assert torch.equal(b(x).patch_tokens, old.patch_tokens)
    with pytest.raises(ValueError, match='already'):
        inject_dino_lora(b, blocks=(2, 3))
    sliced = slice_features(new, 1, 2)
    assert torch.equal(sliced.cls_token, new.cls_token[1:2])


def test_fused_k_and_roundtrip():
    layer = LoRALinear(nn.Linear(8, 24), fused=True)
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        layer.q.B.fill_(.2)
        layer.v.B.fill_(.3)
    assert torch.equal(layer(x)[..., 8:16], layer.base(x)[..., 8:16])
    clone = copy.deepcopy(layer)
    clone.load_state_dict(layer.state_dict(), strict=True)
    assert torch.equal(clone(x), layer(x))


@pytest.mark.parametrize('centered', [False, True])
def test_causalq_ce_cqe_gradient_and_checkpoint(centered, tmp_path):
    model = QuerySegmentor(tiny(), num_classes=3, decoder_channels=32, dropout=0.,
                           queries_per_class=2, interaction='static', centered_residuals=centered)
    image = torch.randn(1, 3, 32, 32)
    before = model(image).detach()
    inject_dino_lora(model.backbone, blocks=(2, 3))
    assert torch.equal(before, model(image))
    original = model.forward_components(image, include_anchor=True, component_protocol='lora_causalq_1114')
    reference = original.anchor_scaled_delta_logits.detach()
    torch.nn.functional.cross_entropy(original.logits, torch.zeros(1,32,32,dtype=torch.long)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.backbone.named_parameters() if n.endswith('.B'))
    model.zero_grad()
    # Nonzero alpha is diagnostic-only, not a new initialization recipe.
    with torch.no_grad():
        model.query_head.alpha.fill_(.5)
    photo = model.forward_components(image*.8, include_anchor=True, component_protocol='lora_causalq_1114')
    reference = torch.randn_like(reference)
    causal_query_effect_loss(photo.anchor_scaled_delta_logits, reference, torch.zeros(1,32,32,dtype=torch.long)).backward()
    assert reference.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.backbone.named_parameters() if n.endswith('.B'))
    from causalq.utils.checkpoint import save_training_checkpoint, load_training_checkpoint
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    save_training_checkpoint(tmp_path/'state.pt', model=model, optimizer=opt, iteration=0, metadata={})
    state = load_training_checkpoint(tmp_path/'state.pt')['trainable_model']
    assert any(k.startswith('backbone.') and k.endswith('.B') for k in state)


def test_guards_and_old_feature_constructor():
    b = tiny()
    state = copy.deepcopy(b.state_dict())
    with pytest.raises(ValueError):
        inject_dino_lora(b, blocks=(3, 4))
    assert all(torch.equal(v, state[k]) for k,v in b.state_dict().items())
    assert len(MATRIX) == 28 and len(set(MATRIX)) == 28
    with pytest.raises(ValueError):
        require_action('prepare')
    assert DINOv3Features(torch.zeros(1), torch.zeros(1), ()).cls_token is None
    inject_dino_lora(b, blocks=(2,3))
    from causalq.runtime.feature_pipeline import FeaturePipeline
    with pytest.raises(ValueError, match='frozen'):
        FeaturePipeline(b, {}, device='cpu')
