"""Shared PCA for matched original and adapted DINO features."""
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

LAYERS = ('patch', 'layer6', 'layer12', 'layer18', 'layer24', 'cls')
SEED = 20261114


def patch_indices(label, grid, sample_id):
    valid = torch.nn.functional.interpolate(label[:, None].float(), grid, mode='nearest')[0, 0] != 255
    valid = valid.flatten().nonzero().flatten().numpy()
    seed = int.from_bytes(hashlib.sha256(f'{SEED}:{sample_id}'.encode()).digest()[:8], 'big')
    return np.sort(np.random.default_rng(seed).choice(valid, min(32, len(valid)), replace=False))


def extract(features, start, label, sample_id, *, dense=False):
    grid = features.patch_map.shape[-2:]
    indices = patch_indices(label, grid, sample_id)
    sl = slice(start, start + 5)
    maps = [m[sl].flatten(2).transpose(1, 2) for m in features.intermediate_maps]
    patch = features.patch_tokens[sl]
    cls = features.cls_token[sl]
    if len(maps) != 4 or len(patch) != 5 or cls.shape != (5, patch.shape[-1]):
        raise ValueError('Invalid native PCA features')
    gather = torch.as_tensor(indices, dtype=torch.long, device=patch.device)
    data = {key: x.index_select(1, gather).float().cpu().numpy() for key, x in zip(LAYERS[:-1], (patch, *maps))}
    data.update(cls=cls.float().cpu().numpy(), indices=indices, grid=np.asarray(grid))
    if dense:
        data['dense'] = patch[:2].float().cpu().numpy()
    if not all(np.isfinite(x).all() for x in data.values()):
        raise ValueError('Nonfinite PCA features')
    return data


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix('.partial')
    with stage.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(stage, path)


def fit_joint(groups):
    """Equal sized matched groups, one FP64 covariance and one eigendecomposition."""
    shapes = {x.shape for x in groups}
    if len(shapes) != 1 or not groups or groups[0].ndim != 2 or groups[0].shape[1] < 3:
        raise ValueError('PCA groups must have equal shape and aligned observations')
    count = len(groups) * len(groups[0])
    if not 1 < count <= 200000:
        raise ValueError('Invalid PCA sample count')
    mean = sum(x.sum(axis=0, dtype=np.float64) for x in groups) / count
    covariance = np.zeros((len(mean), len(mean)), dtype=np.float64)
    for x in groups:
        for start in range(0, len(x), 4096):
            centered = x[start:start+4096] - mean
            covariance += centered.T @ centered
    eig, vectors = np.linalg.eigh(covariance / (count - 1))
    basis = vectors[:, -3:][:, ::-1].copy()
    basis *= np.where(basis[np.abs(basis).argmax(0), np.arange(3)] < 0, -1, 1)
    projections = [project(x, mean, basis) for x in groups]
    ranges = np.quantile(np.concatenate(projections), [.01, .99], axis=0)
    explained = eig[-3:][::-1].clip(0) / max(eig.clip(0).sum(), np.finfo(float).tiny)
    return mean, basis, projections, ranges, explained


def project(values, mean, basis):
    return (values - mean) @ basis


def drift(original, adapted):
    a, b = original.astype(np.float64), adapted.astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (norm_a > 1e-12) & (norm_b > 1e-12)
    displacement = np.linalg.norm(b-a, axis=-1)
    return {'cosine_mean': float(np.mean((a*b).sum(-1)[valid] / (norm_a[valid]*norm_b[valid]))) if valid.any() else None,
            'cosine_defined_count': int(valid.sum()),
            'relative_l2_mean': float(np.mean(displacement[norm_a > 1e-12]/norm_a[norm_a > 1e-12])) if (norm_a > 1e-12).any() else None,
            'original_norm_mean': float(norm_a.mean()), 'adapted_norm_mean': float(norm_b.mean())}


def rgb(values, ranges):
    low, high = ranges
    return np.rint(255 * np.clip((values-low) / np.maximum(high-low, 1e-12), 0, 1)).astype('uint8')


def draw_scatter(path, title, groups, names, view_ids, ranges):
    """Small multiples share axes; every panel distinguishes the five views."""
    colors = ('#313b59', '#c14d36', '#167b72', '#ad7d16', '#8150ab')
    canvas = Image.new('RGB', (1120, 530), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, fill='#172135')
    for i, color in enumerate(colors):
        draw.text((16+115*i, 32), 'original' if i == 0 else f'photo{i}', fill=color)
    for i, (name, values) in enumerate(zip(names, groups)):
        x, y = (i % 4)*280+13, (i // 4)*235+55
        draw.text((x, y), f'{name} | PC1 x PC2', fill='#172135')
        draw.rectangle((x, y+18, x+252, y+215), outline='#c5ccd8')
        coords = (values[:, :2]-ranges[0, :2]) / np.maximum(ranges[1, :2]-ranges[0, :2], 1e-12)
        chosen = np.linspace(0, len(coords)-1, min(1600, len(coords)), dtype=int)
        for j in chosen:
            u, v = coords[j]
            if 0 <= u <= 1 and 0 <= v <= 1:
                draw.point((x+round(u*252), y+215-round(v*197)), fill=colors[view_ids[j]])
    canvas.save(path)


def draw_panel(path, sample_id, backbone, original, adapted, images, label, predictions, mean, basis, ranges):
    from causalq.datasets.label_space import TRAIN_ID_PALETTE
    palette = np.zeros((256, 3), np.uint8)
    palette[:19] = TRAIN_ID_PALETTE
    h, w = original['grid']
    a, b = original['dense'], adapted['dense']
    cell_w, cell_h = 252, round(252 * h / w)
    names = ('Input', 'Ground truth', 'Original PCA', 'LoRA PCA', 'Relative L2 / 1', 'LoRA segmentation')
    if 'reference' in predictions:
        names += ('D3 U0 L0 segmentation',)
    canvas = Image.new('RGB', (len(names)*cell_w, 45+2*(cell_h+20)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), f'{backbone} | GTA5 {sample_id} | same joint PCA and RGB ranges', fill='#172135')
    for view in range(2):
        rel = np.clip(np.linalg.norm(b[view]-a[view], axis=-1) / np.maximum(np.linalg.norm(a[view], axis=-1), 1e-12), 0, 1)
        heat = np.stack((rel, rel*.46, 1-rel), axis=-1).reshape(h, w, 3)
        panels = [images[view], palette[label], rgb(project(a[view],mean,basis),ranges).reshape(h,w,3),
                  rgb(project(b[view],mean,basis),ranges).reshape(h,w,3), (heat*255).astype(np.uint8),
                  palette[predictions['adapted'][view]]]
        if 'reference' in predictions: panels.append(palette[predictions['reference'][view]])
        y = 28+view*(cell_h+20)
        for k, (name, array) in enumerate(zip(names, panels)):
            x = k*cell_w
            draw.text((x+4,y), f'{"Original" if view == 0 else "Photo1"} | {name}', fill='#172135')
            canvas.paste(Image.fromarray(array).resize((cell_w,cell_h), Image.Resampling.NEAREST), (x,y+16))
    canvas.save(path)
