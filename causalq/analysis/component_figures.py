"""Systematic scientific plots of source query components, with shared scales."""

from pathlib import Path

import numpy as np
import torch

from causalq.datasets.label_space import colorize_train_ids

CLASSES = ((0, 'road'), (2, 'building'), (8, 'vegetation'), (13, 'car'), (11, 'person'))


def render_component_figures(directory, source_id, image, photo, label, outputs, effects, *, classes=CLASSES):
    """Plot one preselected image; consumes cached outputs, never encodes images.

    Effects share one signed scale across six maps per class, preserving
    magnitude differences. Responsibility uses native grid and fixed [0,1].
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import colormaps

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    gt = label[0].detach().cpu().numpy()
    valid = gt != 255
    if not valid.any():
        raise ValueError('Cannot render an all-ignore source frame')
    mean = np.asarray((.485, .456, .406))
    std = np.asarray((.229, .224, .225))
    rgb = [(value[0].detach().float().cpu().permute(1, 2, 0).numpy() * std + mean).clip(0, 1) for value in (image, photo)]
    pred = [out.logits[0].argmax(0).detach().cpu().numpy().astype(np.uint8) for out in outputs]
    for value in pred:
        value[~valid] = 255
    paths = []

    def save(figure, name):
        FigureCanvasAgg(figure)
        path = directory / name
        figure.savefig(path, dpi=120, metadata={'Software': 'CausalQ-DG component analysis'})
        figure.clear()
        paths.append(path.name)

    figure = Figure(figsize=(16, 3.2), layout='constrained')
    axes = figure.subplots(1, 5)
    panels = [rgb[0], colorize_train_ids(gt.astype(np.uint8)), colorize_train_ids(pred[0]), rgb[1], colorize_train_ids(pred[1])]
    for ax, panel, title in zip(axes, panels, ('Original', 'Source GT', 'Prediction / original', 'Photometric', 'Prediction / photo')):
        ax.imshow(panel, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    figure.suptitle(f'Source {source_id} | fixed frame and appearance seed | no result-based selection', fontsize=11)
    save(figure, 'overview.png')

    figure = Figure(figsize=(16, 10), layout='constrained')
    axes = figure.subplots(len(classes), 6, squeeze=False)
    scales = {}
    palette = colormaps['RdBu_r'].copy()
    palette.set_bad('#999999')
    for row, (c, name) in enumerate(classes):
        maps = [maps[component][0, c].detach().float().cpu().numpy() for maps in effects for component in ('full', 'anchor', 'residual')]
        scale = max(max(float(np.abs(value[valid]).max()) for value in maps), 1e-12)
        scales[str(c)] = scale
        for col, value in enumerate(maps):
            ax = axes[row, col]
            plotted = ax.imshow(np.ma.masked_where(~valid, value), cmap=palette, vmin=-scale, vmax=scale, interpolation='nearest')
            if row == 0:
                ax.set_title(('Orig / full', 'Orig / anchor', 'Orig / residual', 'Photo / full', 'Photo / anchor', 'Photo / residual')[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{name} (ID {c})\nGT pixels: {int((gt == c).sum())}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        figure.colorbar(plotted, ax=list(axes[row]), shrink=.7, format='%.2g', label='logit contribution')
    figure.suptitle(f'Source {source_id} | signed effects: shared scale per class; grey=ignore255', fontsize=11)
    save(figure, 'effects.png')

    responsibilities = [out.query_score_maps[0].detach().float().softmax(1).cpu().numpy() for out in outputs]
    slots = responsibilities[0].shape[1]
    figure = Figure(figsize=(max(8, 3 * slots * 2), 10), layout='constrained')
    axes = figure.subplots(len(classes), slots * 2, squeeze=False)
    for row, (c, name) in enumerate(classes):
        for view in range(2):
            for slot in range(slots):
                ax = axes[row, view * slots + slot]
                plotted = ax.imshow(responsibilities[view][c, slot], cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
                if row == 0:
                    ax.set_title(f'{("Orig", "Photo")[view]} / query {slot}', fontsize=10)
                if view == 0 and slot == 0:
                    ax.set_ylabel(f'{name} (ID {c})\nGT pixels: {int((gt == c).sum())}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
    figure.colorbar(plotted, ax=list(axes.flat), shrink=.7, label='softmax over query slots')
    figure.suptitle(f'Source {source_id} | responsibility: native score grid, attribution only', fontsize=11)
    save(figure, 'responsibility.png')
    return {'source_id': source_id, 'files': paths, 'effect_signed_scale_by_class': scales,
            'gt_class_pixel_counts': {str(c): int((gt == c).sum()) for c, _ in classes}}
