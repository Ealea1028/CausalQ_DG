"""Export three frozen source examples without changing appearance or RNG."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def rgb_image(normalized, style):
    value = style._unit(normalized).detach().cpu()[0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(value * 255).astype(np.uint8))


def export_examples(directory, raw_image, views, traces, seeds, style, source_id, *, identity=None):
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True)
    raw_image.save(directory / 'source_original.png')
    pictures = [('input_original', rgb_image(views[0], style))]
    pictures += [(f'photometric_{i:02d}', rgb_image(view, style)) for i, view in enumerate(views[1:], 1)]
    unit = style._unit(views[0]);p = traces[0]
    for name in ('brightness', 'contrast', 'saturation', 'gamma', 'temperature', 'grayscale'):
        scalar = p.get(name, [1.])[0]
        if name == 'brightness':value = unit * scalar
        elif name == 'contrast':value = (unit - unit.mean((2, 3), keepdim=True)) * scalar + unit.mean((2, 3), keepdim=True)
        elif name == 'saturation':value = (unit - unit.mean(1, keepdim=True)) * scalar + unit.mean(1, keepdim=True)
        elif name == 'gamma':value = unit.clamp(1e-6, 1).pow(scalar)
        elif name == 'temperature':value = unit * unit.new_tensor([scalar, 1., 1. / scalar])[None, :, None, None]
        else:value = unit.mean(1, keepdim=True).expand_as(unit)
        image = rgb_image(style._normalize(value, views[0].dtype), style)
        pictures.append((f'diagnostic_only_{name}', image))
    for name, image in pictures:image.save(directory / f'{name}.png')
    sheet = Image.new('RGB', (960, 4 * 226), 'white');draw = ImageDraw.Draw(sheet)
    panels = [('source_original', raw_image), *pictures]
    for i, (name, image) in enumerate(panels):
        thumb = image.copy();thumb.thumbnail((316, 198))
        x, y = (i % 3) * 320, (i // 3) * 226
        sheet.paste(thumb, (x, y + 22));draw.text((x + 3, y + 3), name, fill='black')
    sheet.save(directory / 'contact_sheet.png')
    (directory / 'interventions.json').write_bytes((json.dumps({'source_id': source_id, 'selection_seed': 20261020,
        'identity': identity or {},
        'geometry': 'SourceEvalTransform longedge1024 aspect pad16', 'input_shape': list(views[0].shape),
        'raw_size': list(raw_image.size), 'photometric_seeds': seeds, 'photometric_parameters': traces,
        'single_methods': 'diagnostic_only; independent from input_original; grayscale forced on',
        'display': 'inverse ImageNet normalization; clamp; no independent min-max'}, indent=2) + '\n').encode())
