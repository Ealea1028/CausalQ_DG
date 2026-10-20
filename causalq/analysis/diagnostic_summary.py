"""Exact image-macro quantiles and separate pixel-weighted responsibility."""

from array import array
import math
import numpy as np

IDENTITY = ('class_id', 'component', 'view_index', 'scope', 'term', 'query_r')
OMIT = (*IDENTITY, 'present', 'valid_pixels', 'pixels')


def scalar_items(value, prefix=''):
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from scalar_items(item, f'{prefix}.{i}')
    elif value is None or isinstance(value, (int, float, bool)):
        yield prefix, value


class DiagnosticSummary:
    def __init__(self):
        self.values = {}
        self.weighted = {}
        self.global_fisher = {}
        self.images = 0

    def add_rows(self, kind, rows):
        image_macro = {}
        for row in rows:
            identity = {k: row[k] for k in IDENTITY if k in row}
            populations = ['all_classes']
            if kind != 'responsibility' and 'present' in row:
                population = 'gt_present' if row['present'] else 'gt_absent'
                if row['present'] or kind == 'gradients':
                    populations.append(population)
            numbers = [(name, number) for field, value in row.items() if field not in OMIT
                       for name, number in scalar_items(value, field)]
            for population in populations:
                key = (kind, population, tuple(sorted(identity.items())))
                stats = self.values.setdefault(key, {})
                global_key = (kind, population, tuple(sorted({**identity, 'class_id': -1}.items())))
                bucket = image_macro.setdefault(global_key, {})
                for name, number in numbers:
                    entry = stats.setdefault(name, [array('d'), 0])
                    total = bucket.setdefault(name, [0., 0, 0., 0])
                    if number is None:
                        entry[1] += 1
                    else:
                        if not math.isfinite(number):raise ValueError('Nonfinite diagnostic aggregate')
                        entry[0].append(float(number));total[0] += float(number);total[1] += 1
                        if name.endswith('pearson'):
                            clipped = min(1-1e-7, max(-1+1e-7, float(number)))
                            total[2] += math.atanh(clipped);total[3] += int(clipped != number)
            if kind == 'responsibility' and row['pixels']:
                key = (row['class_id'], row['scope'], row['view_index'])
                slots = self.weighted.setdefault(key, {})
                value = slots.setdefault(row['query_r'], {'pixels': 0, 'rho': 0., 'entropy': 0.})
                n = row['pixels'];value['pixels'] += n
                value['rho'] += n * row['mean_responsibility'];value['entropy'] += n * row['pixel_entropy']
        for key, fields in image_macro.items():
            stats = self.values.setdefault(key, {})
            for name, (total, count, z_total, clips) in fields.items():
                entry = stats.setdefault(name, [array('d'), 0])
                if count:entry[0].append(total / count)
                else:entry[1] += 1
                if count and name.endswith('pearson'):
                    z_entry = self.global_fisher.setdefault(key, {}).setdefault(name, [array('d'), 0])
                    z_entry[0].append(z_total / count);z_entry[1] += clips

    def add_image(self, record):
        for kind in ('effects', 'correlations', 'gradients', 'responsibility'):
            self.add_rows(kind, record[kind])
        self.images += 1

    def result(self):
        table = []
        for (kind, population, identity), fields in sorted(self.values.items(), key=lambda x: repr(x[0])):
            metrics = {}
            for field, (values, nulls) in fields.items():
                x = np.frombuffer(values, dtype=np.float64)
                metric = {'count': len(x), 'null_count': nulls, 'mean': float(x.mean()) if len(x) else None}
                if len(x) and kind in ('effects', 'correlations'):
                    metric.update(zip(('p10', 'median', 'p90'), np.quantile(x, [.1, .5, .9]).tolist()))
                    if field.endswith('pearson'):
                        if dict(identity).get('class_id') == -1:
                            zs, clips = self.global_fisher[(kind, population, identity)][field]
                            metric['fisher_mean'] = float(np.tanh(np.frombuffer(zs, dtype=np.float64).mean()))
                            metric['fisher_clipped_count'] = clips
                        else:
                            clipped = np.clip(x, -1 + 1e-7, 1 - 1e-7)
                            metric['fisher_mean'] = float(np.tanh(np.arctanh(clipped).mean()))
                            metric['fisher_clipped_count'] = int((clipped != x).sum())
                metrics[field] = metric
            unit = 'image_macro_over_defined_classes' if dict(identity).get('class_id') == -1 else 'image_class'
            table.append({'kind': kind, 'population': population, **dict(identity), 'observation_unit': unit, 'metrics': metrics})
        weighted = []
        for (c, scope, view), slots in sorted(self.weighted.items()):
            counts = {v['pixels'] for v in slots.values()}
            if len(counts) != 1:
                raise ValueError('Slot pixel coverage differs')
            n = counts.pop();rho = [slots[r]['rho'] / n for r in sorted(slots)]
            entropy = -sum(p * math.log(max(p, 1e-30)) for p in rho)
            weighted.append({'class_id': c, 'scope': scope, 'view_index': view, 'pixels': n,
                'mean_responsibility': rho, 'usage_entropy': entropy, 'effective_query_count': math.exp(entropy),
                'normalized_usage_entropy': entropy / math.log(len(rho)),
                'pixel_entropy': slots[min(slots)]['entropy'] / n})
        return {'sample_count': self.images, 'image_macro_statistics': table,
                'pixel_weighted_responsibility': weighted}
