"""Phase45 FP64 effect reductions on the device that owns the logits."""
import math

import torch

from causalq.analysis.selective_stability import EPS, effect_and_correlation_rows, structural_effect_and_correlation_rows


def _gram(values):
    """Return spatial second moments using bounded spatial tiles."""
    classes, rows, pixels = values.shape
    width = 4096
    total = values.new_zeros(classes, rows, rows)
    for start in range(0, pixels, 16 * width):
        tile = values[..., start:start+16*width]
        padding = (-tile.shape[-1]) % width
        if padding:
            tile = torch.nn.functional.pad(tile, (0, padding))
        blocks = tile.reshape(classes, rows, -1, width).transpose(1, 2)
        total += (blocks @ blocks.transpose(-1,-2)).sum(dim=1)
    return total / pixels


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > EPS else None


def _similarity(raw, centered, i, j):
    def corr(matrix):
        ni, nj = (math.sqrt(max(0., matrix[k][k])) for k in (i,j))
        return max(-1.,min(1.,matrix[i][j]/(ni*nj))) if min(ni,nj) > EPS else None
    return {'pearson':corr(centered),'cosine':corr(raw),'covariance':centered[i][j]}


def effect_rows(full, anchor, labels):
    """Use the same valid pixels, FP32 maps and FP64 statistics as Phase45 v1."""
    if full.ndim != 4 or full.shape[0] != 5 or labels.shape != (1,*full.shape[-2:]):
        raise ValueError('Expected five aligned views and one label map')
    if anchor is not None and anchor.shape != full.shape:
        raise ValueError('Anchor shape differs')
    labels = labels.to(full.device)
    valid = labels[0] != 255
    idx = valid.flatten().nonzero().flatten()
    if not idx.numel(): raise ValueError('No valid source pixels')
    present = set(labels.unique().tolist())
    effects, correlations = [], []
    for start in range(0,full.shape[1],2):
        cut = slice(start,start+2)
        f = full[:,cut].float().flatten(2).index_select(2,idx).transpose(0,1)
        a = f if anchor is None else anchor[:,cut].float().flatten(2).index_select(2,idx).transpose(0,1)
        # Keep subtraction on the same FP32 representation used in v1.
        x = torch.stack((f,a,f-a),dim=1).double()  # class, component, view, pixel
        if not torch.isfinite(x).all(): raise ValueError('Nonfinite source effects')
        means = x.mean(-1)
        mean_abs = x.abs().mean(-1)
        rms = x.square().mean(-1).sqrt()
        delta = x-x.mean(2,keepdim=True)
        raw_variance = delta.square().mean((2,3))
        covariance = (delta[:,1]*delta[:,2]).mean((1,2))
        del delta
        pair_raw = (x[:,:,0]-x[:,:,1]).square().mean(-1)/4
        normalized = x/rms.clamp_min(EPS)[...,None]
        norm_var = (normalized-normalized.mean(2,keepdim=True)).square_().mean((2,3))
        pair_norm = (normalized[:,:,0]-normalized[:,:,1]).square().mean(-1)/4
        del normalized
        # One matrix per class covers every requested spatial pair. The last
        # eight rows are photo-minus-original Anchor and Residual maps.
        da = x[:,1,1:] - x[:,1,:1]
        dr = x[:,2,1:] - x[:,2,:1]
        joint = torch.cat((x.flatten(1,2),da,dr),dim=1)
        raw_gram = _gram(joint).cpu().tolist()
        spatial_gram = _gram(joint-joint.mean(-1,keepdim=True)).cpu().tolist()
        del joint, x, da, dr
        view_stats = torch.stack((means,mean_abs,rms),dim=-1).cpu().tolist()
        component_stats = torch.stack((raw_variance,norm_var,pair_raw,pair_norm),dim=-1).cpu().tolist()
        for local, cov in enumerate(covariance.cpu().tolist()):
            category = start+local
            common = {'class_id':category,'present':category in present,'valid_pixels':idx.numel()}
            for j, component in enumerate(('full','anchor','residual')):
                signed, absolute, energy = zip(*view_stats[local][j])
                var, normalized_var, pair, normalized_pair = component_stats[local][j]
                mean_rms = sum(energy)/5
                cv = math.sqrt(sum((r-mean_rms)**2 for r in energy)/5)
                effects.append({**common,'component':component,'mean_abs':sum(absolute)/5,
                    'joint_rms':math.sqrt(sum(r*r for r in energy)/5),'view_rms':list(energy),
                    'view_mean_abs':list(absolute),'view_signed_mean':list(signed),'raw_variance':var,
                    'rms_normalized_variance':normalized_var if min(energy)>EPS else None,
                    'pair_raw_variance':pair,'pair_rms_normalized_variance':normalized_pair if min(energy[:2])>EPS else None,
                    'near_zero_view_count':sum(r<=EPS for r in energy),'rms_style_cv':_ratio(cv,mean_rms)})
            vf, va, vr = (v[0] for v in component_stats[local])
            correlations.append({**common,'component':'variance_decomposition','view_index':-1,
                'anchor_variance':va,'residual_variance':vr,'full_variance':vf,'style_covariance':cov,
                'variance_identity_abs_error':abs(vf-va-vr-2*cov)})
            raw, center = raw_gram[local], spatial_gram[local]
            for view in range(5):
                ff, aa, rr = (v[view][2] for v in view_stats[local])
                row = {**common,'component':'spatial','view_index':view,
                       'full_mean_abs':view_stats[local][0][view][1],'anchor_mean_abs':view_stats[local][1][view][1],
                       'difference_mean_abs':view_stats[local][2][view][1],'full_rms':ff,'anchor_rms':aa,'difference_rms':rr,
                       'residual_full_ratio':_ratio(rr,ff),'residual_anchor_ratio':_ratio(rr,aa),
                       'addition_cancellation_ratio':_ratio(ff,aa+rr),'difference_shrink_ratio':_ratio(rr,ff+aa)}
                pairs = [('anchor_residual',5+view,10+view),('full_anchor',view,5+view)]
                if view:
                    pairs += [('anchor_cross_style',5,5+view),('residual_cross_style',10,10+view),
                              ('delta_anchor_residual',14+view,18+view)]
                for name,i,j in pairs:
                    row.update({f'{name}_{k}':v for k,v in _similarity(raw,center,i,j).items()})
                correlations.append(row)
    return effects, correlations


def validate_reference(full, anchor, label, actual):
    """Compare device reductions with v1 CPU statistics on the same forward."""
    full = full.float().cpu()
    if anchor is None:
        expected = structural_effect_and_correlation_rows(full,label)
    else:
        anchor = anchor.float().cpu()
        maps = [{'full':f[None],'anchor':a[None],'residual':(f-a)[None]} for f,a in zip(full,anchor)]
        expected = effect_and_correlation_rows(maps,label)
    largest = 0.
    def compare(a,b):
        nonlocal largest
        if isinstance(a,dict):
            if a.keys()!=b.keys(): raise ValueError('Phase45 statistic columns differ')
            for k in a: compare(a[k],b[k])
        elif isinstance(a,list):
            if len(a)!=len(b): raise ValueError('Phase45 statistic coverage differs')
            for x,y in zip(a,b): compare(x,y)
        elif isinstance(a,float) and isinstance(b,float):
            largest = max(largest,abs(a-b))
            if not math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10):
                raise ValueError(f'Phase45 FP64 reference mismatch: {a} versus {b}')
        elif a!=b: raise ValueError(f'Phase45 metadata/null mismatch: {a} versus {b}')
    for a,b in zip(expected,actual):
        key=lambda r:(r['class_id'],r['component'],r.get('view_index',-1))
        compare(sorted(a,key=key),sorted(b,key=key))
    return largest
