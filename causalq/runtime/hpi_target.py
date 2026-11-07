"""Frozen seven-head Cityscapes evaluation with image-level recovery."""
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from causalq.datasets.label_space import read_index_mask, label_ids_to_train_ids
from causalq.datasets.segmentation import _tensorize
from causalq.metrics.miou import MeanIoU
from causalq.protocols.hpi_dinov2_20261031 import ROOT
from .cohort_runner import head_state
from .identity import file_sha, tree_digest, write_json
from .input_stream import worker_init

DIRECTORY = ROOT/'experiments/HPI_DINOV2_20261031'


def target_contract():
    contract = json.loads((DIRECTORY/'target_freeze.json').read_text())
    for name, expected in contract['files'].items():
        if file_sha(ROOT/name) != expected:
            raise ValueError(f'Frozen target dependency differs {name}')
    if list(contract['models']) != ['U0','UA','B1','B2','B3','B4','B5']:
        raise ValueError('Expected all seven fixed-final models')
    return contract


class NativeTargetDataset(Dataset):
    def __init__(self, root, ids, geometry=(1024, 2048)):
        self.root, self.ids, self.geometry = Path(root), tuple(ids), tuple(geometry)
        image_root = self.root/'leftImg8bit/val'
        actual = sorted(p.relative_to(image_root).as_posix().removesuffix('_leftImg8bit.png')
                        for p in image_root.rglob('*_leftImg8bit.png'))
        if actual != list(self.ids):
            raise ValueError('Cityscapes val membership differs from frozen 500 IDs')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image_path = self.root/'leftImg8bit/val'/f'{sample_id}_leftImg8bit.png'
        label_path = self.root/'gtFine/val'/f'{sample_id}_gtFine_labelIds.png'
        raw_ids = label_path.is_file()
        if not raw_ids:
            label_path = self.root/'gtFine/val'/f'{sample_id}_gtFine_labelTrainIds.png'
        with Image.open(image_path) as image, Image.open(label_path) as label:
            if image.size != self.geometry[::-1] or label.size != image.size:
                raise ValueError('Target must have native 1024x2048 image and label geometry')
            mask = read_index_mask(label)
            if raw_ids:
                mask = label_ids_to_train_ids(mask)
            image_tensor, labels = _tensorize(image.convert('RGB'), Image.fromarray(mask))
        if ((labels != 255) & ((labels < 0) | (labels >= 19))).any():
            raise ValueError('Invalid target train IDs')
        return {'id': sample_id, 'image': image_tensor, 'label': labels,
                'image_sha256': file_sha(image_path), 'label_sha256': file_sha(label_path)}


def load_final_head(model, checkpoint, record):
    if file_sha(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint file hash differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['identity'] != record['identity'] or state['cursor'] != 40000:
        raise ValueError('Frozen checkpoint training identity/cursor differs')
    if set(state['model']) != set(head_state(model)):
        raise ValueError('Frozen compact head coverage differs')
    incompatible = model.load_state_dict(state['model'], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith('backbone.') for k in incompatible.missing_keys):
        raise ValueError('Frozen compact head load differs')
    if tree_digest(head_state(model)) != record['head_sha256']:
        raise ValueError('Frozen head tensor digest differs')
    model.requires_grad_(False).eval()


@torch.inference_mode()
def evaluate_batches(heads, backbone, loader, out, identity, *, device='cuda', amp=True):
    out = Path(out)
    metrics = {name: MeanIoU(19) for name in heads}
    ids = []
    for batch in loader:
        pending = []
        for index, sample_id in enumerate(batch['id']):
            path = out/'records'/f'{sample_id}.json'
            key = {'identity': identity, 'id': sample_id,
                   'image_sha256': batch['image_sha256'][index], 'label_sha256': batch['label_sha256'][index]}
            if path.is_file():
                record = json.loads(path.read_text())
                if record['key'] != key or set(record['confusion']) != set(heads):
                    raise ValueError('Target recovery record identity differs')
            else:
                pending.append((index, path, key))
        if pending:
            images = batch['image'][[i for i, _, _ in pending]].to(device, non_blocking=True)
            labels = batch['label'][[i for i, _, _ in pending]]
            records = [{'key': key, 'confusion': {}} for _, _, key in pending]
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=amp):
                features = backbone(images)
                for name, head in heads.items():
                    logits = head.forward_features(features, images.shape[-2:]).logits
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite target logits')
                    predicted = logits.argmax(1).cpu()
                    for j, record in enumerate(records):
                        metric = MeanIoU(19)
                        metric.update(predicted[j], labels[j])
                        record['confusion'][name] = metric.confusion.tolist()
                    del logits, predicted
            del features, images
            if torch.device(device).type == 'cuda' and torch.cuda.max_memory_reserved() > torch.cuda.get_device_properties(device).total_memory*.85:
                raise RuntimeError('Target evaluation exceeded frozen 85% GPU memory budget')
            for (_, path, _), record in zip(pending, records, strict=True):
                write_json(path, record)
        for sample_id in batch['id']:
            if sample_id in ids:
                raise ValueError('Repeated target ID')
            record = json.loads((out/'records'/f'{sample_id}.json').read_text())
            for name, metric in metrics.items():
                matrix = torch.tensor(record['confusion'][name])
                if matrix.shape != (19,19) or matrix.dtype != torch.int64 or (matrix < 0).any():
                    raise ValueError('Invalid target confusion record')
                metric.confusion += matrix
            ids.append(sample_id)
        print('target_images', len(ids), flush=True)
    return metrics, ids


def evaluate_target(out, profile, *, data_root, pretrained_root, output_root, git_sha):
    from causalq.models.dinov2_wrapper import DINOv2Backbone
    from causalq.protocols.hpi_dinov2_20261031 import build_head
    from .hpi_phase35 import build_ablation_head
    from .hpi_prepare import configure
    configure()
    contract = target_contract()
    runtime = json.loads((DIRECTORY/'runtime.json').read_text())
    if (torch.__version__ != runtime['torch'] or torch.cuda.get_device_name() != runtime['gpu']
            or list(torch.cuda.get_device_capability()) != runtime['compute_capability']):
        raise ValueError('Frozen target runtime/GPU differs')
    from importlib.metadata import version
    import platform
    import subprocess
    import sys
    packages = {name:version(name) for name in ('numpy','Pillow','PyYAML','torchvision')}
    locked = {line.split('==')[0].lower():line.split('==')[1] for line in runtime['pip_freeze'] if '==' in line}
    if any(locked.get(name.lower()) != value for name,value in packages.items()):
        raise ValueError('Frozen target runtime package differs')
    write_json(Path(out)/'runtime_actual.json', {
        'python':sys.version,'platform':platform.platform(),'packages':packages,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'cuda_runtime':torch.version.cuda,
        'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)})
    weights = Path(pretrained_root)/'dinov2_vitl14/dinov2_vitl14_hpi_p16.pth'
    if file_sha(weights) != json.loads((DIRECTORY/'weight_identity.json').read_text())['converted_file_sha256']:
        raise ValueError('Frozen backbone file differs')
    backbone = DINOv2Backbone.from_pretrained(weights, dtype=torch.bfloat16).cuda()
    digest = tree_digest(backbone.state_dict())
    heads = {}
    for name, record in contract['models'].items():
        if digest != record['identity']['backbone']:
            raise ValueError('Frozen runtime backbone tensor digest differs')
        model = (build_head(backbone) if name in ('U0','UA') else
                 build_ablation_head(backbone, 'B3' if name in ('B4','B5') else name, None))
        load_final_head(model, Path(output_root)/record['run_directory']/record['checkpoint'], record)
        heads[name] = model.cuda().eval()
    ids = (ROOT/contract['target_manifest']).read_text().splitlines()
    dataset = NativeTargetDataset(Path(data_root)/'cityscapes', ids)
    loader = DataLoader(dataset, batch_size=contract['batch_size'], num_workers=profile['data_workers'],
                        worker_init_fn=worker_init, multiprocessing_context='spawn',
                        prefetch_factor=profile['prefetch_factor'], pin_memory=True,
                        generator=torch.Generator().manual_seed(0), timeout=180)
    identity = {'evaluation_git_sha': git_sha, 'target_freeze_sha256': file_sha(DIRECTORY/'target_freeze.json')}
    started = time.monotonic()
    metrics, covered = evaluate_batches(heads, backbone, loader, out, identity)
    if covered != ids or len(ids) != contract['target_samples']:
        raise ValueError('Incomplete target membership coverage')
    scores = {}
    for name, metric in metrics.items():
        record = contract['models'][name]
        if tree_digest(head_state(heads[name])) != record['head_sha256']:
            raise ValueError('Evaluation modified frozen head')
        result = {**metric.compute(), 'sample_count': len(ids), 'ids': ids,
                  'miou_percent': metric.compute()['miou']*100, 'confusion_matrix': metric.confusion.tolist(),
                  'valid_pixels': int(metric.confusion.sum()), 'checkpoint': record, 'geometry': contract['geometry']}
        write_json(Path(out)/name/'target_metrics.json', result)
        scores[name] = result['miou']
    if tree_digest(backbone.state_dict()) != digest:
        raise ValueError('Evaluation modified backbone')
    with (Path(out)/'class_iou.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','class_id','iou','iou_percent'])
        for name, metric in metrics.items():
            for category, value in enumerate(metric.compute()['class_iou']):
                writer.writerow([name,category,value,None if value is None else 100*value])
    write_json(Path(out)/'target_report.json', {
        'identity': identity, 'target_miou': scores, 'source_miou': {k:v['source_miou'] for k,v in contract['models'].items()},
        'comparisons_pp': {f'{a}_minus_{b}':100*(scores[a]-scores[b]) for a,b in contract['comparisons']},
        'formal_optimizer_updates':0, 'target_data_accessed':True, 'sample_count_per_model':len(ids),
        'invocation_evaluation_seconds':time.monotonic()-started,
        'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
