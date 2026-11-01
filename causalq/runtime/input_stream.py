"""Absolute-step paired source input; independent of workers and head RNG."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from causalq.datasets.segmentation import TrainTransform, align_scale_equivalent_label
from causalq.datasets.label_space import read_index_mask
from causalq.interventions.style import StyleInterventionBank

STREAM_VERSION = 'hpi_step_sha256_v1'


@lru_cache(maxsize=4)
def epoch_order(length, seed, epoch):
    return tuple(torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist())


def input_seed(seed, step, sample_id, view):
    value = json.dumps([STREAM_VERSION, int(seed), int(step), sample_id, view], separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2 ** 63 - 1)


class CanonicalInputStream(Dataset):
    def __init__(self, pairs, *, seed=0, start=0, stop=40000, transform=None):
        self.pairs = tuple(pairs)
        if not self.pairs or not 0 <= start < stop:
            raise ValueError('Invalid source stream bounds')
        self.seed, self.start, self.stop = seed, start, stop
        self.transform = transform or TrainTransform((512, 512))
        self.style = StyleInterventionBank()

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        step = self.start + index
        epoch, cursor = divmod(step, len(self.pairs))
        # A local generator is safe even when thread workers complete out of order.
        order = epoch_order(len(self.pairs), self.seed, epoch)
        pair = self.pairs[int(order[cursor])]
        seeds = {v: input_seed(self.seed, step, pair.sample_id, v) for v in ('geometry', 'photometric')}
        with Image.open(pair.image) as raw, Image.open(pair.label) as target:
            image = raw.convert('RGB')
            label = align_scale_equivalent_label(image, Image.fromarray(read_index_mask(target)))
            values = np.asarray(label)
            if np.any((values != 255) & (values >= 19)):
                raise ValueError('Source label outside 19 trainIds')
            image, label = self.transform(image, label, generator=torch.Generator().manual_seed(seeds['geometry']))
        if not (label != 255).any():
            raise ValueError('Source crop has no valid label')
        photo, trace = self.style.photometric_view_with_trace(image[None], generator=torch.Generator().manual_seed(seeds['photometric']))
        metadata = {'stream': STREAM_VERSION, 'seed': self.seed, 'step_zero_based': step,
                    'id': pair.sample_id, 'epoch': epoch, 'seeds': seeds, 'photometric': trace}
        metadata['id_parameters_sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return {'views': torch.stack((image, photo[0])), 'label': label, 'metadata': metadata}


def worker_init(_):
    torch.set_num_threads(1)


def loader_policy(profile, *, shm_bytes=None):
    workers, prefetch = profile['data_workers'], profile['prefetch_factor']
    pinned = max(1, workers) * prefetch * 10 * 2 ** 20
    host = max(1, workers) * 512 * 2 ** 20 + pinned
    if pinned > profile['pinned_memory_budget_gib'] * 2 ** 30 or host > profile['host_pipeline_budget_gib'] * 2 ** 30:
        raise ValueError('Input pipeline exceeds aggregate host/pinned budget')
    if shm_bytes is None:
        import shutil
        shm_bytes = shutil.disk_usage('/dev/shm').free if Path('/dev/shm').exists() else 0
    return {'mode': 'threads' if workers and shm_bytes < pinned * 2 else 'spawn', 'workers': workers,
            'torch_threads_per_process':1 if workers and shm_bytes < pinned * 2 else profile['cpu_main_threads'],
            'estimated_pinned_bytes': pinned, 'estimated_host_bytes': host, 'shm_available_bytes': shm_bytes}


def make_loader(dataset, profile, *, shm_bytes=None):
    policy = loader_policy(profile, shm_bytes=shm_bytes)
    workers = policy['workers']
    if policy['mode'] == 'threads':
        return ThreadLoader(dataset, workers, profile['prefetch_factor'], profile['pin_memory'])
    options = dict(batch_size=None, num_workers=workers, pin_memory=profile['pin_memory'],
                   generator=torch.Generator().manual_seed(0))
    if workers:
        options.update(multiprocessing_context='spawn', worker_init_fn=worker_init,
                       prefetch_factor=profile['prefetch_factor'], persistent_workers=True)
    return DataLoader(dataset, **options)


class ThreadLoader:
    """Bounded ordered futures; transforms use local generators exclusively."""
    def __init__(self, dataset, workers, prefetch, pin):
        self.dataset, self.workers, self.prefetch, self.pin = dataset, workers, prefetch, pin

    def __iter__(self):
        from collections import deque
        previous_threads = torch.get_num_threads()
        # Intraop pools are process-wide: fallback uses one Torch thread in
        # the whole decoder process instead of oversubscribing every worker.
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(self.workers) as pool:
                pending = deque()
                index = 0
                while pending or index < len(self.dataset):
                    while index < len(self.dataset) and len(pending) < self.workers * self.prefetch:
                        pending.append(pool.submit(self.dataset.__getitem__, index)); index += 1
                    sample = pending.popleft().result()
                    if self.pin and torch.cuda.is_available():
                        sample = {**sample, 'views': sample['views'].pin_memory(), 'label': sample['label'].pin_memory()}
                    yield sample
        finally:
            torch.set_num_threads(previous_threads)
