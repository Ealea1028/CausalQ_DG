"""Bounded decode and five-view preparation for the source holdout."""
import hashlib
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Dataset

from causalq.analysis.query_components import intervention_seed
from .input_stream import ThreadLoader, worker_init


def diagnostic_seeds(identity):
    return [intervention_seed(identity)] + [
        int(hashlib.sha256(f'diagnostic_20261020:photo-{view}:{identity}'.encode()).hexdigest()[:8], 16)
        for view in (2, 3, 4)
    ]


class SourceDiagnosticDataset(Dataset):
    def __init__(self, dataset, indices, style):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.style = style

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        seeds = diagnostic_seeds(sample['id'])
        image = sample['image'][None]
        views, traces = [image], []
        for seed in seeds:
            view, trace = self.style.photometric_view_with_trace(image, generator=torch.Generator().manual_seed(seed))
            views.append(view)
            traces.append(trace)
        return {'id': sample['id'], 'views': torch.cat(views), 'label': sample['label'][None],
                'seeds': seeds, 'traces': traces}


def keep_geometry(samples):
    # A loader packet may contain different H/W. GPU batches are bucketed later.
    return samples


def source_loader_policy(profile, workers, *, maximum_edge=1024, shm_bytes=None):
    prefetch = profile['prefetch_factor']
    requested = 32
    sample_bytes = (5 * 3 * 4 + 8) * maximum_edge**2
    # Worker-local scratch and a separate allowance for the main process, head
    # outputs and up to four partial geometry buckets.
    scratch = workers * (512 << 20) + (4 << 30)
    host_budget = profile['host_pipeline_budget_gib'] * (1 << 30)
    if workers == 0:
        return {'mode': 'single_process', 'workers': 0, 'batch_size': 1,
                'requested_batch_size': requested, 'sample_bound_bytes': sample_bytes}
    if shm_bytes is None:
        shm = Path('/dev/shm')
        shm_bytes = shutil.disk_usage(shm).free if shm.is_dir() else 0
    # Bound completed prefetched packets, in-flight packets and their temporary
    # worker/shared-memory copies. The IPC tensors remain pageable.
    bytes_per_packet_sample = sample_bytes * (workers * (prefetch + 2) + 2)
    host_capacity = (host_budget - scratch) // bytes_per_packet_sample
    ipc_capacity = int(.8 * shm_bytes) // bytes_per_packet_sample
    if host_capacity < 1:
        raise ValueError('Source diagnostics exceed the host loader budget')
    batch_size = min(requested, host_capacity, ipc_capacity)
    if batch_size < 1:
        mode, batch_size = 'threads', 1
    else:
        mode = 'spawn'
    return {'mode': mode, 'workers': workers, 'batch_size': batch_size, 'prefetch': prefetch,
            'requested_batch_size': requested, 'sample_bound_bytes': sample_bytes,
            'shm_free_bytes': shm_bytes, 'host_bound_bytes': scratch + batch_size * bytes_per_packet_sample,
            'pin_memory': False}


def source_loader(dataset, profile, *, workers, maximum_edge=1024, shm_bytes=None):
    policy = source_loader_policy(profile, workers, maximum_edge=maximum_edge, shm_bytes=shm_bytes)
    if policy['mode'] == 'threads':
        loader = ThreadLoader(dataset, workers, policy['prefetch'], False)
        return ([sample] for sample in loader), policy
    options = {'batch_size': policy['batch_size'], 'num_workers': workers, 'collate_fn': keep_geometry,
               'generator': torch.Generator().manual_seed(0)}
    if workers:
        options.update(multiprocessing_context='spawn', worker_init_fn=worker_init,
                       prefetch_factor=policy['prefetch'], timeout=180)
    return DataLoader(dataset, **options), policy
