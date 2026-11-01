"""At most two GPU windows, frozen features, logical batch one."""
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import time
import torch
from causalq.models.dinov3_wrapper import DINOv3Features


def feature_tensors(features):
    return (features.patch_map, features.patch_tokens, *features.intermediate_maps)


def slice_features(features, start, stop):
    return DINOv3Features(features.patch_map[start:stop], features.patch_tokens[start:stop],
                          tuple(x[start:stop] for x in features.intermediate_maps))


@dataclass
class FeatureWindow:
    samples: list
    features: DINOv3Features
    ready: object = None
    data_wait_seconds: float = 0.

    def logical(self):
        for i, sample in enumerate(self.samples):
            yield sample, slice_features(self.features, 2*i, 2*i+1), slice_features(self.features, 2*i+1, 2*i+2)


class FeaturePipeline:
    def __init__(self, backbone, profile, *, device='cuda:0', amp=True):
        self.backbone, self.profile, self.device, self.amp = backbone, profile, torch.device(device), amp
        if not backbone.freeze or any(p.requires_grad for p in backbone.parameters()):
            raise ValueError('Feature producer requires frozen backbone')
        if profile['backbone_microbatch_views'] % 2 or not 1 <= profile['feature_queue_windows'] <= 2:
            raise ValueError('Invalid bounded paired feature window')

    def __call__(self, loader):
        iterator = iter(loader)
        size = self.profile['backbone_microbatch_views'] // 2
        stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        queue = deque()
        exhausted = False
        while queue or not exhausted:
            # One queued window plus the caller's live consumption window <=2.
            # CPU loader prefetch overlaps decoding; GPU overlap awaits timing evidence.
            while len(queue) < 1 and not exhausted:
                start = time.perf_counter()
                samples = []
                for _ in range(size):
                    try:
                        samples.append(next(iterator))
                    except StopIteration:
                        exhausted = True
                        break
                if not samples:
                    break
                wait = time.perf_counter() - start
                views = torch.cat([s['views'] for s in samples])
                if stream is not None:
                    views = views.pin_memory() if not views.is_pinned() else views
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                        features = self.backbone(views.to(self.device, non_blocking=True))
                    ready = torch.cuda.Event() if stream is not None else None
                    if ready is not None:
                        ready.record(stream)
                queue.append(FeatureWindow(samples, features, ready, wait))
            if queue:
                window = queue.popleft()
                if stream is not None:
                    consumer = torch.cuda.current_stream(self.device)
                    consumer.wait_event(window.ready)
                    for x in feature_tensors(window.features):
                        x.record_stream(consumer)
                yield window
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                del window, features


def exact_shape_batches(items, batch_size):
    """Diagnostics: group exact H/W without padding or losing tails."""
    buckets = {}
    for key, tensor in items:
        bucket = buckets.setdefault(tuple(tensor.shape), [])
        bucket.append((key, tensor))
        if len(bucket) == batch_size:
            yield bucket[:]
            bucket.clear()
    for bucket in buckets.values():
        if bucket:
            yield bucket
