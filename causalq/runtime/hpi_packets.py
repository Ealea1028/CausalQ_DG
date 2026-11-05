"""Ordered CPU packets with a separately bounded pinned staging budget."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import Dataset, DataLoader
from .input_stream import ThreadLoader, worker_init


def packet_collate(samples):
    start=time.perf_counter()
    result={'views':torch.stack([s['views'] for s in samples]),
            'labels':torch.stack([s['label'] for s in samples]),
            'metadata':[s['metadata'] for s in samples]}
    result['collate_seconds']=time.perf_counter()-start
    return result


@dataclass
class PacketDataset(Dataset):
    stream: object
    packet_size: int = 8

    def __len__(self):
        return (len(self.stream)+self.packet_size-1)//self.packet_size

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        start=index*self.packet_size
        return packet_collate([self.stream[i] for i in range(start,min(start+self.packet_size,len(self.stream)))])


def packet_policy(profile,*,shm_bytes=None,shape=(512,512)):
    workers=profile['data_workers'];prefetch=profile['prefetch_factor'];packet=profile['loader_packet_samples']
    h,w=shape
    # Two FP32 RGB views and one int64 label, plus one active packet/worker.
    sample_bytes=(2*3*4+8)*h*w
    shm_need=sample_bytes*packet*workers*(prefetch+1)
    window_bytes=sample_bytes*(profile['backbone_microbatch_views']//2)
    pinned=(sample_bytes*packet*(workers*prefetch+2)+2*window_bytes) if profile['pin_memory'] else 0
    host=shm_need+pinned+workers*(512<<20)+2*window_bytes
    if pinned>profile['pinned_memory_budget_gib']*(1<<30) or host>profile['host_pipeline_budget_gib']*(1<<30):
        raise ValueError('Packet/staging memory budget exceeded')
    if shm_bytes is None:shm_bytes=shutil.disk_usage('/dev/shm').free if Path('/dev/shm').is_dir() else 0
    threaded=bool(workers and shm_bytes<shm_need)
    return {'mode':'threads' if threaded else 'spawn' if workers else 'single_process',
            'reason':'insufficient /dev/shm' if threaded else None,'workers':workers,'prefetch':prefetch,
            'packet_samples':packet,'shared_memory_required_bytes':shm_need,'shared_memory_available_bytes':shm_bytes,
            'pinned_bound_bytes':pinned,'host_bound_bytes':host,'sample_tensor_bytes':sample_bytes}


def packet_loader(stream,profile,*,shm_bytes=None):
    policy=packet_policy(profile,shm_bytes=shm_bytes,shape=stream.transform.crop_size)
    dataset=PacketDataset(stream,profile['loader_packet_samples'])
    if policy['mode']=='threads':
        # Thread fallback does not pin individual packets: window staging does it once.
        return ThreadLoader(dataset,policy['workers'],policy['prefetch'],False),policy
    options={'batch_size':None,'num_workers':policy['workers'],'pin_memory':profile['pin_memory'],
             'generator':torch.Generator().manual_seed(0)}
    if policy['workers']:
        options.update(multiprocessing_context='spawn',worker_init_fn=worker_init,prefetch_factor=policy['prefetch'],persistent_workers=True)
    return DataLoader(dataset,**options),policy
