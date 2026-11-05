"""Two bounded slots: CPU staging, copy stream, frozen producer, head consumer."""
from collections import deque
from dataclasses import dataclass
import time
import torch
from .feature_pipeline import FeatureWindow, feature_tensors


@dataclass
class Slot:
    window: object = None
    staging: object = None
    consumed: object = None


class DoubleBuffer:
    def __init__(self,backbone,profile,*,device='cuda:0',amp=True):
        self.backbone,self.profile,self.device,self.amp=backbone,profile,torch.device(device),amp
        self.size=profile['backbone_microbatch_views']//2
        if not backbone.freeze or any(p.requires_grad for p in backbone.parameters()) or self.size<1:
            raise ValueError('Double buffer requires a frozen backbone and positive size')
        if self.size%profile['loader_packet_samples']:
            raise ValueError('Window must align to packet size')
        self.peak_live_slots=0

    def __call__(self,loader):
        iterator=iter(loader)
        gpu=self.device.type=='cuda'
        consumer=torch.cuda.current_stream(self.device) if gpu else None
        copy=torch.cuda.Stream(device=self.device) if gpu else None
        producer=torch.cuda.Stream(device=self.device) if gpu else None
        if gpu:producer.wait_stream(consumer)
        slots=[Slot(),Slot()];pending=deque()
        exhausted=False

        def submit(slot):
            nonlocal exhausted
            # Reclaim only the slot being reused, including its unfinished CUDA work.
            if slot.consumed is not None and not slot.consumed.query():slot.consumed.synchronize()
            slot.staging=None
            start=time.perf_counter();packets=[];count=0
            while count<self.size:
                packet=next(iterator,None)
                if packet is None:exhausted=True;break
                packets.append(packet);count+=len(packet['metadata'])
            if not packets:return False
            if count>self.size:raise ValueError('Packet crossed a feature window')
            wait=time.perf_counter()-start;stage_start=time.perf_counter()
            def stage(key):
                first=packets[0][key]
                result=torch.empty((count,*first.shape[1:]),dtype=first.dtype,pin_memory=gpu and self.profile['pin_memory'])
                offset=0
                for packet in packets:
                    value=packet[key];result[offset:offset+len(value)].copy_(value);offset+=len(value)
                return result
            views,labels=stage('views'),stage('labels');stage_seconds=time.perf_counter()-stage_start
            meta=[m for packet in packets for m in packet['metadata']]
            timing=tuple(torch.cuda.Event(enable_timing=True) for _ in range(4)) if gpu else ()
            if gpu:
                with torch.cuda.stream(copy):
                    timing[0].record();device_views=views.flatten(0,1).to(self.device,non_blocking=True)
                    device_labels=labels.to(self.device,non_blocking=True);timing[1].record()
                producer.wait_event(timing[1]);device_views.record_stream(producer)
                with torch.cuda.stream(producer),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):
                    timing[2].record();features=self.backbone(device_views);timing[3].record()
            else:
                device_labels=labels
                with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=self.amp):features=self.backbone(views.flatten(0,1))
            samples=[{'metadata':m,'label':device_labels[i]} for i,m in enumerate(meta)]
            window=FeatureWindow(samples,features,timing[-1] if gpu else None,wait,timing)
            window.staging_seconds=stage_seconds
            window.collate_seconds=sum(p['collate_seconds'] for p in packets)
            window.pinned=views.is_pinned() and labels.is_pinned()
            slot.staging=(views,labels);slot.window=window;slot.consumed=None
            pending.append(slot)
            self.peak_live_slots=max(self.peak_live_slots,sum(s.window is not None for s in slots))
            return True

        try:
            submit(slots[0])
            if not exhausted:submit(slots[1])
            while pending:
                slot=pending.popleft();window=slot.window
                if gpu:
                    consumer.wait_event(window.ready)
                    for tensor in feature_tensors(window.features):tensor.record_stream(consumer)
                    for sample in window.samples:sample['label'].record_stream(consumer)
                yield window
                if gpu:
                    slot.consumed=torch.cuda.Event();slot.consumed.record(consumer)
                # Invalidate returned windows before a slot can allocate again.
                window.features=None;window.samples.clear();slot.window=None
                if not exhausted:submit(slot)
        finally:
            if gpu:
                copy.synchronize();producer.synchronize();consumer.synchronize()
            for slot in slots:
                if slot.window is not None:
                    slot.window.features=None;slot.window.samples.clear()
                slot.window=None;slot.staging=None
            close=getattr(iterator,'close',None)
            if close:close()
