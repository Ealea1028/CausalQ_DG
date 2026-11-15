"""Bounded LoRA input staging and predeclared performance acceptance."""
from collections import deque
from contextlib import contextmanager
import csv
import math
import statistics
import time
import torch

CANDIDATES=({'name':'micro1_serial','microbatch':1,'buffer_slots':1},
            {'name':'micro4_double','microbatch':4,'buffer_slots':2},
            {'name':'micro8_double','microbatch':8,'buffer_slots':2})
BF16_BATCH_TOLERANCES={'atol':.1,'rtol':.05,'relative_l2':.02}
# BF16 physical-batch tolerances follow historical Phase33 calibration;
# zero-adapter equality remains exact, recovery remains 1e-6/1e-5.


def percentile(values,q):
    values=sorted(values)
    if not values: return None
    x=(len(values)-1)*q; lo=int(x); hi=math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def telemetry(path):
    with open(path) as stream: rows=list(csv.DictReader(stream))
    valid=[]
    for r in rows:
        try:
            value=float(r['utilization_percent'])
            if math.isfinite(value): valid.append(value)
        except (ValueError,TypeError): pass
    return {'samples':len(valid),'mean':statistics.mean(valid) if valid else None,
            'median':statistics.median(valid) if valid else None,'p10':percentile(valid,.1),'p90':percentile(valid,.9),
            'fraction_ge60':sum(x>=60 for x in valid)/len(valid) if valid else None,
            'fraction_ge70':sum(x>=70 for x in valid)/len(valid) if valid else None,
            'power_w_mean':_column_mean(rows,'power_w'),
            'memory_used_mib_mean':_column_mean(rows,'memory_used_mib')}


def _column_mean(rows,key):
    vals=[]
    for r in rows:
        try:
            value=float(r[key])
            if math.isfinite(value): vals.append(value)
        except (ValueError,TypeError): pass
    return statistics.mean(vals) if vals else None


def select(rows,memory_bytes):
    eligible=[r for r in rows if r.get('ok') and r['timed_seconds']>=120 and r.get('numerical_ok')
              and r['peak_reserved_bytes']<=.85*memory_bytes and r['telemetry']['samples']>=100]
    if not eligible: return {'selected':None,'performance_target_met':False,'reason':'No eligible candidate'}
    fastest=max(r['source_images_per_second'] for r in eligible)
    tied=[r for r in eligible if r['source_images_per_second']>=fastest*.95]
    winner=min(tied,key=lambda r:(r['peak_reserved_bytes'],-r['source_images_per_second']))
    baseline=next((r for r in rows if r['name']=='micro1_serial' and r.get('ok')),None)
    util=winner['telemetry']
    faster=baseline is not None and winner['source_images_per_second']>baseline['source_images_per_second']
    met=util['median']>=70 and util['p10']>=60 and winner['data_wait_fraction']<.1 and faster
    return {'selected':winner['name'],'performance_target_met':bool(met),'faster_than_baseline':faster,
            'source_images_per_second':winner['source_images_per_second'],'profile_frozen':False,
            'reason':'Engineering targets met, awaiting local evidence acceptance' if met else 'Return measured bottlenecks before formal training'}


class CudaTimings:
    def __init__(self): self.events=[]
    @contextmanager
    def __call__(self,name):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        try: yield
        finally:
            b.record(); self.events.append((name,a,b))
    def summarize(self):
        out={}
        for name,a,b in self.events: out[name]=out.get(name,0)+a.elapsed_time(b)
        return out


class InputPrefetch:
    """At most two input batches; never share/cache adapted features or graphs."""
    def __init__(self,loader,device,slots=2):
        if slots not in (1,2): raise ValueError('Input slots must be one or two')
        self.loader,self.device,self.slots=loader,torch.device(device),slots
        self.peak_live_slots=0
        self.total_data_wait_seconds=0.
        self.total_staging_seconds=0.
    def __iter__(self):
        gpu=self.device.type=='cuda'; it=iter(self.loader)
        stream=torch.cuda.Stream(device=self.device) if gpu else None
        consumer=torch.cuda.current_stream(self.device) if gpu else None
        queue=deque(); exhausted=False
        def fill():
            nonlocal exhausted
            start=time.perf_counter(); packet=next(it,None); wait=time.perf_counter()-start
            if packet is None: exhausted=True; return
            self.total_data_wait_seconds+=wait
            views,labels=packet['views'],packet['labels']
            if len(packet['metadata'])!=8: raise ValueError('Incomplete effective batch')
            stage_start=time.perf_counter()
            if gpu:
                views=views if views.is_pinned() else views.pin_memory()
                labels=labels if labels.is_pinned() else labels.pin_memory()
                with torch.cuda.stream(stream):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record(); dv=views.to(self.device,non_blocking=True); dl=labels.to(self.device,non_blocking=True); end.record()
                ready=end
            else: dv,dl,ready,begin,end=views,labels,None,None,None
            queue.append({'views':dv,'labels':dl,'metadata':packet['metadata'],'host':(views,labels),
                          'ready':ready,'copy_events':(begin,end),'data_wait_seconds':wait,
                          'staging_seconds':time.perf_counter()-stage_start})
            self.total_staging_seconds+=queue[-1]['staging_seconds']
            self.peak_live_slots=max(self.peak_live_slots,len(queue))
        try:
            while len(queue)<self.slots and not exhausted: fill()
            while queue:
                current=queue.popleft()
                if gpu:
                    consumer.wait_event(current['ready'])
                    current['views'].record_stream(consumer); current['labels'].record_stream(consumer)
                yield current
                # Host pin buffers cannot be released until copy completion.
                if gpu: current['ready'].synchronize()
                del current
                if not exhausted: fill()
        finally:
            if stream is not None: stream.synchronize()
            queue.clear()
            close=getattr(it,'close',None)
            if close: close()
