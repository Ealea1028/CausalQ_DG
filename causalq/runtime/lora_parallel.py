"""CPU resource budgeting and an owned optional MPS service; no model changes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from causalq.runtime.hpi_packets import packet_policy

GIB = 1 << 30


def hardware():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        amount, period = quota.read_text().split()
        if amount != 'max': cpus = min(cpus, max(1, int(amount)//int(period)))
    else:
        for folder in ('cpu', 'cpu,cpuacct'):
            v1 = Path('/sys/fs/cgroup')/folder/'cpu.cfs_quota_us'
            if v1.exists() and int(v1.read_text()) > 0:
                period = int((v1.parent/'cpu.cfs_period_us').read_text())
                cpus = min(cpus, max(1, int(v1.read_text())//period))
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])*1024
    limit = Path('/sys/fs/cgroup/memory.max'); current = Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
        available = min(available, max(0, int(limit.read_text())-int(current.read_text())))
    elif not limit.exists():
        v1 = Path('/sys/fs/cgroup/memory')
        if (v1/'memory.limit_in_bytes').exists() and (v1/'memory.usage_in_bytes').exists():
            available = min(available, max(0, int((v1/'memory.limit_in_bytes').read_text())
                                           -int((v1/'memory.usage_in_bytes').read_text())))
    return {'cpu_budget': cpus, 'available_host_bytes': available,
            'dev_shm_free_bytes': shutil.disk_usage('/dev/shm').free if Path('/dev/shm').is_dir() else 0}


def resource_plan(requested, machine, base):
    if requested not in (1, 2, 4): raise ValueError('Concurrency must be1,2 or4')
    for jobs in (j for j in (4, 2, 1) if j <= requested):
        # Reserve model loading/runtime memory before budgeting the input pipeline.
        host = min(64*GIB, int(max(0, machine['available_host_bytes']-(4+2*jobs)*GIB)*.7))
        pinned = min(16*GIB, host//3)
        max_workers = min(12, max(0, (machine['cpu_budget']-1)//jobs))
        for workers in range(max_workers, -1, -1):
            profile = dict(base, data_workers=workers, pinned_memory_budget_gib=pinned/jobs/GIB,
                           host_pipeline_budget_gib=host/jobs/GIB)
            try:
                policy = packet_policy(profile, shm_bytes=machine['dev_shm_free_bytes']//jobs)
            except ValueError: continue
            return {'requested_jobs': requested, 'jobs': jobs, 'workers_per_job': workers,
                    'total_workers': jobs*workers, 'profile': profile, 'loader_policy_estimate': policy,
                    'hardware': machine, 'host_pipeline_budget_bytes': host, 'pinned_budget_bytes': pinned}
    raise RuntimeError('Insufficient host memory even for a bounded single-model pipeline')


class MPS:
    """Use a private daemon only. Never change compute mode or stop a global MPS."""
    def __init__(self, output, mode='auto'):
        self.output, self.mode = Path(output), mode
        self.env = os.environ.copy(); self.started = False
        self.record = {'requested': mode, 'enabled': False}

    def __enter__(self):
        binary = shutil.which('nvidia-cuda-mps-control')
        if self.mode == 'off' or binary is None:
            self.record['reason'] = 'disabled' if self.mode == 'off' else 'MPS command unavailable; ordinary CUDA processes'
            return self
        # Control sockets need a short Unix path; a long experiment path can
        # exceed sockaddr_un's path limit. Keep ordinary log files in the run.
        pipe = Path(tempfile.mkdtemp(prefix='cq42mps_'))
        logs = self.output/'mps_logs'/uuid.uuid4().hex[:12]
        logs.mkdir(parents=True)
        self.env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logs))
        self.record.update(binary=binary, directory=str(pipe), log_directory=str(logs))
        try:
            launch = subprocess.run([binary, '-d'], env=self.env, capture_output=True, text=True, timeout=20)
            self.started = launch.returncode == 0
            self.record.update(startup_exit_code=launch.returncode, startup_output=launch.stdout+launch.stderr)
            if self.started:
                check = subprocess.run([binary], input='get_server_list\n', env=self.env,
                                       capture_output=True, text=True, timeout=10)
                self.record['enabled'] = check.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.record['startup_error'] = str(exc)
        if not self.record['enabled']:
            if self.started: self.__exit__(None, None, None)
            self.env = os.environ.copy(); self.record['reason'] = 'private MPS startup failed; ordinary CUDA processes'
        return self

    def __exit__(self, *args):
        if self.started:
            try:
                result = subprocess.run([self.record['binary']], input='quit\n', env=self.env,
                                        capture_output=True, text=True, timeout=15)
                self.record['shutdown_exit_code'] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.record['shutdown_error'] = str(exc)
            self.started = False
