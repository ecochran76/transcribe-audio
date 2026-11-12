"""Explicit, packet-bound native smoke harness; never used by source tests.

Run inside a MemoryMax=12G, CPUQuota=400% user scope while holding the shared
native lock. The packet and all outputs must remain outside the checkout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import stat
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    path.chmod(0o600)


def packet_telemetry(packet):
    """Select the prepared controller's private configuration without fallback."""
    if 'telemetry_config_path' not in packet:
        return packet.get('telemetry')
    path = Path(packet['telemetry_config_path'])
    if not path.is_absolute() or 'telemetry' in packet:
        raise ValueError('one explicit private telemetry configuration required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size > 65536):
            raise ValueError('private owned bounded telemetry configuration required')
        return json.loads(stream.read(65537))


def gpu(telemetry=None, *, max_age_seconds=2):
    if (type(max_age_seconds) not in (int, float) or not math.isfinite(max_age_seconds)
            or not 0 < max_age_seconds <= 5):
        raise ValueError('telemetry age must be finite and in (0,5]')
    if telemetry is not None:
        from inference_manager.windows_telemetry import WindowsGpuTelemetry
        expected = {'endpoint', 'host_id', 'gpu_uuid', 'token_path', 'expected_source'}
        if not isinstance(telemetry, dict) or set(telemetry) != expected:
            raise ValueError('explicit pinned telemetry configuration required')
        token_path = Path(telemetry['token_path'])
        info = token_path.lstat()
        if (not token_path.is_absolute() or token_path.is_symlink()
                or not token_path.is_file() or info.st_uid != os.getuid()
                or info.st_mode & 0o077):
            raise ValueError('private owned telemetry token required')
        client = WindowsGpuTelemetry(telemetry['endpoint'],
            host_id=telemetry['host_id'], gpu_uuid=telemetry['gpu_uuid'],
            expected_source=telemetry['expected_source'],
            token=token_path.read_text().strip(), max_age_seconds=max_age_seconds)
        try:
            sample = client.sample()
        except Exception as error:
            error.capacity_rejection = dict(error_type=type(error).__name__,
                expected_source=telemetry['expected_source'], age_limit_seconds=max_age_seconds,
                rejected_sample_available=False)
            raise
        return dict(free_bytes=sample['gpu']['free_bytes'],
                    used_bytes=sample['gpu']['used_bytes'],
                    observed_at=sample['observed_at'], source=sample['source'],
                    age_seconds=sample['age_seconds'], generation=sample.get('generation'),
                    sequence=sample.get('sequence'), sampled_at_utc=sample.get('sampled_at_utc'))
    raw = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=memory.free,memory.used',
                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    free, used = [int(value.strip()) * 1024 ** 2 for value in raw.strip().split(',')]
    return dict(free_bytes=free, used_bytes=used, observed_at=time.monotonic())


def native_limits(packet):
    """Retain original packet budgets while bounding ASR and pilot successors."""
    attempts = packet.get('max_physical_stage_load_attempts', 4)
    seconds = packet.get('limits', {}).get('case_seconds', 600)
    cleanup = packet.get('limits', {}).get('cleanup_seconds', 15)
    if (type(attempts) is not int or not 1 <= attempts <= 15
            or type(seconds) not in (int, float) or not 0 < seconds <= (160 if attempts > 4 else 600)
            or type(cleanup) not in (int, float) or not 0 < cleanup <= 15):
        raise ValueError('bounded native attempt/runtime/cleanup limits required')
    return attempts, seconds, cleanup


def worker_exec(packet_path, command):
    packet = json.loads(packet_path.read_text())
    sys.path.insert(0, packet['localai_qualification'])
    attempt_limit, _, _ = native_limits(packet)
    stage = command[command.index('--stage') + 1]
    root = packet_path.parent
    with (root/'attempts.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = root/'attempts.json'
        attempts = json.loads(ledger.read_text()) if ledger.exists() else []
        if len(attempts) >= attempt_limit:
            raise RuntimeError('physical stage-load allowance exhausted')
        sample = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
        required = packet['registration']['stages'][stage]['reserved_bytes']
        if sample['free_bytes'] < required + packet['headroom_bytes']:
            raise RuntimeError('fresh native headroom gate failed')
        attempts.append(dict(stage=stage, pid=os.getpid(), parent=os.getppid(),
                             started_at=time.time(), gpu=sample, command=command))
        save(ledger, attempts)
        number = len(attempts)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    fd = os.open(root/f'worker-{number}-{stage}.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(fd, 2)
    os.close(fd)
    os.execv(command[0], command)


def run(packet_path):
    packet = json.loads(packet_path.read_text())
    root = packet_path.parent
    _, case_seconds, cleanup_seconds = native_limits(packet)
    case_deadline = time.monotonic()+case_seconds
    # The launcher must impose actual RAM/CPU bounds, not CUDA-incompatible AS limits.
    relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    cgroup = Path('/sys/fs/cgroup')/relative.lstrip('/')
    if int((cgroup/'memory.max').read_text()) > 12 * 1024 ** 3:
        raise RuntimeError('12GiB cgroup limit required')
    quota, period = (cgroup/'cpu.max').read_text().split()
    if quota == 'max' or int(quota) / int(period) > 4:
        raise RuntimeError('four CPU cgroup limit required')
    properties = subprocess.check_output(['systemctl', '--user', 'show', packet['systemd_unit'],
        '--property=MemoryMax,CPUQuotaPerSecUSec,RuntimeMaxUSec,TimeoutStopUSec'], text=True, timeout=5)
    save(root/(packet_path.stem+'-resource-control.json'), dict(cgroup=str(cgroup), properties=properties,
        memory_max=(cgroup/'memory.max').read_text().strip(), cpu_max=(cgroup/'cpu.max').read_text().strip()))
    for path, digest in packet['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen source changed')
    sys.path.insert(0, packet['localai_qualification'])
    sys.path.insert(0, packet['application_root'])
    from inference_manager import Capacity, Principal
    from inference_manager.adapters import ProcessAdapter
    from inference_manager.shared_service import SharedService, ConsumerPolicy, WorkloadPolicy
    from inference_manager.recording_runtime import register_recording_profiles
    from managed_recording import RecordingProcessor
    operator, consumer = Principal('native0173', 'operator'), Principal('native0173', 'recording')
    def capacity():
        sample = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
        return Capacity(max(0, sample['free_bytes']-packet['headroom_bytes']), sample['observed_at'])
    service = SharedService(root/packet_path.stem/'manager', budget_bytes=4*1024**3,
        capacity=capacity, clock=time.monotonic,
        operator=operator, consumers=[ConsumerPolicy(consumer, ('native-asr','native-diarization'), ('historical',))],
        workloads=[WorkloadPolicy('historical', 3, 'background', wait_seconds=case_seconds, max_run_seconds=case_seconds)])
    adapters = []
    def adapter_factory(command, directory, **kwargs):
        adapter = ProcessAdapter([sys.executable, str(Path(__file__).resolve()), '--exec-worker',
                                  str(packet_path), *command], directory, **kwargs)
        adapters.append(adapter)
        return adapter
    report = dict(packet_sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                  started_at=time.time(), native_execution=True, timings=[], observations=[])
    try:
        bindings = register_recording_profiles(service, packet['registration'], adapter_factory=adapter_factory)
        report['bindings'] = bindings
        class Client:
            def submit(self, request): return service.submit(consumer, request)
            def observe(self, request_id): return service.observe(consumer, request_id)
            def result(self, request_id): return service.result(consumer, request_id)
        if packet.get('only_stage'):
            stage = packet['only_stage']
            config = packet['registration']['stages'][stage]
            sample = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
            report['pre_admission_gpu'] = sample
            if sample['free_bytes'] < config['reserved_bytes'] + packet['headroom_bytes']:
                raise RuntimeError('fresh native headroom admission refusal')
            request = dict(id='native0173-'+stage, **bindings[stage], workload_class='historical',
                payload=dict(artifact_id=Path(packet['input_wav']).name,
                             sha256=packet['input_sha256'], options={}))
            service.submit(consumer, request)
            deadline = case_deadline
            sampled_at = 0
            while time.monotonic() < deadline:
                service.tick()
                view = service.observe(consumer, request['id'])
                if time.monotonic()-sampled_at >= 1:
                    sample = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
                    report['gpu_samples'] = report.get('gpu_samples', [])+[sample]
                    sampled_at = time.monotonic()
                    if sample['free_bytes'] < packet['headroom_bytes']:
                        raise RuntimeError('loaded GPU headroom floor violated')
                report['observations'].append(dict(at=time.time(), job=view,
                    resources=service.manager.resources(operator)))
                save(root/(packet_path.stem+'-progress.json'), report)
                if view['state'] == 'succeeded':
                    report['stage_result'] = service.result(consumer, request['id'])
                    report['verdict'] = 'native_stage_smoke_complete'
                    return
                if view['state'] in ('failed','cancelled','unknown','unavailable'):
                    raise RuntimeError('native stage '+view['state']+': '+str(view.get('reason')))
                time.sleep(.2)
            raise RuntimeError('native stage runtime bound exceeded')
        config = dict(packet['processor'], profiles=bindings)
        processor = RecordingProcessor(config, client=Client())
        key = processor.prepare(packet['input_wav'], recording_id='native0173-smoke',
                                conversation_id='native0173-smoke')
        report['recording_key'] = key
        previous = None
        stage_start = time.monotonic()
        sample_time = 0
        for iteration in range(100000):
            record = processor.snapshot(key)
            current = next((stage for stage in ('asr','diarization')
                            if any('result' not in unit['stages'][stage] for unit in record['units'])), None)
            if current != previous:
                if previous is not None:
                    report['timings'].append(dict(stage=previous, elapsed_seconds=time.monotonic()-stage_start))
                    cleanup = service.stop_owned(timeout_seconds=cleanup_seconds)
                    report['observations'].append(dict(at=time.time(), cleanup=cleanup))
                    if cleanup['models'] or cleanup['reserved_bytes']:
                        raise RuntimeError('stage release unproved')
                previous, stage_start = current, time.monotonic()
            if time.monotonic() > case_deadline:
                raise RuntimeError('native stage runtime bound exceeded')
            service.tick()
            if time.monotonic()-sample_time >= 1:
                sample = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
                if sample['free_bytes'] < packet['headroom_bytes']:
                    raise RuntimeError('loaded GPU headroom floor violated')
                report['observations'].append(dict(at=time.time(), gpu=sample, resources=service.manager.resources(operator)))
                save(root/(packet_path.stem+'-progress.json'), report)
                sample_time = time.monotonic()
            record = processor.advance(key)
            if record['state'] == 'ready':
                report['output'] = record['output']
                report['verdict'] = 'native_smoke_complete'
                break
            time.sleep(.05)
        else:
            raise RuntimeError('bounded observation count exceeded')
    except BaseException as error:
        report['verdict'] = 'failed'
        report['error'] = repr(error)
        if hasattr(error, 'capacity_rejection'):
            report['capacity_rejection'] = dict(error.capacity_rejection,
                last_accepted=report.get('gpu_samples', [])[-1:] or [report.get('pre_admission_gpu')])
        raise
    finally:
        report['cleanup'] = service.stop_owned(timeout_seconds=cleanup_seconds)
        report['cleanup_proved'] = not report['cleanup']['models'] and report['cleanup']['reserved_bytes'] == 0
        if not report['cleanup_proved']:
            report['verdict'] = 'cleanup_unproved'
        try:
            report['final_gpu'] = gpu(packet_telemetry(packet), max_age_seconds=packet.get('telemetry_max_age_seconds', 2))
        except Exception as error:
            report['final_gpu_error'] = repr(error)
        ledger = root/'attempts.json'
        report['worker_exit'] = [dict(pid=a['pid'], process_absent=not Path('/proc',str(a['pid'])).exists())
                                 for a in json.loads(ledger.read_text())] if ledger.exists() else []
        report['cgroup_memory_peak'] = (cgroup/'memory.peak').read_text().strip()
        report['cgroup_memory_events'] = (cgroup/'memory.events').read_text()
        report['completed_at'] = time.time()
        save(root/(packet_path.stem+'-result.json'), report)
        service.close()
        for adapter in adapters:
            adapter.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--exec-worker':
        worker_exec(Path(sys.argv[2]).resolve(), sys.argv[3:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    run(args.packet.resolve())


if __name__ == '__main__':
    main()
