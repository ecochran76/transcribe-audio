"""Supported, explicit checkpoint operations; no implicit recovery or model routing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import ssl
import time

import managed_recording_cli as recording
from managed_recording import RecordingProcessor, RecordingProcessingError


class PublicationOnly:
    def __getattr__(self, name):
        raise RuntimeError('inference forbidden in publication-only operation')


def failure_receipt(error, *, operation):
    """Use the manager's shared allowlist, never arbitrary exception strings."""
    from inference_manager.failure_receipts import exception_receipt
    return dict(state='failed', operation=operation,
                diagnostic=exception_receipt(error, code='cpu_operation_failed'))


def preflight(plan):
    if plan.get('schema_version') != 1 or plan.get('mode') not in ('publication','resume'):
        raise ValueError('explicit checkpoint operation required')
    source = Path(plan['source']).expanduser().resolve(strict=True)
    if recording._hash_file(source) != plan['source_sha256']:
        raise ValueError('source identity conflict')
    commands = plan.get('required_commands', [])
    if not isinstance(commands, list) or len(commands) > 16:
        raise ValueError('bounded required commands list')
    for command in commands:
        if not isinstance(command, str) or shutil.which(command) is None:
            raise ValueError('required command unavailable')
    resources = plan['resources']
    limits = dict(workers=(1, 2), gpu_budget_bytes=(1, 12*1024**3),
                  headroom_bytes=(2*1024**3, 16*1024**3), system_ram_bytes=(1, 12*1024**3))
    if set(resources) != set(limits):
        raise ValueError('explicit resource limits required')
    for name, (low, high) in limits.items():
        if type(resources[name]) is not int or not low <= resources[name] <= high:
            raise ValueError('resource limit exceeds approved pilot')
    cp = Path(plan['checkpoint']).expanduser().resolve(strict=True)
    if cp.is_symlink() or cp.stat().st_uid != os.getuid():
        raise ValueError('owned checkpoint required')
    value = json.loads(cp.read_text())
    if cp.stem != value['key'] or value['state'] not in ('ready','processing'):
        raise ValueError('ready checkpoint identity required')
    if value.get('publication_state') not in (None, 'complete'):
        raise ValueError('publication uncertain; reconcile first')
    if plan['mode']=='publication' and (value['state']!='ready' or not value.get('units')
            or not all('result' in entry for unit in value['units'] for entry in unit['stages'].values())):
        raise ValueError('all GPU results must already be durable')
    reference=value['identity'].get('metadata',{}).get('original_source',value['identity'])
    if (Path(reference['source']).expanduser().resolve()!=source
            or reference['sha256']!=plan['source_sha256']):
        raise ValueError('checkpoint source identity conflict')
    config = value['identity']['config']
    if Path(config['state_directory']).resolve() != cp.parent:
        raise ValueError('checkpoint state directory conflict')
    if value.get('publication_state') is None and (plan['mode']=='publication' or plan.get('cli_manifest')):
        manifest = json.loads(Path(plan['cli_manifest']).read_text())
        frozen = manifest['frozen']
        if (manifest.get('processor_key') != cp.stem
                or frozen['config']['processor'] != config
                or frozen['identity']['source'] != str(source)
                or frozen['identity']['sha256'] != plan['source_sha256']):
            raise ValueError('CLI manifest identity conflict')
        generation = Path(frozen['config']['output_directory'])/('recording-'+cp.stem)
        if generation.exists():
            raise ValueError('unreceipted output generation requires reconciliation')
    # No tokens are opened and no network client is constructed.
    return dict(source=source, checkpoint=cp, record=value)


def check_script(path):
    """Apply the existing pinned lint tool to a private script without importing it."""
    from scripts.check_local import ruff_command
    result = subprocess.run(ruff_command()+['check','--select','F','--output-format','json',str(path)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode not in (0, 1):
        raise RuntimeError('private script lint unavailable')
    findings = json.loads(result.stdout)
    return dict(passed=not findings, findings=[dict(code=f['code'], line=f['location']['row'])
                                              for f in findings])


def execute(plan, *, client=None):
    prepared = preflight(plan)
    cp, value = prepared['checkpoint'], prepared['record']
    if plan['mode']=='resume':
        wait=recording._bounded(plan,'wait_seconds',300,.01,86400)
        poll=recording._bounded(plan,'poll_seconds',.25,.01,60)
        if client is None:
            from inference_manager.service_client import ManagedClient
            connection=plan['connection']
            client=ManagedClient(connection['endpoint'],recording._private_read(
                Path(connection['token_file']),4096).decode().strip(),
                ssl_context=ssl.create_default_context(cafile=connection['ca_file']))
        processor=RecordingProcessor(value['identity']['config'],client=client)
        deadline=time.monotonic()+wait
        while time.monotonic()<deadline:
            value=processor.snapshot(cp.stem)
            if value['state']=='ready':
                break
            unfinished=[e for u in value['units'] for e in u['stages'].values() if 'result'not in e]
            if any(e['state']in('failed','cancelled','unknown','submitting') for e in unfinished):
                raise RecordingProcessingError('held checkpoint requires explicit reconciliation')
            try:
                processor.advance(cp.stem)
            except RecordingProcessingError:
                # Observe the same admitted IDs after a transient read failure.
                # snapshot on the next iteration holds terminal/uncertain states.
                pass
            time.sleep(min(poll,max(0,deadline-time.monotonic())))
        else:
            raise RecordingProcessingError('bounded observation deadline; identities retained')
        if not plan.get('cli_manifest'):
            return dict(state='ready',checkpoint=str(cp),recording_id=value['identity']['recording_id'])
    else:
        processor = RecordingProcessor(value['identity']['config'], client=PublicationOnly())
    def publish(output):
        frozen = json.loads(Path(plan['cli_manifest']).read_text())['frozen']
        if (frozen['identity']['source'] != str(prepared['source'])
                or frozen['identity']['sha256'] != plan['source_sha256']):
            raise ValueError('CLI source identity conflict')
        config = frozen['config']
        if config['processor'] != value['identity']['config']:
            raise ValueError('CLI processor configuration conflict')
        return recording._publish(output, cp.stem, prepared['source'], config,
                                  frozen['existing_transcript'])
    receipt = processor.publish(cp.stem, publish)
    for kind, path in receipt['output_paths'].items():
        if recording._hash_file(Path(path)) != receipt['sha256'][kind]:
            raise ValueError('published output digest conflict')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--check-script', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.check_script:
            result = check_script(args.check_script)
        else:
            plan = json.loads(args.plan.read_text())
            result = dict(state='ready') if args.preflight_only and preflight(plan) else execute(plan)
        code = 0 if result.get('passed', True) else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        result = failure_receipt(error, operation='publication-preflight' if args.preflight_only else 'publication')
        code = 1
    root = recording._private_directory(args.receipt.parent)
    recording._save(root/args.receipt.name, result)
    print(json.dumps(dict(state=result.get('state', 'complete'), receipt=str(args.receipt))))
    return code




def publish_sidecars(receipt, prior_artifact):
    """Publish additive immutable named copies; existing conflicting bytes hold."""
    prior=Path(prior_artifact)
    paths=receipt['output_paths']
    copies={}
    stem=prior.name.removesuffix('.transcript.json')+' LocalAI Diarized'
    for kind,extension in (('docx','.docx'),('txt','.txt'),('json','.transcript.json')):
        source=Path(paths[kind])
        if recording._hash_file(source)!=receipt['sha256'][kind]:
            raise ValueError('published output digest conflict')
        dest=prior.with_name(stem+extension)
        if dest.exists():
            if recording._hash_file(dest)!=receipt['sha256'][kind]:
                raise ValueError('existing diarized sidecar conflict')
        else:
            import tempfile
            fd,name=tempfile.mkstemp(dir=dest.parent,prefix='.diarized-')
            temporary=Path(name)
            try:
                with os.fdopen(fd,'wb') as out, source.open('rb') as inp:
                    shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
                if recording._hash_file(temporary)!=receipt['sha256'][kind]:
                    raise ValueError('sidecar copy digest conflict')
                try:
                    os.link(temporary,dest)
                except FileExistsError:
                    if recording._hash_file(dest)!=receipt['sha256'][kind]:
                        raise ValueError('existing diarized sidecar conflict') from None
                recording._sync_directory(dest.parent)
            finally:
                temporary.unlink(missing_ok=True)
        copies[kind]=str(dest)
    return copies


if __name__ == '__main__':
    raise SystemExit(main())
