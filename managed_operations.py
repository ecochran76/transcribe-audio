"""Supported, explicit checkpoint operations; no implicit recovery or model routing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import managed_recording_cli as recording
from managed_recording import RecordingProcessor


class PublicationOnly:
    def __getattr__(self, name):
        raise RuntimeError('inference forbidden in publication-only operation')


def failure_receipt(error, *, operation):
    """Metadata only: never serialize messages, locals, paths or environment."""
    return dict(schema_version=1, state='failed', operation=operation,
                exception_type=type(error).__name__, errno=getattr(error, 'errno', None),
                frames=[dict(module=Path(f.filename).name, function=f.name, line=f.lineno)
                        for f in traceback.extract_tb(error.__traceback__)[-8:]])


def preflight(plan):
    if plan.get('schema_version') != 1 or plan.get('mode') != 'publication':
        raise ValueError('explicit publication operation required')
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
    if cp.stem != value['key'] or value['state'] != 'ready':
        raise ValueError('ready checkpoint identity required')
    if value.get('publication_state') not in (None, 'complete'):
        raise ValueError('publication uncertain; reconcile first')
    if not value.get('units') or not all('result' in entry for unit in value['units']
                                      for entry in unit['stages'].values()):
        raise ValueError('all GPU results must already be durable')
    config = value['identity']['config']
    if Path(config['state_directory']).resolve() != cp.parent:
        raise ValueError('checkpoint state directory conflict')
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


def execute(plan):
    prepared = preflight(plan)
    cp, value = prepared['checkpoint'], prepared['record']
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


if __name__ == '__main__':
    raise SystemExit(main())
