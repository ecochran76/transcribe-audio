#!/usr/bin/env python3
"""Refresh reviewed ownership, install tracked code, and account initial placement.

No inference, recursive adoption, source erasure or replica provisioning occurs.
The private adoption document is an explicit inventory authority, not a boolean
assertion that every object under a directory is inactive.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_maintenance import Maintenance, config_digest, plain, private_json, write_receipt


def owned(path, config):
    path = plain(path)
    if not any(path.is_relative_to(plain(root)) for root in config['source_roots']):
        raise ValueError('inventory_path_outside_owned_roots')
    return path


def refresh(config_path, adoption_path):
    """Reconcile current registered blobs and exact completed frozen derivatives.

    Adoption entries bind id/path/hash and explicit custody evidence. New blobs
    without that evidence remain held. Processor directories are explicitly
    reviewed; publication and immutable CLI manifests qualify each new WAV.
    """
    config = private_json(config_path)
    adoption = private_json(adoption_path)
    if adoption.get('version') != 1 or adoption.get('scope_roots') != config['source_roots']:
        raise ValueError('reviewed_adoption_scope_mismatch')
    objects, held = {}, []
    catalog = Maintenance(config, config_path=config_path).catalog
    approved = {entry['id']: entry for entry in adoption.get('objects', [])}
    with closing(sqlite3.connect(Path(config['live_database']).as_uri() + '?mode=ro', uri=True)) as db:
        for identity, path, digest in db.execute('SELECT id,stored_path,sha256 FROM blobs'):
            try:
                path = catalog.object(identity)['logical_path']
            except LookupError:
                pass
            path = owned(path, config)
            item = approved.get(identity, {})
            if (item.get('path') != str(path) or item.get('sha256') != digest
                    or not item.get('custody_evidence') or type(item.get('active')) is not bool):
                held.append({'id': identity, 'path': str(path), 'reason': 'registered_blob_custody_unqualified'})
                continue
            objects[identity] = {**item, 'ownership_known': True, 'classification': 'canonical_source'}
    for processor in adoption.get('processors', []):
        root = owned(processor['path'], config)
        if not processor.get('custody_evidence'):
            raise ValueError('processor_custody_evidence_required')
        completed_records = {}
        for checkpoint in root.glob('*.json'):
            record = private_json(checkpoint)
            if record.get('publication_state') == 'complete':
                rid = record.get('identity', {})
                completed_records[rid.get('source')] = (checkpoint.stem, rid)
        for manifest_path in sorted((root / 'cli').glob('*.json')):
            manifest = private_json(manifest_path)
            path = owned(root / 'cli' / 'media' / (manifest_path.stem + '.wav'), config)
            identity = 'wav-' + manifest_path.stem
            frozen = manifest.get('frozen', {}).get('identity', {})
            digest = manifest.get('decoded_sha256')
            # Completed authoritative processor state must identify this exact
            # logical WAV and preserve original recording/conversation lineage.
            checkpoint_key, rid = completed_records.get(str(path), (None, {}))
            completed = (rid.get('source') == str(path)
                and checkpoint_key == manifest.get('processor_key')
                and rid.get('sha256') == digest
                and rid.get('metadata', {}).get('original_source') == frozen
                and rid.get('recording_id') == frozen.get('recording_id')
                and rid.get('conversation_id') == frozen.get('conversation_id'))
            if not (completed and digest and frozen.get('sha256') and frozen.get('recording_id') and frozen.get('conversation_id')):
                held.append({'id': identity, 'path': str(path), 'reason': 'frozen_lineage_or_publication_unqualified'})
                continue
            # Catalog registration verifies bytes once. Refresh never rehashes
            # retired logical paths or requires their old placement to exist.
            objects[identity] = {'id': identity, 'path': str(path), 'sha256': digest,
                'source_sha256': frozen['sha256'], 'classification': 'historical_derivative',
                'ownership_known': True, 'active': False,
                'custody_evidence': processor['custody_evidence']}
    # Reviewed extra classes are never inferred from directory names or mtimes.
    for identity, item in approved.items():
        if item.get('classification') != 'canonical_source':
            owned(item['path'], config)
            if not item.get('custody_evidence') or type(item.get('active')) is not bool:
                raise ValueError('object_custody_evidence_required')
            objects[identity] = {**item, 'ownership_known': True}
    result = {'version': 1, 'source_generation': config['source_generation'],
        'ownership_inventory': 'transcribe-owned-v1', 'objects': list(objects.values()), 'held': held,
        'adoption_sha256': config_digest(adoption_path)}
    write_receipt(config['initial_inventory'], result)
    return result


def install_product(source, destination, expected_commit):
    """Install exactly tracked committed product files, with no checkout dirt."""
    source, destination = plain(source), plain(destination)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected_commit:
        raise ValueError('integrated_source_commit_mismatch')
    archive = subprocess.check_output(['git', '-C', str(source), 'archive', '--format=tar', actual])
    release = destination / actual
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.stat().st_uid != os.getuid() or destination.stat().st_mode & 0o077:
        raise ValueError('private_release_root_required')
    release.mkdir(mode=0o700, exist_ok=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as entries:
        for member in entries:
            name = Path(member.name)
            if member.isdir():
                continue
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError('unsupported_tracked_release_entry')
            if any(part in ('.venv', 'node_modules', '.codex', 'tmp') for part in name.parts):
                continue
            content = entries.extractfile(member).read()
            target = release / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError('immutable_product_generation_conflict')
            else:
                target.write_bytes(content)
                target.chmod(0o500 if member.mode & 0o111 else 0o400)
            files[str(target)] = hashlib.sha256(content).hexdigest()
    receipt = {'version': 1, 'source_commit': actual, 'release': str(release), 'files': files}
    write_receipt(destination / 'product-installation.json', receipt)
    return receipt


def account(config_path):
    config = private_json(config_path)
    runner = Maintenance(config, config_path=config_path)
    states_path = runner.state / 'item-states.json'
    states = private_json(states_path) if states_path.exists() else {}
    latest_path = runner.state / 'latest.json'
    if latest_path.exists():
        for item in private_json(latest_path).get('items', []):
            states[item['object_id']] = item
    inventory = private_json(config['initial_inventory'])
    items = {item['id']: {'object_id': item['id'], 'state': 'held', 'reason': item['reason']}
             for item in inventory.get('held', [])}
    for item in inventory['objects']:
        identity = item['id']
        migration = runner.catalog.migration(identity)
        state = states.get(identity, {})
        items[identity] = {'object_id': identity, **state}
        if migration:
            items[identity]['journal_phase'] = migration['phase']
            if migration['phase'] == 'retired':
                items[identity].update(state='retired', reason='')
            elif migration['phase'] == 'retirement-pending' and migration.get('reason') == 'copy_only':
                items[identity].update(state='copy-only', reason=state.get('reason', 'copy_only'))
        elif not state:
            items[identity].update(state='held', reason='not_yet_migrated')
    # Each retirement is counted once from its original durable pass receipt.
    recovered = {}
    for path in runner.state.glob('*.json'):
        receipt = private_json(path)
        if not receipt.get('receipt_id'):
            continue
        for item in receipt.get('items', []):
            recovered[item['object_id']] = max(recovered.get(item['object_id'], 0), item.get('exclusive_blocks_recovered_bytes', 0))
    result = {'version': 1, 'items': list(items.values()),
              'exclusive_blocks_recovered_bytes': sum(recovered.values()),
              'filesystem_free_bytes': os.statvfs(config['store_root']).f_bavail * os.statvfs(config['store_root']).f_frsize}
    write_receipt(runner.state / 'initial-accounting.json', result)
    return result


def qualify_api(config_path, installation_receipt, pid, health_url):
    """Bind qualification to actual running immutable API code and live health.

    Other consumers require their own compatibility evidence in the reviewed
    adoption inventory; this function grants no blanket worker qualification.
    """
    config = private_json(config_path)
    installation = private_json(installation_receipt)
    release = plain(installation['release'])
    script = str(release / 'transcript_api.py')
    before = Path('/proc') / str(int(pid)) / 'stat'
    process_identity = before.read_text().split(') ', 1)[1].split()[19]
    command = (Path('/proc') / str(int(pid)) / 'cmdline').read_bytes().split(b'\0')
    if script.encode() not in command:
        raise ValueError('api_process_not_installed_generation')
    if any(config_digest(path) != digest for path, digest in installation['files'].items()):
        raise ValueError('installed_product_digest_mismatch')
    with urlopen(health_url, timeout=10) as response:
        if response.status != 200:
            raise ValueError('api_health_failed')
        response.read(65536)
    if before.read_text().split(') ', 1)[1].split()[19] != process_identity:
        raise ValueError('api_process_changed_during_qualification')
    receipt = {'version': 1, 'status': 'qualified',
        'source_generation': config['source_generation'], 'config_sha256': config_digest(config_path),
        'scope_roots': config['source_roots'], 'ownership_inventory': 'transcribe-owned-v1',
        'files': installation['files'], 'api_pid': int(pid), 'api_process_start_ticks': process_identity,
        'api_health_url': health_url, 'qualified_consumer': script,
        'other_consumers': 'per-object reviewed adoption required'}
    write_receipt(config['consumer_receipt'], receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--adoption', type=Path)
    parser.add_argument('command', choices=('refresh', 'pass', 'maintenance', 'account'))
    args = parser.parse_args(argv)
    if args.command in ('refresh', 'pass', 'maintenance'):
        if not args.adoption:
            parser.error('--adoption is required for current ownership refresh')
        result = refresh(args.config, args.adoption)
        if args.command != 'refresh':
            config = private_json(args.config)
            result = Maintenance(config, config_path=args.config).run(
                'maintenance' if args.command == 'maintenance' else 'migrate')
    else:
        result = account(args.config)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
