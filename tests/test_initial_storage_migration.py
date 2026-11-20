import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from scripts.initial_storage_migration import install_product, refresh, account
from storage_catalog import StorageCatalog
from storage_maintenance import Maintenance, config_digest, write_receipt
from storage_policy import TimeObservation


def environment(tmp_path):
    store, cold, state = [tmp_path / name for name in ('store', 'cold', 'state')]
    for path in (store, cold, state):
        path.mkdir(mode=0o700)
    db = store / 'live.sqlite'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE blobs(id TEXT,stored_path TEXT,sha256 TEXT,role TEXT)')
    config = {'version': 1, 'source_roots': [str(store)], 'store_root': str(store),
              'cold_root': str(cold), 'cold_uuid': 'synthetic', 'state_root': str(state),
              'live_database': str(db), 'source_generation': 'test',
              'consumer_receipt': str(state / 'consumer.json'),
              'initial_inventory': str(state / 'inventory.json'),
              'inventories': [str(state / 'inventory.json')]}
    path = state / 'config.json'
    write_receipt(path, config)
    return config, path


def test_refresh_requires_reviewed_custody_and_discovers_new_blobs(tmp_path):
    config, path = environment(tmp_path)
    blob = Path(config['store_root']) / 'clip.mp4'
    blob.write_bytes(b'canonical')
    digest = config_digest(blob)
    with sqlite3.connect(config['live_database']) as con:
        con.execute('INSERT INTO blobs(id,stored_path,sha256) VALUES(?,?,?)', ('one', str(blob), digest))
    adoption = Path(config['state_root']) / 'adoption.json'
    write_receipt(adoption, {'version': 1, 'scope_roots': config['source_roots'], 'objects': []})
    assert refresh(path, adoption)['held'][0]['reason'] == 'registered_blob_custody_unqualified'
    value = json.loads(adoption.read_text())
    value['objects'] = [{'id': 'one', 'path': str(blob), 'sha256': digest,
                         'classification': 'canonical_source', 'active': False,
                         'custody_evidence': 'reviewed immutable registration and inactive producer'}]
    write_receipt(adoption, value)
    assert refresh(path, adoption)['objects'][0]['id'] == 'one'
    with sqlite3.connect(config['live_database']) as con:
        con.execute('INSERT INTO blobs(id,stored_path,sha256) VALUES(?,?,?)', ('new', str(blob), digest))
    assert refresh(path, adoption)['held'][0]['id'] == 'new'


def test_completed_frozen_wav_refresh_survives_retired_logical_path(tmp_path):
    config, path = environment(tmp_path)
    root = Path(config['store_root']) / 'processor'
    media = root / 'cli' / 'media'
    media.mkdir(parents=True)
    wav = media / ('a' * 64 + '.wav')
    wav.write_bytes(b'RIFFexact-wave')
    origin = hashlib.sha256(b'original').hexdigest()
    identity = {'source': '/original', 'sha256': origin, 'recording_id': 'r', 'conversation_id': 'c'}
    write_receipt(root / 'cli' / ('a' * 64 + '.json'), {'frozen': {'identity': identity}, 'decoded_sha256': config_digest(wav), 'processor_key': 'job'})
    write_receipt(root / 'job.json', {'identity': {**identity, 'source': str(wav), 'sha256': config_digest(wav), 'metadata': {'original_source': identity}}, 'publication_state': 'complete'})
    adoption = Path(config['state_root']) / 'adoption.json'
    write_receipt(adoption, {'version': 1, 'scope_roots': config['source_roots'],
                            'processors': [{'path': str(root), 'custody_evidence': 'reviewed processor ownership'}]})
    first = refresh(path, adoption)
    assert first['objects'][0]['source_sha256'] == origin
    wav.unlink()
    assert refresh(path, adoption)['objects'] == first['objects']
    write_receipt(root / 'job.json', {'identity': {**identity, 'source': str(wav)}, 'publication_state': 'publishing'})
    assert refresh(path, adoption)['held'][0]['reason'] == 'frozen_lineage_or_publication_unqualified'


def test_full_tracked_install_ignores_checkout_dirt_and_rejects_changed_release(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    (source / 'storage_admission.py').write_text('committed\n')
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    (source / 'storage_admission.py').write_text('unrelated dirty work\n')
    result = install_product(source, tmp_path / 'releases', commit)
    installed = Path(result['release']) / 'storage_admission.py'
    assert installed.read_text() == 'committed\n'
    installed.chmod(0o600)
    installed.write_text('changed')
    with pytest.raises(ValueError, match='immutable_product_generation_conflict'):
        install_product(source, tmp_path / 'releases', commit)


def test_canonical_without_replica_held_before_any_payload_write(tmp_path):
    config, path = environment(tmp_path)
    blob = Path(config['store_root']) / 'source.mp4'
    blob.write_bytes(b'original')
    catalog = StorageCatalog(Path(config['store_root']), cold_root=Path(config['cold_root']),
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('source', blob, classification='canonical_source')
    write_receipt(config['initial_inventory'], {'version': 1, 'source_generation': 'test',
        'ownership_inventory': 'transcribe-owned-v1', 'objects': [{'id': 'source', 'path': str(blob),
        'sha256': config_digest(blob), 'ownership_known': True, 'active': False}]})
    write_receipt(config['consumer_receipt'], {'version': 1, 'status': 'qualified', 'source_generation': 'test',
        'config_sha256': config_digest(path), 'scope_roots': config['source_roots'],
        'ownership_inventory': 'transcribe-owned-v1', 'files': {str(path): config_digest(path)}})
    runner = Maintenance(config, config_path=path, catalog=catalog,
        clock=lambda: TimeObservation(100), free_bytes=lambda _: 200 * 1024**3)
    result = runner.run('migrate')
    assert result['items'][0]['reason'] == 'independent_replica_custody_unqualified'
    assert result['written_bytes'] == 0
    assert catalog.migration('source') is None
    assert blob.exists()


def test_retirement_and_cumulative_accounting_are_idempotent(tmp_path):
    config, path = environment(tmp_path)
    wav = Path(config['store_root']) / 'finished.wav'
    wav.write_bytes(b'RIFFexact')
    catalog = StorageCatalog(Path(config['store_root']), cold_root=Path(config['cold_root']),
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'})
    origin = hashlib.sha256(b'source').hexdigest()
    catalog.register('wav-finished', wav, classification='historical_derivative', source_sha256=origin)
    write_receipt(config['initial_inventory'], {'version': 1, 'source_generation': 'test',
        'ownership_inventory': 'transcribe-owned-v1', 'objects': [{'id': 'wav-finished',
        'path': str(wav), 'sha256': config_digest(wav), 'source_sha256': origin,
        'ownership_known': True, 'active': False}]})
    write_receipt(config['consumer_receipt'], {'version': 1, 'status': 'qualified', 'source_generation': 'test',
        'config_sha256': config_digest(path), 'scope_roots': config['source_roots'],
        'ownership_inventory': 'transcribe-owned-v1', 'files': {str(path): config_digest(path)}})
    ticks = iter(range(100, 10000))
    runner = Maintenance(config, config_path=path, catalog=catalog,
        clock=lambda: TimeObservation(next(ticks)), free_bytes=lambda _: 200 * 1024**3)
    first = runner.run('migrate')
    assert first['items'][0]['state'] == 'retired'
    assert not wav.exists()
    second = runner.run('migrate')
    assert second['written_bytes'] == 0
    assert second['read_bytes'] == 0
    measured = first['items'][0]['exclusive_blocks_recovered_bytes']
    assert measured > 0
    assert account(path)['exclusive_blocks_recovered_bytes'] == measured
    assert account(path)['exclusive_blocks_recovered_bytes'] == measured
