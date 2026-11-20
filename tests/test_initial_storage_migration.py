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


@pytest.mark.parametrize("registered", [True, False])
def test_canonical_without_replica_held_before_any_payload_write(tmp_path, registered):
    config, path = environment(tmp_path)
    blob = Path(config['store_root']) / 'source.mp4'
    blob.write_bytes(b'original')
    catalog = StorageCatalog(Path(config['store_root']), cold_root=Path(config['cold_root']),
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'})
    digest = config_digest(blob)
    if registered:
        catalog.register('source', blob, classification='canonical_source')
    else:
        with sqlite3.connect(config['live_database']) as db:
            db.execute('INSERT INTO blobs VALUES(?,?,?,?)', ('source', str(blob), digest, 'source_recording'))
    write_receipt(config['initial_inventory'], {'version': 1, 'source_generation': 'test',
        'ownership_inventory': 'transcribe-owned-v1', 'objects': [{'id': 'source', 'path': str(blob),
        'sha256': digest, 'classification': 'canonical_source', 'ownership_known': True, 'active': False}]})
    write_receipt(config['consumer_receipt'], {'version': 1, 'status': 'qualified', 'source_generation': 'test',
        'config_sha256': config_digest(path), 'scope_roots': config['source_roots'],
        'ownership_inventory': 'transcribe-owned-v1', 'files': {str(path): config_digest(path)}})
    runner = Maintenance(config, config_path=path, catalog=catalog,
        clock=lambda: TimeObservation(100), free_bytes=lambda _: 200 * 1024**3)
    blob.chmod(0)  # A policy-held source must not need any payload read.
    assert account(path)['items'][0]['reason'] == 'independent_replica_custody_unqualified'
    result = runner.run('migrate')
    assert result['read_bytes'] == 0
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


@pytest.mark.parametrize('module_name', ['managed_recording_cli', 'managed_recording_transcribe'])
@pytest.mark.parametrize('selection', ['environment', 'explicit', 'unset'])
def test_managed_entrypoints_forward_storage_selector_outside_frozen_config(tmp_path, monkeypatch, module_name, selection):
    import importlib
    module = importlib.import_module(module_name)
    config = tmp_path / 'processing.json'
    write_receipt(config, {'frozen_processing': 'unchanged'})
    before = config.read_bytes()
    captured = {}
    def run(media, processing, **options):
        captured.update(options)
        assert processing == {'frozen_processing': 'unchanged'}
        return {'output_paths': {'json': 'unused.json'}}
    monkeypatch.setattr(module, 'run', run)
    monkeypatch.delenv('TRANSCRIBE_STORAGE_ROOT', raising=False)
    argv = [str(tmp_path / 'media.wav'), '--config', str(config)]
    expected = None
    if selection != 'unset':
        monkeypatch.setenv('TRANSCRIBE_STORAGE_ROOT', str(tmp_path / 'env-store'))
        expected = tmp_path / 'env-store'
    if selection == 'explicit':
        expected = tmp_path / 'explicit-store'
        argv += ['--storage-root', str(expected)]
    assert module.main(argv) == 0
    assert captured['storage_root'] == expected
    assert config.read_bytes() == before


def test_watcher_adapter_requires_policy_before_decode_or_provider(tmp_path):
    from managed_recording_transcribe import run
    from storage_catalog import StorageError
    from tests.test_managed_recording import audio, config
    root = tmp_path / 'store'
    StorageCatalog(root)
    token = tmp_path / 'token'
    token.write_text('unused-local-test-token')
    token.chmod(0o600)
    settings = dict(processor=config(tmp_path), token_file=str(token),
                    endpoint='http://127.0.0.1:1', allow_insecure_loopback=True,
                    output_directory=str(tmp_path / 'exports'))
    source = audio(tmp_path / 'source.wav')
    original = source.read_bytes()
    with pytest.raises(StorageError, match='storage_configuration_required'):
        run(source, settings, recording_id='recording',
            conversation_id='conversation', storage_root=root)
    assert not list((tmp_path / 'state' / 'cli' / 'media').glob('*.wav'))
    assert source.read_bytes() == original


def test_new_completed_registration_uses_reviewed_producer_custody(tmp_path):
    config, path = environment(tmp_path)
    source = Path(config['store_root']) / 'new.mp4'
    source.write_bytes(b'original recording')
    digest = config_digest(source)
    payload = {'recording_id': 'recording', 'conversation_id': 'conversation',
               'source_media_path': str(source), 'backend': 'assemblyai'}
    with sqlite3.connect(config['live_database']) as db:
        db.execute('ALTER TABLE blobs ADD COLUMN original_path TEXT')
        db.execute('CREATE TABLE documents(id TEXT,kind TEXT,json_payload TEXT)')
        db.execute('CREATE TABLE document_blobs(document_id TEXT,blob_id TEXT)')
        db.execute('INSERT INTO blobs VALUES(?,?,?,?,?)', ('new', str(source), digest, 'source_recording', str(source)))
        db.execute('INSERT INTO documents VALUES(?,?,?)', ('doc', 'transcript', json.dumps(payload)))
        db.execute('INSERT INTO document_blobs VALUES(?,?)', ('doc', 'new'))
    write_receipt(config['consumer_receipt'], {'version': 1, 'status': 'qualified', 'source_generation': 'test',
        'config_sha256': config_digest(path), 'scope_roots': config['source_roots'],
        'ownership_inventory': 'transcribe-owned-v1', 'files': {str(path): config_digest(path)}})
    adoption = Path(config['state_root']) / 'adoption.json'
    write_receipt(adoption, {'version': 1, 'scope_roots': config['source_roots'],
        'completed_transcript_registration': {'root': config['store_root'], 'source_generation': 'test',
            'custody_evidence': 'reviewed installed producer and completed artifact publication'}})
    result = refresh(path, adoption)
    assert [item['id'] for item in result['objects']] == ['new']
    assert result['objects'][0]['classification'] == 'canonical_source'
    with sqlite3.connect(config['live_database']) as db:
        db.execute('DELETE FROM document_blobs')
    assert refresh(path, adoption)['held'][0]['reason'] == 'registered_blob_custody_unqualified'
    assert source.read_bytes() == b'original recording'


def test_account_preserves_held_placement_when_logical_id_also_retired(tmp_path):
    config, path = environment(tmp_path)
    store = Path(config['store_root'])
    qualified, unresolved = store / 'qualified.wav', store / 'unresolved.wav'
    qualified.write_bytes(b'exact completed derivative')
    unresolved.write_bytes(b'unqualified distinct placement')
    catalog = StorageCatalog(store, cold_root=Path(config['cold_root']),
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('shared', qualified, classification='historical_derivative')
    catalog.migrate('shared', consumers_qualified=True)
    write_receipt(config['initial_inventory'], {'objects': [
        {'id': 'shared', 'path': str(qualified), 'classification': 'historical_derivative'}],
        'held': [{'id': 'shared', 'path': str(unresolved), 'reason': 'lineage_unqualified'}]})
    result = account(path)
    assert {item['logical_path']: (item['state'], item['reason']) for item in result['items']} == {
        str(qualified): ('retired', ''), str(unresolved): ('held', 'lineage_unqualified')}
    assert unresolved.read_bytes() == b'unqualified distinct placement'


def test_api_qualification_rejects_health_from_another_store(tmp_path):
    from scripts.initial_storage_migration import qualify_api
    import sys
    config, path = environment(tmp_path)
    release = tmp_path / 'release'
    release.mkdir()
    script = release / 'transcript_api.py'
    script.write_text("""from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok","db_path":"/other/database","store_dir":"/other"}')
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
""")
    installation = tmp_path / 'installation.json'
    write_receipt(installation, {'release': str(release), 'files': {str(script): config_digest(script)}})
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    try:
        port = int(process.stdout.readline())
        with pytest.raises(ValueError, match='api_health_store_mismatch'):
            qualify_api(path, installation, process.pid, f'http://127.0.0.1:{port}/api/health')
        assert not Path(config['consumer_receipt']).exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
