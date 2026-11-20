import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from storage_catalog import StorageCatalog
from storage_policy import TimeObservation
from storage_maintenance import Maintenance, PassBudget, MaintenanceHeld, private_json


def test_pass_budget_debits_writes_and_stops_on_fresh_external_deadline():
    observations = iter([TimeObservation(100), TimeObservation(101), TimeObservation(1901)])
    budget = PassBudget(lambda: next(observations), write_limit=8)
    budget.checkpoint(write_bytes=5)
    assert budget.written == 5
    with pytest.raises(MaintenanceHeld, match='pass_time_budget'):
        budget.checkpoint(write_bytes=1)
    assert budget.written == 5


def test_copy_stops_at_chunk_boundary_preserving_original(tmp_path):
    source = tmp_path / 'store' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'x' * (5 * 1024 * 1024))
    cold = tmp_path / 'cold'
    cold.mkdir()
    budget = PassBudget(lambda: TimeObservation(100), write_limit=4 * 1024 * 1024)
    catalog = StorageCatalog(source.parent, cold_root=cold,
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'},
        checkpoint=budget.checkpoint)
    catalog.register('clip', source, classification='historical_derivative')
    with pytest.raises(MaintenanceHeld, match='pass_write_budget'):
        catalog.migrate('clip', consumers_qualified=True)
    assert source.exists()
    assert budget.written == 4 * 1024 * 1024
    assert not list(cold.rglob('*.partial'))


def setup_operator(tmp_path):
    store, cold, backups, state = [tmp_path / name for name in ('store', 'cold', 'backups', 'state')]
    for path in (store, cold, backups, state):
        path.mkdir(mode=0o700)
    source = store / 'one.wav'
    source.write_bytes(b'exact bytes')
    config = {'version': 1, 'store_root': str(store), 'cold_root': str(cold),
              'cold_uuid': 'synthetic', 'source_roots': [str(store)],
              'backup_root': str(backups), 'live_database': str(store / 'transcripts.sqlite3'),
              'state_root': str(state), 'source_generation': 'synthetic'}
    catalog = StorageCatalog(store, cold_root=cold,
        validate_cold=lambda _: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('one', source, classification='historical_derivative')
    return config, catalog, source


def test_maintenance_holds_unqualified_consumers_and_mount_outage(tmp_path):
    config, catalog, source = setup_operator(tmp_path)
    runner = Maintenance(config, catalog=catalog, clock=lambda: TimeObservation(100),
                         free_bytes=lambda _: 200 * 1024 ** 3)
    outcome = runner.run('migrate')
    assert outcome['items'][0]['reason'] == 'consumer_compatibility_unqualified'
    assert source.exists()


def test_private_configuration_rejects_shared_or_symlinked_json(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{}')
    path.chmod(0o644)
    with pytest.raises(MaintenanceHeld, match='private'):
        private_json(path)
    path.chmod(0o600)
    assert private_json(path) == {}
    link = tmp_path / 'link'
    link.symlink_to(path)
    with pytest.raises(MaintenanceHeld):
        private_json(link)


def qualify_operator(tmp_path, config):
    config_file = tmp_path / 'operator.json'
    config['consumer_receipt'] = str(tmp_path / 'qualified.json')
    config_file.write_text(json.dumps(config))
    config_file.chmod(0o600)
    consumer = tmp_path / 'consumer.py'
    consumer.write_text('# installed consumer\n')
    receipt = {'version': 1, 'status': 'qualified', 'source_generation': 'synthetic',
               'config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest(),
               'scope_roots': config['source_roots'], 'ownership_inventory': 'transcribe-owned-v1',
               'files': {str(consumer): hashlib.sha256(consumer.read_bytes()).hexdigest()}}
    Path(config['consumer_receipt']).write_text(json.dumps(receipt))
    Path(config['consumer_receipt']).chmod(0o600)
    return config_file


from pathlib import Path


def test_qualified_inventory_migrates_derivative_and_origin_fence_blocks_access(tmp_path):
    from storage_catalog import StorageError
    config, catalog, source = setup_operator(tmp_path)
    origin = hashlib.sha256(b'canonical original').hexdigest()
    inventory = tmp_path / 'inventory.json'
    config['inventories'] = [str(inventory)]
    inventory.write_text(json.dumps({'version': 1, 'source_generation': 'synthetic',
        'ownership_inventory': 'transcribe-owned-v1', 'objects': [
            {'id': 'one', 'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'source_sha256': origin, 'classification': 'historical_derivative',
             'ownership_known': True, 'active': False}]}))
    inventory.chmod(0o600)
    config_file = qualify_operator(tmp_path, config)
    runner = Maintenance(config, config_path=config_file, catalog=catalog,
                         clock=lambda: TimeObservation(100), free_bytes=lambda _: 200 * 1024 ** 3)
    outcome = runner.run('migrate')
    assert outcome['status'] == 'complete'
    assert outcome['items'][0]['state'] == 'retired'
    assert not source.exists()
    assert catalog.resolve('one').read_bytes() == b'exact bytes'
    with sqlite3.connect(catalog.filename) as db:
        db.execute('CREATE TABLE recording_source_erasures(sha256 TEXT PRIMARY KEY,owner TEXT,request TEXT)')
        db.execute('INSERT INTO recording_source_erasures VALUES(?,?,?)', (origin, 'owner', 'request'))
    with pytest.raises(StorageError, match='recording_source_erased'):
        catalog.resolve('one')


def test_restore_applies_actual_current_live_tombstones_and_invalidates_origin(tmp_path):
    config, catalog, source = setup_operator(tmp_path)
    origin = hashlib.sha256(b'canonical').hexdigest()
    catalog.register('one', source, classification='historical_derivative', source_sha256=origin)
    config['erasure_authority'] = config['live_database']
    runner = Maintenance(config, catalog=catalog, clock=lambda: TimeObservation(100))
    with sqlite3.connect(catalog.filename) as live:
        live.execute('CREATE TABLE recording_source_erasures(sha256 TEXT PRIMARY KEY,owner TEXT,request TEXT)')
        live.execute('INSERT INTO recording_source_erasures VALUES(?,?,?)', (origin, 'owner', 'request'))
    restored = sqlite3.connect(':memory:')
    restored.executescript('CREATE TABLE storage_objects(id TEXT,sha256 TEXT,active_path TEXT); CREATE TABLE storage_origins(object_id TEXT,source_sha256 TEXT);')
    restored.execute('INSERT INTO storage_objects VALUES(?,?,?)', ('one', 'other-sha', str(source)))
    restored.execute('INSERT INTO storage_origins VALUES(?,?)', ('one', origin))
    result = runner.reconcile_erasure(restored)
    assert result['fences'] == 1
    assert restored.execute('SELECT active_path FROM storage_objects').fetchone()[0] == ''
    assert restored.execute('SELECT sha256 FROM recording_source_erasures').fetchone()[0] == origin
    restored.close()


def test_backup_budget_aborts_actual_sqlite_copy_without_publishing(tmp_path):
    from storage_backups import BackupManager, TrustedTime
    live = tmp_path / 'live.sqlite'
    with sqlite3.connect(live) as db:
        db.execute('CREATE TABLE payload(value BLOB)')
        db.execute('INSERT INTO payload VALUES(?)', (b'x' * 1048576,))
    backups = tmp_path / 'backups'
    backups.mkdir()
    budget = PassBudget(lambda: TimeObservation(100), write_limit=1024)
    manager = BackupManager(live, backups, lambda _: True,
        lambda: TrustedTime(datetime(2026, 11, 19, tzinfo=timezone.utc)), checkpoint=budget.checkpoint)
    with pytest.raises(MaintenanceHeld, match='pass_write_budget'):
        manager.snapshot()
    assert list(backups.iterdir()) == []
    assert budget.written == 0


def test_installer_prepares_immutable_generation_without_activation(tmp_path):
    from scripts.install_storage_maintenance import install
    config = tmp_path / 'config.json'
    config.write_text('{"version":1}')
    config.chmod(0o600)
    import sys
    result = install(config, tmp_path / 'releases', Path(sys.executable))
    assert result['activated'] is False
    assert result['consumer_qualification'] == 'not_proven'
    assert Path(result['release'], 'scripts/storage_maintenance.py').is_file()
    unit = Path(result['units'], 'transcribe-storage-maintenance.service').read_text()
    assert '--config' in unit and str(config) in unit
    assert 'maintenance' in unit and '@CONFIG@' not in unit


def test_missing_time_holds_mutation_but_inventory_still_works(tmp_path):
    config, catalog, source = setup_operator(tmp_path)
    runner = Maintenance(config, catalog=catalog, clock=lambda: None)
    assert runner.run('migrate')['reason'] == 'trusted_time_unavailable'
    assert runner.run('inventory')['objects'][0]['id'] == 'one'
    assert source.read_bytes() == b'exact bytes'


def test_systemic_mount_failure_pauses_queue_without_object_retries(tmp_path):
    config, catalog, source = setup_operator(tmp_path)
    inventory = tmp_path / 'inventory.json'
    config['inventories'] = [str(inventory)]
    inventory.write_text(json.dumps({'version': 1, 'source_generation': 'synthetic',
        'ownership_inventory': 'transcribe-owned-v1', 'objects': [
            {'id': 'one', 'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'classification': 'private_cache', 'ownership_known': True, 'active': False}]}))
    inventory.chmod(0o600)
    # Re-register as cache in a fresh catalog; role alone cannot relabel an
    # existing immutable object classification.
    second = source.parent / 'two.bin'
    second.write_bytes(b'cache')
    catalog.register('two', second, classification='private_cache')
    data = json.loads(inventory.read_text())
    data['objects'][0].update(id='two', path=str(second), sha256=hashlib.sha256(b'cache').hexdigest())
    inventory.write_text(json.dumps(data))
    config_file = qualify_operator(tmp_path, config)
    from storage_catalog import StorageError
    def missing(_):
        raise StorageError('cold_mount_unavailable')
    catalog.validator = missing
    runner = Maintenance(config, config_path=config_file, catalog=catalog,
                         clock=lambda: TimeObservation(100), free_bytes=lambda _: 200 * 1024 ** 3)
    result = runner.run('migrate')
    assert result['status'] == 'held'
    assert result['reason'] == 'cold_mount_unavailable'
    assert result['written_bytes'] == 0
    assert second.exists() and source.exists()
    states = runner.item_states
    assert not states


def test_single_process_ownership_uses_kernel_flock(tmp_path):
    config, catalog, _ = setup_operator(tmp_path)
    first = Maintenance(config, catalog=catalog, clock=lambda: TimeObservation(100))
    second = Maintenance(config, catalog=catalog, clock=lambda: TimeObservation(100))
    with first.worker():
        with pytest.raises(MaintenanceHeld, match='worker_already_running'):
            second.run('inventory')


def test_stable_time_rejects_unbounded_and_accepts_wide_authenticated_interval():
    from storage_maintenance import StableTimeClock
    value = {'contract_version': '2.0', 'status': 'qualified_live', 'bound': 'finite',
        'epoch': '12345678-1234-5678-1234-567812345678', 'authentication': 'NTS-verified',
        'operators': ['one', 'two'], 'time_scale': 'UTC-step; announced-leaps-unavailable',
        'evidence_generation': 1, 'acquisition_age_ns': 0,
        'measurement': {'counter': '123', 'frequency_hz': '1000'},
        'counter_policy': {'rate_ppm': 5000, 'maximum_age_ns': 300000000000},
        'estimate': {'earliest_utc': '2026-11-19T00:00:00Z', 'utc': '2026-11-19T00:00:01Z',
                     'latest_utc': '2026-11-19T00:00:02Z', 'maximum_error_ns': 1000000000,
                     'maximum_error_ms': 1000}}
    assert StableTimeClock.decode(value).uncertainty_seconds == 1
    value['bound'] = 'unbounded'
    with pytest.raises(MaintenanceHeld, match='invalid_evidence'):
        StableTimeClock.decode(value)


def test_origin_erasure_during_copy_prevents_publication_and_retirement(tmp_path):
    from storage_catalog import StorageError
    config, catalog, source = setup_operator(tmp_path)
    origin = hashlib.sha256(b'canonical').hexdigest()
    catalog.register('one', source, source_sha256=origin)
    fenced = [False]
    def checkpoint(**io):
        if io.get('write_bytes') and not fenced[0]:
            with sqlite3.connect(catalog.filename) as db:
                db.execute('CREATE TABLE recording_source_erasures(sha256 TEXT PRIMARY KEY,owner TEXT,request TEXT)')
                db.execute('INSERT INTO recording_source_erasures VALUES(?,?,?)', (origin, 'owner', 'request'))
            fenced[0] = True
    catalog.checkpoint = checkpoint
    with pytest.raises(StorageError, match='recording_source_erased'):
        catalog.migrate('one', consumers_qualified=True)
    assert source.exists()
    assert catalog.migration('one')['phase'] == 'verified'


def test_shared_admission_counts_real_retained_payload_and_releases_after_cold_move(tmp_path, monkeypatch):
    from storage_maintenance import reserve_transient
    import shutil
    config, catalog, source = setup_operator(tmp_path)
    transient = source.parent / 'staging'
    transient.mkdir(mode=0o700)
    config['transient_roots'] = [str(transient)]
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 200 * 1024 ** 3})())
    payload = transient / 'prepared.wav'
    with reserve_transient(config_path, 'decode', 20 * 1024 ** 3) as lease:
        lease.checkpoint(write_bytes=3)
        payload.write_bytes(b'PCM')
        with pytest.raises(MaintenanceHeld, match='transient_budget'):
            with reserve_transient(config_path, 'other', 1):
                pytest.fail('overlapping reservation admitted')
    # A retained sparse object represents existing debt without a large fixture.
    with payload.open('wb') as stream:
        stream.truncate(20 * 1024 ** 3)
    with pytest.raises(MaintenanceHeld, match='transient_budget'):
        with reserve_transient(config_path, 'decode', 1):
            pytest.fail('retained transient debt ignored')
    payload.unlink()
    with reserve_transient(config_path, 'decode', 4) as lease:
        with pytest.raises(MaintenanceHeld, match='reservation_exceeded'):
            lease.checkpoint(write_bytes=5)
        assert lease.written == 0


def test_shared_admission_preserves_nvme_reserve(tmp_path, monkeypatch):
    from storage_maintenance import reserve_transient
    import shutil
    config, catalog, source = setup_operator(tmp_path)
    config['transient_roots'] = [str(source.parent)]
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 100 * 1024 ** 3})())
    with pytest.raises(MaintenanceHeld, match='nvme_reserve'):
        with reserve_transient(config_path, 'stage', 1):
            pytest.fail('NVMe reserve was spent')
