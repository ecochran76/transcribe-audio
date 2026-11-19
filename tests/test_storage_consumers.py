"""Public consumer compatibility across catalog placement changes."""
from pathlib import Path

import pytest

from storage_catalog import StorageCatalog
from transcript_store import connect, init_db, prepare_blob
from transcript_api import get_blob


def test_ingest_reuses_published_blob_without_rehydrating_nvme(tmp_path, monkeypatch):
    root = tmp_path / 'store'
    source = tmp_path / 'original.mp3'
    source.write_bytes(b'original recording bytes')
    root.mkdir()
    with connect(root) as con:
        init_db(con)
    blob = prepare_blob(root, str(source))
    catalog = StorageCatalog(root)
    catalog.register(blob['id'], Path(blob['stored_path']), expected_sha256=blob['sha256'])
    cold = tmp_path / 'cold'
    cold.mkdir(mode=0o700)
    catalog = StorageCatalog(root, cold_root=cold,
                             validate_cold=lambda p: {'identity': 'disk', 'failure_domain': 'disk'})
    catalog.migrate(blob['id'], consumers_qualified=True)
    assert not Path(blob['stored_path']).exists()
    # The production resolver still qualifies the saved device at the external boundary.
    monkeypatch.setattr('storage_catalog.native_storage_identity', lambda *a: {'identity': 'disk', 'failure_domain': 'disk'})
    repeated = prepare_blob(root, str(source))
    assert repeated['id'] == blob['id']
    assert repeated['stored_path'] == str(catalog.resolve(blob['id']))
    assert not Path(blob['stored_path']).exists()


def test_managed_resume_keeps_generation_after_source_and_unit_move(tmp_path, monkeypatch):
    from tests.test_managed_recording import audio, config, Client
    from managed_recording import RecordingProcessor, RecordingProcessingError
    root = tmp_path / 'store'
    source = audio(tmp_path / 'source.wav')
    client = Client()
    processor = RecordingProcessor(config(tmp_path), client=client, storage_root=root)
    key = processor.prepare(source, recording_id='recording', conversation_id='conversation')
    frozen = processor.snapshot(key)
    catalog = StorageCatalog(root, source_roots=[tmp_path])
    cold = tmp_path / 'cold'
    cold.mkdir(mode=0o700)
    catalog = StorageCatalog(root, source_roots=[tmp_path], cold_root=cold,
                             validate_cold=lambda p: {'identity': 'disk', 'failure_domain': 'disk'})
    catalog.register('source', source)
    catalog.migrate('source', consumers_qualified=True)
    for unit in frozen['units']:
        path = processor.spool / unit['artifact_id']
        if path.exists():
            catalog.register(unit['artifact_id'], path, expected_sha256=unit['sha256'])
            catalog.migrate(unit['artifact_id'], consumers_qualified=True)
    monkeypatch.setattr('storage_catalog.native_storage_identity', lambda *a: {'identity': 'disk', 'failure_domain': 'disk'})
    assert processor.prepare(source, recording_id='recording', conversation_id='conversation') == key
    assert processor.snapshot(key) == frozen
    with pytest.raises(RecordingProcessingError):
        processor.advance(key)
    assert client.requests[0]['id'] == frozen['units'][0]['stages']['asr']['request']['id']


def test_playback_and_corpus_resolve_cold_blob_with_mount_fail_closed(tmp_path, monkeypatch):
    from tests.test_speech_source_erasure import fixture
    from acoustic_evaluation_corpus import harden_candidate_sources, HARDEN_TOKEN
    root, original, blob, source = fixture(tmp_path)
    cold = tmp_path / 'cold'
    cold.mkdir(mode=0o700)
    catalog = StorageCatalog(root, cold_root=cold,
                             validate_cold=lambda p: {'identity': 'disk', 'failure_domain': 'disk'})
    catalog.register(blob['id'], Path(blob['stored_path']))
    catalog.migrate(blob['id'], consumers_qualified=True)
    monkeypatch.setattr('storage_catalog.native_storage_identity', lambda *a: {'identity': 'disk', 'failure_domain': 'disk'})
    assert get_blob(blob['id'], root=root)['path'] == catalog.resolve(blob['id'])
    receipt = harden_candidate_sources([{'source_blob': blob}], store_root=root, approval_token=HARDEN_TOKEN)
    assert receipt['paths'] == [str(catalog.resolve(blob['id']))]
    from storage_catalog import StorageError
    def missing(*args):
        raise StorageError('cold_mount_unavailable')
    monkeypatch.setattr('storage_catalog.native_storage_identity', missing)
    with pytest.raises(StorageError, match='cold_mount_unavailable'):
        get_blob(blob['id'], root=root)
    with connect(root) as con:
        assert con.execute('SELECT text_content FROM documents').fetchone()[0] == 'Fixture\nFixture'


def test_derivative_replay_preserves_evidence_after_media_move(tmp_path, monkeypatch):
    from tests.test_acoustic_audio_derivatives import write_wav
    from acoustic_audio_derivatives import dry_run, apply_derivative, replay_derivative, resolve_active_derivative, APPLY_TOKEN
    source = write_wav(tmp_path / 'source.wav')
    runtime = tmp_path / 'runtime'
    plan = dry_run(source, runtime_root=runtime)
    apply_derivative(source, runtime_root=runtime, approval_token=APPLY_TOKEN)
    before = replay_derivative(plan['run_id'], runtime_root=runtime)
    reference = resolve_active_derivative(plan['run_id'], runtime_root=runtime)
    cold = tmp_path / 'cold'
    cold.mkdir(mode=0o700)
    root = tmp_path / 'store'
    catalog = StorageCatalog(root, source_roots=[tmp_path], cold_root=cold,
                             validate_cold=lambda p: {'identity': 'disk', 'failure_domain': 'disk'})
    for identifier, path in [('source', source), ('artifact', Path(reference['artifact_path']))]:
        catalog.register(identifier, path)
        catalog.migrate(identifier, consumers_qualified=True)
    monkeypatch.setattr('storage_catalog.native_storage_identity', lambda *a: {'identity': 'disk', 'failure_domain': 'disk'})
    assert replay_derivative(plan['run_id'], runtime_root=runtime, storage_root=root) == before
    after = resolve_active_derivative(plan['run_id'], runtime_root=runtime, storage_root=root)
    assert after['artifact_sha256'] == reference['artifact_sha256']
    assert Path(after['artifact_path']).is_file()


def test_erasure_retains_owner_binding_and_purges_exact_catalog_cold_copy(tmp_path, monkeypatch):
    from tests.test_speech_source_erasure import fixture, OWNER, PRINCIPAL
    from speech_source_erasure import RecordingSourceOwner
    root, original, blob, source = fixture(tmp_path)
    cold = tmp_path / 'cold'
    cold.mkdir(mode=0o700)
    catalog = StorageCatalog(root, cold_root=cold,
                             validate_cold=lambda p: {'identity': 'disk', 'failure_domain': 'disk'})
    catalog.register(blob['id'], Path(blob['stored_path']))
    catalog.migrate(blob['id'], consumers_qualified=True)
    destination = catalog.resolve(blob['id'])
    monkeypatch.setattr('storage_catalog.native_storage_identity', lambda *a: {'identity': 'disk', 'failure_domain': 'disk'})
    service = RecordingSourceOwner(store_root=root, directory=tmp_path/'owner', owner=OWNER,
                                   token='independent', principals=[PRINCIPAL], original_roots=[original.parent])
    try:
        inventory = service.request('inventory', {'principal': PRINCIPAL, 'source': source}, 'Bearer independent')['inventory']
        assert inventory['available'] is True
        intent = {'principal': PRINCIPAL, 'requestId': 'erase-cold', 'source': source,
                  'expectedFingerprint': inventory['fingerprint']}
        service.request('fence', intent, 'Bearer independent')
        service.request('purge', intent, 'Bearer independent')
        assert not destination.exists()
        assert not original.exists()
        with pytest.raises(Exception, match='erased'):
            get_blob(blob['id'], root=root)
    finally:
        service.close()


def test_erasure_cannot_fence_an_active_catalog_reader(tmp_path):
    from tests.test_speech_source_erasure import fixture, OWNER, PRINCIPAL
    from speech_source_erasure import RecordingSourceOwner
    root, original, blob, source = fixture(tmp_path)
    catalog = StorageCatalog(root)
    catalog.register(blob['id'], Path(blob['stored_path']))
    owner = RecordingSourceOwner(store_root=root, directory=tmp_path/'owner', owner=OWNER,
                                 token='independent', principals=[PRINCIPAL], original_roots=[original.parent])
    try:
        with catalog.pin(blob['id']):
            inventory = owner.request('inventory', {'principal': PRINCIPAL, 'source': source}, 'Bearer independent')['inventory']
            assert inventory['available'] is False
            assert inventory['reason'] == 'speech_owner_storage_reader_active'
    finally:
        owner.close()
