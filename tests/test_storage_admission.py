"""New producer writes respect the installed storage policy; resumes stay readable."""
import pytest

from storage_catalog import StorageCatalog, StorageError


def test_new_blob_write_requires_installed_policy_for_catalog_store(tmp_path):
    from transcript_store import prepare_blob
    root = tmp_path / 'store'
    StorageCatalog(root)
    source = tmp_path / 'original.wav'
    source.write_bytes(b'original recording')
    with pytest.raises(StorageError, match='storage_configuration_required'):
        prepare_blob(root, str(source))
    assert not list((root / 'blobs').rglob('*.wav'))
    assert source.read_bytes() == b'original recording'
    import json
    policy = root / 'storage-maintenance.json'
    policy.write_text(json.dumps({'store_root': str(tmp_path / 'different-store')}))
    policy.chmod(0o600)
    with pytest.raises(StorageError, match='storage_store_scope_mismatch'):
        prepare_blob(root, str(source))
    assert not list((root / 'blobs').rglob('*.wav'))


def test_new_staging_requires_policy_without_contacting_inference(tmp_path):
    from managed_recording import RecordingProcessor
    from tests.test_managed_recording import audio, config, Client
    root = tmp_path / 'store'
    StorageCatalog(root)
    client = Client()
    processor = RecordingProcessor(config(tmp_path), client=client, storage_root=root)
    source = audio(tmp_path / 'source.wav')
    with pytest.raises(StorageError, match='storage_configuration_required'):
        processor.prepare(source, recording_id='recording', conversation_id='conversation')
    assert list(processor.spool.iterdir()) == []
    assert client.requests == []


def test_decode_admission_precedes_creating_expanded_audio(tmp_path):
    from managed_recording_cli import run
    from tests.test_managed_recording import audio, config
    root = tmp_path / 'store'
    StorageCatalog(root)
    token = tmp_path / 'token'
    token.write_text('unused-local-test-token')
    token.chmod(0o600)
    settings = dict(processor=config(tmp_path), token_file=str(token),
                    endpoint='http://127.0.0.1:1', allow_insecure_loopback=True,
                    output_directory=str(tmp_path / 'exports'))
    with pytest.raises(StorageError, match='storage_configuration_required'):
        run(audio(tmp_path / 'source.wav'), settings, recording_id='recording',
            conversation_id='conversation', storage_root=root)
    assert not list((tmp_path / 'state' / 'cli' / 'media').glob('*.wav'))


from tests import test_managed_recording_cli as cli_support
recording_service = cli_support.recording_service


def test_configured_cpu_decode_and_staging_publish_without_changing_generation(recording_service, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    from managed_recording_cli import run
    source, settings, workspace = recording_service
    root = workspace / 'store'
    StorageCatalog(root)
    config = json.loads(settings.read_text())
    policy = {'store_root': str(root), 'source_roots': [str(workspace)],
              'transient_roots': [str(workspace / 'state' / 'cli' / 'media'), str(workspace / 'spool')],
              'state_root': str(workspace / 'admission')}
    path = root / 'storage-maintenance.json'
    path.write_text(json.dumps(policy))
    path.chmod(0o600)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 200 * 1024 ** 3})())
    result = run(source, config, recording_id='recording', conversation_id='conversation', storage_root=root)
    assert Path(result['output_paths']['json']).is_file()
    # A completed resume needs no new reservation and must retain its generation.
    path.unlink()
    assert run(source, config, recording_id='recording', conversation_id='conversation', storage_root=root) == result


def test_browser_cache_obeys_policy_and_reuses_existing_audio(tmp_path, monkeypatch):
    import json
    import shutil
    from transcript_api import get_browser_playback_blob
    from transcript_store import TranscriptStoreError
    from tests.test_managed_recording import audio
    from transcript_store import ingest_artifact, prepare_blob
    root = tmp_path / 'store'
    original = audio(tmp_path / 'source.wav')
    artifact = tmp_path / 'transcript.json'
    artifact.write_text(json.dumps({'source_media_path': str(original), 'transcript_text': 'clip',
                                    'backend': 'synthetic', 'utterances': []}))
    ingest_artifact(artifact, root=root, embedding_provider='debug-hash', embedding_model='debug-hash')
    blob = prepare_blob(root, str(original))
    StorageCatalog(root)
    before = original.read_bytes()
    with pytest.raises(TranscriptStoreError, match='storage_configuration_required'):
        get_browser_playback_blob(blob['id'], root=root, end_ms=1000)
    assert not list((root / 'playback-cache').glob('*.mp3'))
    policy = root / 'storage-maintenance.json'
    policy.write_text(json.dumps({'store_root': str(root), 'source_roots': [str(root)],
        'transient_roots': [str(root / 'playback-cache')], 'state_root': str(root / 'admission')}))
    policy.chmod(0o600)
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 200 * 1024 ** 3})())
    result = get_browser_playback_blob(blob['id'], root=root, end_ms=1000)
    assert result['mime_type'] == 'audio/mpeg'
    assert result['bytes'] > 0
    policy.unlink()
    assert get_browser_playback_blob(blob['id'], root=root, end_ms=1000) == result
    assert original.read_bytes() == before
