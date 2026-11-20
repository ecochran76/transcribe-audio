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
