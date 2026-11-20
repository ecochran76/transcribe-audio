"""Storage acceptance through public APIs and actual files."""
import hashlib

from storage_catalog import StorageCatalog


def test_migrated_recording_keeps_identity_and_access_after_old_copy_retires(tmp_path):
    source = tmp_path / 'store' / 'blobs' / 'source.bin'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original recording bytes')
    replica = tmp_path / 'backup' / 'source.bin'
    replica.parent.mkdir()
    replica.write_bytes(b'original recording bytes')
    cold = tmp_path / 'cold'
    cold.mkdir()
    catalog = StorageCatalog(source.parents[1], cold_root=cold,
                             validate_cold=lambda p: {'identity': 'test-cold', 'failure_domain': 'disk-b'})
    catalog.register('recording-1', source, classification='canonical_source')
    outcome = catalog.migrate('recording-1', independent_replica=replica,
                              replica_failure_domain='disk-c', replica_custody='operator backup', consumers_qualified=True)
    assert outcome['state'] == 'retired'
    assert not source.exists()
    assert catalog.resolve('recording-1').read_bytes() == b'original recording bytes'
    assert catalog.object('recording-1')['sha256'] == hashlib.sha256(b'original recording bytes').hexdigest()


def test_copy_only_can_finish_after_restart_and_pins_preserve_source(tmp_path):
    source = tmp_path / 'store' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'PCM evidence')
    cold = tmp_path / 'cold'
    cold.mkdir()
    guard = lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'}
    catalog = StorageCatalog(source.parent, cold_root=cold, validate_cold=guard)
    catalog.register('clip', source, classification='processing_derivative')
    with catalog.pin('clip') as readable:
        assert readable.read_bytes() == b'PCM evidence'
        assert catalog.migrate('clip', consumers_qualified=True)['reason'] == 'active_pin'
        assert source.exists()
    copied = catalog.migrate('clip', retire=False)
    assert copied['state'] == 'held'
    assert source.exists()
    restarted = StorageCatalog(source.parent, cold_root=cold, validate_cold=guard)
    assert restarted.migrate('clip', consumers_qualified=True)['state'] == 'retired'
    assert restarted.resolve(source).read_bytes() == b'PCM evidence'


def test_conflicting_destination_never_discards_original(tmp_path):
    import pytest
    from storage_catalog import StorageError
    source = tmp_path / 'store' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'PCM evidence')
    cold = tmp_path / 'cold'
    cold.mkdir()
    catalog = StorageCatalog(source.parent, cold_root=cold,
        validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('clip', source, classification='processing_derivative')
    destination = catalog.migrate('clip', retire=False)['path']
    from pathlib import Path
    Path(destination).write_bytes(b'bad evidence')
    with pytest.raises(StorageError, match='destination_hash_conflict'):
        catalog.migrate('clip', consumers_qualified=True)
    assert source.read_bytes() == b'PCM evidence'


def test_original_requires_independent_replica_and_uncertified_consumers_hold(tmp_path):
    source = tmp_path / 'store' / 'original.bin'
    source.parent.mkdir()
    source.write_bytes(b'source')
    cold = tmp_path / 'cold'
    cold.mkdir()
    catalog = StorageCatalog(source.parent, cold_root=cold,
        validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('source', source, classification='canonical_source')
    assert catalog.migrate('source')['reason'] == 'consumer_compatibility_unqualified'
    assert catalog.migrate('source', consumers_qualified=True)['reason'] == 'independent_replica_required'
    assert source.exists()


def test_changed_published_media_is_not_served(tmp_path):
    import pytest
    from storage_catalog import StorageError
    source = tmp_path / 'store' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'PCM evidence')
    cold = tmp_path / 'cold'
    cold.mkdir()
    catalog = StorageCatalog(source.parent, cold_root=cold,
        validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('clip', source, classification='processing_derivative')
    result = catalog.migrate('clip', consumers_qualified=True)
    from pathlib import Path
    Path(result['path']).write_bytes(b'bad evidence')
    with pytest.raises(StorageError, match='placement_changed'):
        catalog.resolve('clip')


def test_missing_cold_mount_holds_media_without_breaking_catalog_metadata(tmp_path):
    import pytest
    from storage_catalog import StorageError
    source = tmp_path / 'store' / 'clip.wav'
    source.parent.mkdir()
    source.write_bytes(b'PCM evidence')
    cold = tmp_path / 'cold'
    cold.mkdir()
    catalog = StorageCatalog(source.parent, cold_root=cold,
        validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('clip', source, classification='processing_derivative')
    catalog.migrate('clip', retire=False)

    def unavailable(path):
        raise StorageError('cold_mount_unavailable')

    offline = StorageCatalog(source.parent, cold_root=cold, validate_cold=unavailable)
    assert offline.object('clip')['id'] == 'clip'
    with pytest.raises(StorageError, match='cold_mount_unavailable'):
        offline.resolve('clip')
    assert source.exists()


def test_unclassified_unknown_and_durable_objects_hold_retirement(tmp_path):
    for index, classification in enumerate((None, 'unknown', 'unclassified', 'live_metadata', 'runtime', 'active_transient', 'restore_point')):
        root = tmp_path / str(index)
        root.mkdir()
        source = root / 'media.bin'
        source.write_bytes(b'evidence')
        catalog = StorageCatalog(root, cold_root=tmp_path / ('cold' + str(index)),
            validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
        kwargs = {} if classification is None else {'classification': classification}
        catalog.register('media', source, **kwargs)
        result = catalog.migrate('media', consumers_qualified=True)
        assert result['state'] == 'held'
        assert source.read_bytes() == b'evidence'
        if classification is None:
            assert catalog.object('media')['classification'] == 'unclassified'


def test_eligible_disposal_needs_no_cold_copy_and_preserves_identity(tmp_path):
    import pytest
    from storage_catalog import StorageError
    for kind in ('private_cache', 'completed_scratch'):
        source = tmp_path / (kind + '.bin')
        source.write_bytes(b'disposable')
        catalog = StorageCatalog(tmp_path)
        original = catalog.register(kind, source, classification=kind)
        assert catalog.evict(kind)['state'] == 'held'
        assert catalog.evict(kind, consumers_qualified=True)['state'] == 'held'
        with catalog.pin(kind):
            assert catalog.evict(kind, consumers_qualified=True, eligibility_verified=True)['reason'] == 'active_pin'
        assert catalog.evict(kind, consumers_qualified=True, eligibility_verified=True)['state'] == 'evicted'
        assert not source.exists()
        assert catalog.migration(kind)['phase'] == 'evicted'
        assert catalog.object(source)['sha256'] == original['sha256']
        assert catalog.object(kind)['active_path'] == ''
        with pytest.raises(StorageError, match='media_placement_unavailable'):
            catalog.resolve(source)
        assert catalog.evict(kind, consumers_qualified=True, eligibility_verified=True)['idempotent']
    retained = tmp_path / 'retained.bin'
    retained.write_bytes(b'retained')
    catalog.register('retained', retained)
    assert catalog.evict('retained', consumers_qualified=True, eligibility_verified=True)['state'] == 'held'
    assert retained.exists()


def test_canonical_retirement_persists_verified_replica_custody(tmp_path):
    import json
    from storage_catalog import file_identity
    root = tmp_path / 'store'
    root.mkdir()
    source = root / 'source.bin'
    source.write_bytes(b'canonical')
    replica = tmp_path / 'replica.bin'
    replica.write_bytes(b'canonical')
    catalog = StorageCatalog(root, cold_root=tmp_path / 'cold',
        validate_cold=lambda p: {'identity': 'cold', 'failure_domain': 'disk-b'})
    catalog.register('source', source, classification='canonical_source')
    missing = catalog.migrate('source', independent_replica=replica,
        replica_failure_domain='disk-c', consumers_qualified=True)
    assert missing['state'] == 'held'
    result = catalog.migrate('source', independent_replica=replica,
        replica_failure_domain='disk-c', replica_custody='operator backup', consumers_qualified=True)
    assert result['state'] == 'retired'
    proof = json.loads(catalog.migration('source')['replica'])
    assert proof == {'sha256': hashlib.sha256(b'canonical').hexdigest(), 'size': 9,
        'path': str(replica), 'file_identity': file_identity(replica),
        'failure_domain': 'disk-c', 'custody': 'operator backup', 'verification_state': 'verified'}
