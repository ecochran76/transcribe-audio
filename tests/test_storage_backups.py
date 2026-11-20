from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from storage_backups import BackupManager, TrustedTime


def test_snapshot_and_isolated_restore_preserve_live_and_apply_current_erasure(tmp_path):
    live = tmp_path / 'live.sqlite'
    with sqlite3.connect(live) as db:
        db.execute('CREATE TABLE media (id TEXT PRIMARY KEY)')
        db.execute("INSERT INTO media VALUES ('erased-after-snapshot')")
    root = tmp_path / 'backups'
    root.mkdir()
    manager = BackupManager(live, root, lambda path: path == root,
                            lambda: TrustedTime(datetime(2026, 11, 19, tzinfo=timezone.utc)))
    point = manager.snapshot()
    assert manager.verify(point['snapshot_id'])['database_size'] > 0
    with sqlite3.connect(live) as db:
        db.execute('DELETE FROM media')
    def reconcile(db):
        db.execute("DELETE FROM media WHERE id = 'erased-after-snapshot'")
        db.execute('CREATE TABLE current_erasure_tombstones (id TEXT)')
        db.execute("INSERT INTO current_erasure_tombstones VALUES ('erased-after-snapshot')")
        return {'current_tombstones_applied': True, 'source': 'current-erasure-authority'}
    restored = manager.restore(point['snapshot_id'], tmp_path / 'isolated', reconcile)
    with sqlite3.connect(restored) as db:
        assert db.execute('SELECT * FROM media').fetchall() == []
        assert db.execute('SELECT * FROM current_erasure_tombstones').fetchall() == [('erased-after-snapshot',)]
    with sqlite3.connect(live) as db:
        assert db.execute('SELECT * FROM media').fetchall() == []
    assert sorted(path.name for path in (root / point['snapshot_id']).iterdir()) == ['database.sqlite.gz', 'manifest.json']
    assert restored.stat().st_mode & 0o777 == 0o600


def test_rotation_keeps_hour_day_week_points_and_holds_unknown_time(tmp_path):
    from datetime import timedelta
    live = tmp_path / 'live.sqlite'
    with sqlite3.connect(live) as db:
        db.execute('CREATE TABLE example (id INTEGER)')
    root = tmp_path / 'backups'
    root.mkdir()
    now = [TrustedTime(datetime(2026, 10, 1, tzinfo=timezone.utc))]
    manager = BackupManager(live, root, lambda path: True, lambda: now[0])
    points = []
    # 35 daily restore points plus the last day's 24 hourly points.
    for day in range(35):
        now[0] = TrustedTime(datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=day))
        points.append(manager.snapshot()['snapshot_id'])
    for hour in range(1, 24):
        now[0] = TrustedTime(datetime(2026, 11, 4, hour, tzinfo=timezone.utc))
        points.append(manager.snapshot()['snapshot_id'])
    now[0] = None
    unknown = manager.snapshot()['snapshot_id']
    def read_only_checkpoint(*, write_bytes=0, read_bytes=0):
        assert write_bytes == 0, 'rotation must not unpack published database history'
    manager = BackupManager(live, root, lambda path: True, lambda: now[0],
                            checkpoint=read_only_checkpoint)
    assert manager.rotate()['deleted'] == []
    now[0] = TrustedTime(datetime(2026, 11, 5, tzinfo=timezone.utc))
    result = manager.rotate()
    assert len(result['tiers']['hourly']) == 24
    assert len(result['tiers']['daily']) == 7
    assert len(result['tiers']['weekly']) == 4
    assert unknown in result['held']
    assert points[0] in result['deleted']
    manager = BackupManager(live, root, lambda path: True, lambda: now[0])
    assert all(manager.verify(point) for point in result['retained'])


def test_monthly_drill_reports_bounded_recovery_and_failures_privately(tmp_path):
    import hashlib
    live = tmp_path / 'live.sqlite'
    with sqlite3.connect(live) as db:
        db.execute('CREATE TABLE example (id INTEGER)')
    root = tmp_path / 'backups'
    root.mkdir()
    now = [TrustedTime(datetime(2026, 11, 19, tzinfo=timezone.utc))]
    manager = BackupManager(live, root, lambda path: True, lambda: now[0])
    point = manager.snapshot()['snapshot_id']
    def reconcile(db):
        return {'current_tombstones_applied': True, 'source': 'current-authority'}
    receipt = manager.monthly_restore_drill(
        point, tmp_path / 'drill', reconcile_erasure=reconcile,
        check_references=lambda db: [], recover_media=lambda identity, max_bytes: b'original',
        samples=[{'media_id': 'recording-1', 'sha256': hashlib.sha256(b'original').hexdigest()}])
    assert receipt['status'] == 'passed'
    assert receipt['coverage'] == {'sampled': 1, 'verified': 1, 'full_corpus': False}
    assert Path(receipt['receipt_path']).stat().st_mode & 0o777 == 0o600
    assert manager.monthly_restore_drill(point, tmp_path / 'again', reconcile_erasure=reconcile,
        check_references=lambda db: [], recover_media=lambda identity, max_bytes: b'', samples=[])['status'] == 'not_due'
    now[0] = TrustedTime(datetime(2026, 12, 1, tzinfo=timezone.utc))
    failed = manager.monthly_restore_drill(point, tmp_path / 'failed', reconcile_erasure=reconcile,
        check_references=lambda db: ['missing placement'], recover_media=lambda identity, max_bytes: b'wrong',
        samples=[{'media_id': 'recording-1', 'sha256': hashlib.sha256(b'original').hexdigest()}])
    assert failed['status'] == 'failed'
    assert len(failed['failures']) == 2
    now[0] = None
    assert manager.monthly_restore_drill(point, tmp_path / 'unknown', reconcile_erasure=reconcile,
        check_references=lambda db: [], recover_media=lambda identity, max_bytes: b'', samples=[])['status'] == 'held'


def test_failure_boundaries_preserve_live_and_last_good_points(tmp_path):
    import pytest
    from storage_backups import BackupError
    live = tmp_path / 'live.sqlite'
    connection = sqlite3.connect(live)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE example (value TEXT)')
    connection.execute("INSERT INTO example VALUES ('committed in WAL')")
    connection.commit()
    root = tmp_path / 'backups'
    root.mkdir()
    allowed = [True]
    now = [TrustedTime(datetime(2026, 11, 19, tzinfo=timezone.utc))]
    manager = BackupManager(live, root, lambda path: allowed[0], lambda: now[0])
    point = manager.snapshot()['snapshot_id']
    assert {item.name for item in (root / point).iterdir()} == {'manifest.json', 'database.sqlite.gz'}
    restored = manager.restore(point, tmp_path / 'wal-restore',
        lambda db: {'current_tombstones_applied': True, 'source': 'empty-current-authority'})
    with sqlite3.connect(restored) as db:
        assert db.execute('SELECT value FROM example').fetchall() == [('committed in WAL',)]
    for destination in (live, tmp_path):
        with pytest.raises(BackupError, match='isolated'):
            manager.restore(point, destination, None)
    with pytest.raises(BackupError, match='reconciliation'):
        manager.restore(point, tmp_path / 'no-erasure', None)
    with pytest.raises(BackupError, match='unqualified'):
        manager.restore(point, tmp_path / 'bad-erasure', lambda db: {})
    assert not (tmp_path / 'bad-erasure').exists()
    allowed[0] = False
    with pytest.raises(BackupError, match='identity'):
        manager.snapshot()
    with pytest.raises(BackupError, match='identity'):
        manager.rotate()
    allowed[0] = True
    good = manager.snapshot()['snapshot_id']
    point, good = sorted((point, good))
    compressed = root / point / 'database.sqlite.gz'
    compressed.write_bytes(b'corrupt')
    with pytest.raises(BackupError, match='checksum'):
        manager.restore(point, tmp_path / 'corrupt-restore', lambda db: {})
    result = manager.rotate()
    assert result['deleted'] == []
    assert result['reason'] == 'corruption pauses expiry'
    assert set(result['retained']) == {point, good}
    assert manager.verify(good)['snapshot_id'] == good
    assert connection.execute('SELECT value FROM example').fetchall() == [('committed in WAL',)]
    connection.close()


def test_missing_mount_and_unknown_or_uncertain_time_do_not_expire(tmp_path):
    import pytest
    from storage_backups import BackupError
    live = tmp_path / 'live.sqlite'
    with sqlite3.connect(live) as db:
        db.execute('CREATE TABLE example (id INTEGER)')
    missing = tmp_path / 'absent-hdd' / 'backups'
    manager = BackupManager(live, missing, lambda path: True, lambda: None)
    with pytest.raises(BackupError):
        manager.snapshot()
    assert not missing.exists()
    root = tmp_path / 'backups'
    root.mkdir()
    now = [TrustedTime(datetime(2026, 11, 19, tzinfo=timezone.utc), 10)]
    manager = BackupManager(live, root, lambda path: True, lambda: now[0])
    uncertain = manager.snapshot()['snapshot_id']
    now[0] = TrustedTime(datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert uncertain in manager.rotate()['held']
    evidence = root / 'frozen-evidence'
    evidence.mkdir()
    (evidence / 'lineage').write_bytes(b'retain')
    assert manager.rotate()['deleted'] == []
    assert (evidence / 'lineage').read_bytes() == b'retain'
    with pytest.raises(ValueError, match='UTC'):
        TrustedTime(datetime(2026, 11, 19))
