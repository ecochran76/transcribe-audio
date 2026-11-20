"""Private operational SQLite restore points, separate from evidence snapshots.

The host supplies a destination guard that verifies the *actual* backup root's
configured native mount identity, ownership and capacity on every write. It must
return True or raise; no directory or guest-clock fallback is provided.

Clock adapters return a fresh StableTime observation as TrustedTime, or None.
Its UTC instant is the service measurement instant, never extrapolated here.
Restore requires a host reconciliation callback applying the CURRENT erasure
fences/tombstones to the isolated connection, including invalidating references
and restoring tombstones. A no-op callback is not a valid installed adapter.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Callable
from uuid import uuid4


class BackupError(RuntimeError):
    """A held or failed backup operation; the live database is unchanged."""


@dataclass(frozen=True)
class TrustedTime:
    observed_at: datetime
    uncertainty_seconds: float = 0

    def __post_init__(self):
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() != timedelta(0):
            raise ValueError('TrustedTime requires an authoritative UTC observation')
        if not math.isfinite(self.uncertainty_seconds) or self.uncertainty_seconds < 0:
            raise ValueError('invalid time uncertainty')

    @property
    def earliest(self):
        return self.observed_at - timedelta(seconds=self.uncertainty_seconds)

    @property
    def latest(self):
        return self.observed_at + timedelta(seconds=self.uncertainty_seconds)


def _integrity(db, checkpoint=None):
    if checkpoint:
        def progress():
            checkpoint()
            return 0
        db.set_progress_handler(progress, 10000)
    if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
        raise BackupError('SQLite integrity check failed')
    if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
        raise BackupError('SQLite foreign key check failed')


def _hash(path, checkpoint=None):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if checkpoint:
                checkpoint(read_bytes=len(block))
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class BackupManager:
    def __init__(self, live_database: Path, backup_root: Path,
                 destination_guard: Callable[[Path], bool],
                 trusted_clock: Callable[[], TrustedTime | None], checkpoint=None):
        self.live_database = Path(live_database).absolute()
        self.backup_root = Path(backup_root).absolute()
        self.destination_guard = destination_guard
        self.trusted_clock = trusted_clock
        self.checkpoint = checkpoint or (lambda **kwargs: None)

    def _guard(self):
        self.checkpoint()
        root = self.backup_root
        if any(part.is_symlink() for part in (root, *root.parents)):
            raise BackupError('backup root must not traverse symlinks')
        if not root.is_dir() or self.destination_guard(root) is not True:
            raise BackupError('backup destination identity unavailable or mismatched')
        os.chmod(root, 0o700)

    def _point(self, snapshot_id):
        if not re.fullmatch(r'[0-9a-f]{32}', snapshot_id):
            raise BackupError('invalid snapshot identity')
        point = self.backup_root / snapshot_id
        if point.is_symlink() or not point.is_dir():
            raise BackupError('snapshot unavailable')
        for name in ('manifest.json', 'database.sqlite.gz'):
            if (point / name).is_symlink() or not (point / name).is_file():
                raise BackupError('invalid snapshot file')
        return point

    def snapshot(self):
        """Use SQLite backup(), verify, compress and publish metadata only."""
        self._guard()
        observation = self.trusted_clock()
        snapshot_id = uuid4().hex
        temp = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=self.backup_root))
        try:
            raw = temp / 'database.sqlite'
            # mode=ro prevents accidentally creating a missing live database.
            with closing(sqlite3.connect(self.live_database.as_uri() + '?mode=ro', uri=True)) as source:
                with closing(sqlite3.connect(raw)) as target:
                    os.chmod(raw, 0o600)
                    page_size = source.execute('PRAGMA page_size').fetchone()[0]
                    page_count = source.execute('PRAGMA page_count').fetchone()[0]
                    # Debit each next bounded SQLite step BEFORE it writes. A
                    # concurrent writer may restart backup(); every repeated step
                    # is still debited and cannot evade the pass ceiling.
                    self.checkpoint(write_bytes=min(128, page_count) * page_size)
                    def progress(status, remaining, total):
                        if remaining:
                            self.checkpoint(write_bytes=min(128, remaining) * page_size)
                        else:
                            self.checkpoint()
                    source.backup(target, pages=128, progress=progress)
                    _integrity(target, self.checkpoint)
            compressed = temp / 'database.sqlite.gz'
            with raw.open('rb') as source, compressed.open('xb') as target:
                os.chmod(compressed, 0o600)
                with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as zipper:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        # Deflate's worst-case output plus headers is conservatively
                        # debited before writing; retries never regain this budget.
                        self.checkpoint(write_bytes=len(block) + 1024, read_bytes=len(block))
                        zipper.write(block)
                target.flush()
                os.fsync(target.fileno())
            manifest = {
                'version': 1, 'kind': 'operational_sqlite', 'snapshot_id': snapshot_id,
                'created_at': observation.observed_at.isoformat() if observation else None,
                'uncertainty_seconds': observation.uncertainty_seconds if observation else None,
                'database_size': raw.stat().st_size, 'database_sha256': _hash(raw, self.checkpoint),
                'compressed_size': compressed.stat().st_size,
                'compressed_sha256': _hash(compressed, self.checkpoint),
                'media_included': False, 'restore_requires_current_erasure': True,
            }
            # Read back compressed bytes before publishing a restore point.
            self._unpack(compressed, temp / 'verified.sqlite', manifest)
            raw.unlink()
            (temp / 'verified.sqlite').unlink()
            _write_json(temp / 'manifest.json', manifest)
            self._guard()
            _sync_dir(temp)
            temp.rename(self.backup_root / snapshot_id)
            _sync_dir(self.backup_root)
            return manifest
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def _unpack(self, compressed, destination, manifest):
        self.checkpoint()
        if shutil.disk_usage(destination.parent).free < manifest['database_size'] + 1024 * 1024:
            raise BackupError('restore destination capacity unavailable')
        if compressed.stat().st_size != manifest['compressed_size'] or _hash(compressed, self.checkpoint) != manifest['compressed_sha256']:
            raise BackupError('compressed checksum mismatch')
        try:
            with gzip.open(compressed, 'rb') as source, destination.open('xb') as target:
                os.chmod(destination, 0o600)
                remaining = manifest['database_size']
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise BackupError('truncated database')
                    self.checkpoint(write_bytes=len(block), read_bytes=len(block))
                    target.write(block)
                    remaining -= len(block)
                if source.read(1):
                    raise BackupError('database exceeds declared size')
                target.flush()
                os.fsync(target.fileno())
        except (OSError, EOFError) as error:
            raise BackupError('compressed snapshot corrupt') from error
        if _hash(destination, self.checkpoint) != manifest['database_sha256']:
            raise BackupError('database checksum mismatch')
        with closing(sqlite3.connect(destination)) as db:
            _integrity(db, self.checkpoint)

    def verify(self, snapshot_id):
        """Verify compressed checksum, exact SQLite bytes and database integrity."""
        self._guard()
        point = self._point(snapshot_id)
        manifest = json.loads((point / 'manifest.json').read_text())
        if (manifest.get('version') != 1 or manifest.get('kind') != 'operational_sqlite'
                or manifest.get('snapshot_id') != snapshot_id
                or not isinstance(manifest.get('database_size'), int)
                or manifest['database_size'] <= 0):
            raise BackupError('invalid operational snapshot manifest')
        with tempfile.TemporaryDirectory(prefix='.verify-', dir=self.backup_root) as temp:
            self._unpack(point / 'database.sqlite.gz', Path(temp) / 'database.sqlite', manifest)
        return manifest

    def restore(self, snapshot_id, destination, reconcile_erasure):
        """Restore only into a new private directory; never overwrite a store.

        reconcile_erasure(connection) must apply current erasure authority and
        return {'current_tombstones_applied': True, 'source': <authority locator>}.
        Failure leaves no published restored database. No media is restored.
        """
        manifest = self.verify(snapshot_id)
        destination = Path(destination).absolute()
        if (destination.exists() or destination.is_symlink()
                or self.live_database == destination
                or destination in self.live_database.parents
                or any(p.is_symlink() for p in destination.parents)):
            raise BackupError('restore requires a new isolated destination')
        if not callable(reconcile_erasure):
            raise BackupError('current source-erasure reconciliation required')
        destination.mkdir(mode=0o700)
        try:
            restored = destination / 'database.sqlite'
            self._unpack(self._point(snapshot_id) / 'database.sqlite.gz', restored, manifest)
            with closing(sqlite3.connect(restored)) as db:
                outcome = reconcile_erasure(db)
                if (not isinstance(outcome, dict)
                        or outcome.get('current_tombstones_applied') is not True
                        or not outcome.get('source')):
                    raise BackupError('current source-erasure reconciliation unqualified')
                db.commit()
                _integrity(db, self.checkpoint)
            _write_json(destination / 'restore.json', {'snapshot_id': snapshot_id, 'erasure': outcome})
            _sync_dir(destination)
            return restored
        except BaseException:
            shutil.rmtree(destination)
            raise

    def rotate(self):
        """Keep newest points in 24 UTC hours, seven days and four ISO weeks.

        Tier sets overlap. Frozen evidence/unrecognized directories are never
        candidates. Any corrupt restore point pauses all expiry; unknown or
        boundary-ambiguous time holds that individual point. Expiry is based on
        service observations only, never mtime or elapsed guest time.
        """
        self._guard()
        now = self.trusted_clock()
        result = {'retained': [], 'deleted': [], 'held': {},
                  'tiers': {'hourly': [], 'daily': [], 'weekly': []}}
        candidates = []
        points = [point for point in sorted(self.backup_root.iterdir())
                  if re.fullmatch(r'[0-9a-f]{32}', point.name)]
        result['retained'] = [point.name for point in points]
        for point in points:
            try:
                manifest = self.verify(point.name)
            except (BackupError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                result['held'][point.name] = 'corrupt restore point: ' + str(error)
                result['reason'] = 'corruption pauses expiry'
                return result
            try:
                created = TrustedTime(datetime.fromisoformat(manifest['created_at']),
                                      manifest['uncertainty_seconds'])
            except (ValueError, TypeError, KeyError):
                result['held'][point.name] = 'unknown creation time'
                continue
            if now is None:
                result['held'][point.name] = 'trusted rotation time unavailable'
                continue
            if created.latest > now.earliest:
                result['held'][point.name] = 'creation time not safely in past'
                continue
            def keys(instant):
                return (instant.strftime('%Y-%m-%dT%H'), instant.date().isoformat(),
                        instant.isocalendar()[:2])
            if keys(created.earliest) != keys(created.latest):
                result['held'][point.name] = 'creation uncertainty crosses retention boundary'
                continue
            candidates.append((created.observed_at, point.name, keys(created.observed_at)))
        if now is None:
            result['reason'] = 'trusted rotation time unavailable'
            return result
        retained = set(result['held'])
        candidates.sort(reverse=True)
        for index, (tier, limit) in enumerate((('hourly', 24), ('daily', 7), ('weekly', 4))):
            seen = set()
            for _, identity, buckets in candidates:
                if buckets[index] not in seen and len(seen) < limit:
                    seen.add(buckets[index])
                    result['tiers'][tier].append(identity)
                    retained.add(identity)
        # Verification of every candidate precedes the first deletion.
        for created, identity, _ in candidates:
            if identity in retained:
                continue
            if created >= now.earliest - timedelta(hours=1):
                retained.add(identity)
                result['held'][identity] = 'not safely superseded for one hour'
                continue
            self._guard()
            point = self._point(identity)
            # Only our two operational files may be retired, never media or an
            # evidence bundle somebody placed alongside the restore point.
            if {path.name for path in point.iterdir()} != {'manifest.json', 'database.sqlite.gz'}:
                retained.add(identity)
                result['held'][identity] = 'unexpected files in operational restore point'
                continue
            (point / 'manifest.json').unlink()
            (point / 'database.sqlite.gz').unlink()
            point.rmdir()
            result['deleted'].append(identity)
        _sync_dir(self.backup_root)
        result['retained'] = sorted(retained)
        return result

    def monthly_restore_drill(self, snapshot_id, destination, *, reconcile_erasure,
                              check_references, recover_media, samples,
                              max_samples=16, max_media_bytes=64 * 1024 * 1024):
        """Run once per authoritative UTC calendar month, recording coverage.

        check_references(connection) returns failure descriptions. samples is a
        bounded list of {media_id, sha256}; recover_media(id, remaining_bytes)
        must enforce that byte limit and CURRENT source-erasure fences before
        returning exact recovered bytes. The drill itself never copies media.
        Adapters are host-owned and must not replay inference or provider calls.
        Failed drills remain retryable; a successful drill is not a corpus scrub.
        """
        self._guard()
        now = self.trusted_clock()
        if now is None:
            return {'status': 'held', 'reason': 'trusted monthly time unavailable'}
        month = now.earliest.strftime('%Y-%m')
        if month != now.latest.strftime('%Y-%m'):
            return {'status': 'held', 'reason': 'time uncertainty crosses month boundary'}
        drills = self.backup_root / 'drills'
        if drills.is_symlink():
            raise BackupError('invalid drill receipt directory')
        drills.mkdir(mode=0o700, exist_ok=True)
        os.chmod(drills, 0o700)
        for path in drills.glob('*.json'):
            if path.is_symlink():
                raise BackupError('invalid drill receipt')
            previous = json.loads(path.read_text())
            if previous.get('month') == month and previous.get('status') == 'passed':
                return {'status': 'not_due', 'month': month, 'receipt_path': str(path)}
        if (not isinstance(samples, (list, tuple)) or not 1 <= len(samples) <= max_samples
                or not 1 <= max_samples <= 16 or not 1 <= max_media_bytes <= 64 * 1024 * 1024):
            return {'status': 'held', 'reason': 'bounded representative media samples required'}
        receipt = {'version': 1, 'snapshot_id': snapshot_id, 'month': month,
                   'observed_at': now.observed_at.isoformat(),
                   'uncertainty_seconds': now.uncertainty_seconds,
                   'status': 'failed', 'failures': [],
                   'coverage': {'sampled': len(samples), 'verified': 0, 'full_corpus': False}}
        try:
            restored = self.restore(snapshot_id, destination, reconcile_erasure)
            with closing(sqlite3.connect(restored.as_uri() + '?mode=ro', uri=True)) as db:
                _integrity(db, self.checkpoint)
                receipt['failures'].extend(check_references(db))
            remaining = max_media_bytes
            for sample in samples:
                if (not isinstance(sample.get('media_id'), str)
                        or not re.fullmatch(r'[0-9a-f]{64}', sample.get('sha256', ''))):
                    raise BackupError('invalid media recovery sample')
                recovered = recover_media(sample['media_id'], remaining)
                if not isinstance(recovered, bytes) or len(recovered) > remaining:
                    raise BackupError('media recovery exceeds byte bound')
                remaining -= len(recovered)
                if hashlib.sha256(recovered).hexdigest() != sample['sha256']:
                    receipt['failures'].append('media checksum mismatch: ' + sample['media_id'])
                else:
                    receipt['coverage']['verified'] += 1
            if not receipt['failures']:
                receipt['status'] = 'passed'
        except Exception as error:
            # Do not put arbitrary callback exception text/private data in receipts.
            receipt['failures'].append('restore drill failed: ' + type(error).__name__)
        self._guard()
        receipt_path = drills / (uuid4().hex + '.json')
        receipt['receipt_path'] = str(receipt_path)
        _write_json(receipt_path, receipt)
        _sync_dir(drills)
        return receipt
