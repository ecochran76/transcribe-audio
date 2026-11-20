"""Durable recording placement, reader pins and verified per-object migration."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
from typing import Callable
from uuid import uuid4

from recording_source_guard import media_access


class StorageError(RuntimeError):
    """A storage operation cannot safely establish its required evidence."""


def file_identity(path: Path) -> list[int]:
    info = path.lstat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, stat.S_IMODE(info.st_mode), info.st_uid,
            info.st_gid, info.st_nlink]


def _plain_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise StorageError('symlink_or_uncontained_path')
    return path


def file_digest(path: Path, *, expected_links: int = 1, checkpoint=None) -> str:
    path = _plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != expected_links:
            raise StorageError('regular_single_link_file_required')
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            if checkpoint:
                checkpoint(read_bytes=len(chunk))
            digest.update(chunk)
        return digest.hexdigest()


def native_storage_identity(path: Path, expected_uuid: str) -> dict:
    """Validate a configured native mount without following a directory fallback."""
    path = _plain_path(path)
    if not expected_uuid:
        raise StorageError('configured_cold_uuid_required')
    entries = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        parts = line.split()
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), parts[4]))
        if path == mount or path.is_relative_to(mount):
            pos = parts.index('-')
            entries.append((mount, parts[pos + 1], parts[pos + 2]))
    if not entries:
        raise StorageError('cold_mount_unavailable')
    mount, filesystem, source = max(entries, key=lambda row: len(row[0].parts))
    if mount == Path('/') or filesystem != 'ext4':
        raise StorageError('native_cold_mount_required')
    try:
        actual = subprocess.check_output(
            ['blkid', '-s', 'UUID', '-o', 'value', source], text=True,
            stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise StorageError('cold_mount_identity_unavailable') from error
    if actual != expected_uuid:
        raise StorageError('cold_mount_identity_mismatch')
    if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()
                          or path.stat().st_mode & 0o077):
        raise StorageError('owned_cold_directory_required')
    return {'uuid': actual, 'identity': actual, 'failure_domain': actual,
            'mount_point': str(mount), 'source': source, 'filesystem': filesystem}


class StorageCatalog:
    """Own placement changes behind register/resolve/pin/migrate public operations."""

    def __init__(self, store_root: Path, *, cold_root: Path | None = None,
                 expected_cold_uuid: str = '', source_roots: list[Path] | None = None,
                 validate_cold: Callable[[Path], dict] | None = None, checkpoint=None):
        self.root = _plain_path(Path(store_root))
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_uid != os.getuid():
            raise StorageError('owned_store_required')
        self.filename = self.root / 'transcripts.sqlite3'
        if self.filename.is_symlink() or (self.filename.exists() and
                                           self.filename.stat().st_nlink != 1):
            raise StorageError('private_catalog_required')
        self.cold_root = _plain_path(Path(cold_root)) if cold_root else None
        self.expected_cold_uuid = expected_cold_uuid
        self.validator = validate_cold
        self.checkpoint = checkpoint
        self.source_roots = [_plain_path(Path(p)) for p in (source_roots or [self.root])]
        with self._db() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS storage_objects(
                    id TEXT PRIMARY KEY,sha256 TEXT NOT NULL,size INTEGER NOT NULL,
                    classification TEXT NOT NULL,logical_path TEXT NOT NULL,
                    active_path TEXT NOT NULL,guard TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS storage_origins(
                    object_id TEXT PRIMARY KEY,source_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS storage_aliases(
                    path TEXT PRIMARY KEY,object_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS storage_pins(
                    object_id TEXT NOT NULL,token TEXT PRIMARY KEY,owner TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS storage_migrations(
                    id TEXT PRIMARY KEY,object_id TEXT NOT NULL,source TEXT NOT NULL,
                    destination TEXT NOT NULL,temporary TEXT NOT NULL,phase TEXT NOT NULL,
                    source_identity TEXT NOT NULL,guard TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',destination_identity TEXT,
                    replica TEXT,policy_version INTEGER NOT NULL DEFAULT 1);
            ''')
        self.filename.chmod(0o600)

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.filename, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA synchronous=FULL')
        try:
            with con:
                yield con
        finally:
            con.close()

    def _available(self, con, digest):
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='recording_source_erasures'").fetchone()
        if exists and con.execute('SELECT 1 FROM recording_source_erasures WHERE sha256=?', (digest,)).fetchone():
            raise StorageError('recording_source_erased')

    def _object_available(self, con, record):
        self._available(con, record['sha256'])
        origin = con.execute('SELECT source_sha256 FROM storage_origins WHERE object_id=?', (record['id'],)).fetchone()
        if origin:
            self._available(con, origin[0])

    def origin(self, object_id):
        with self._db() as con:
            row = con.execute('SELECT source_sha256 FROM storage_origins WHERE object_id=?', (object_id,)).fetchone()
            return row[0] if row else None

    def _guard(self, path=None, recorded=None):
        path = Path(path or self.cold_root or '')
        result = (self.validator(path) if self.validator else
                  native_storage_identity(path, (recorded or {}).get('uuid', self.expected_cold_uuid)))
        if not isinstance(result, dict) or not result.get('identity') or not result.get('failure_domain'):
            raise StorageError('qualified_storage_identity_required')
        if recorded and result['identity'] != recorded['identity']:
            raise StorageError('cold_mount_identity_mismatch')
        return result

    def _digest(self, path, **kwargs):
        return file_digest(path, checkpoint=self.checkpoint, **kwargs)

    def register(self, object_id: str, source: Path, *, classification='unclassified',
                 expected_sha256: str | None = None, source_sha256: str | None = None) -> dict:
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', object_id):
            raise StorageError('invalid_object_id')
        if source_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', source_sha256):
            raise StorageError('invalid_source_origin')
        source = _plain_path(Path(source))
        if not any(source.is_relative_to(root) for root in self.source_roots):
            raise StorageError('source_outside_owned_roots')
        before = file_identity(source)
        digest = self._digest(source)
        if file_identity(source) != before:
            raise StorageError('source_changed')
        if expected_sha256 and expected_sha256 != digest:
            raise StorageError('source_hash_mismatch')
        with media_access(self.root):
            if file_identity(source) != before:
                raise StorageError('source_changed')
            with self._db() as con:
                self._available(con, digest)
                if source_sha256:
                    self._available(con, source_sha256)
                    origin = con.execute('SELECT source_sha256 FROM storage_origins WHERE object_id=?', (object_id,)).fetchone()
                    if origin and origin[0] != source_sha256:
                        raise StorageError('source_origin_conflict')
                    con.execute('INSERT OR IGNORE INTO storage_origins VALUES(?,?)', (object_id, source_sha256))
                existing = con.execute('SELECT * FROM storage_objects WHERE id=?', (object_id,)).fetchone()
                if existing and existing['sha256'] != digest:
                    raise StorageError('object_identity_conflict')
                alias = con.execute('SELECT object_id FROM storage_aliases WHERE path=?', (str(source),)).fetchone()
                if alias and alias[0] != object_id:
                    raise StorageError('legacy_alias_conflict')
                con.execute('INSERT OR IGNORE INTO storage_objects(id,sha256,size,classification,logical_path,active_path) VALUES(?,?,?,?,?,?)',
                            (object_id, digest, before[2], classification, str(source), str(source)))
                con.execute('INSERT OR IGNORE INTO storage_aliases VALUES(?,?)', (str(source), object_id))
        return self.object(object_id)

    def object(self, reference: str | Path) -> dict:
        with self._db() as con:
            row = con.execute('SELECT * FROM storage_objects WHERE id=? OR id=(SELECT object_id FROM storage_aliases WHERE path=?)',
                              (str(reference), str(reference))).fetchone()
            if row is None:
                raise LookupError(str(reference))
            self._object_available(con, row)
            return dict(row)

    def objects(self) -> list[dict]:
        with self._db() as con:
            return [dict(row) for row in con.execute('SELECT * FROM storage_objects ORDER BY id')]

    def resolve(self, reference: str | Path) -> Path:
        record = self.object(reference)
        if not record['active_path']:
            raise StorageError('media_placement_unavailable')
        path = _plain_path(Path(record['active_path']))
        guard = json.loads(record['guard'])
        if guard:
            self._guard(path.parent, guard)
        if not path.is_file() or path.stat().st_size != record['size']:
            raise StorageError('media_placement_unavailable')
        migration = self.migration(record['id'])
        if guard and migration and migration['destination_identity']:
            expected = json.loads(migration['destination_identity'])
            current = file_identity(path)
            if current[1:] != expected[1:]:
                raise StorageError('placement_changed')
        return path

    @contextmanager
    def pin(self, reference: str | Path, *, owner='reader'):
        record = self.object(reference)
        token = uuid4().hex
        with media_access(self.root):
            with self._db() as con:
                self._object_available(con, record)
                con.execute('INSERT INTO storage_pins VALUES(?,?,?)', (record['id'], token, owner))
        try:
            yield self.resolve(record['id'])
        finally:
            with self._db() as con:
                con.execute('DELETE FROM storage_pins WHERE token=?', (token,))

    def is_pinned(self, object_id: str) -> bool:
        with self._db() as con:
            return bool(con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone())

    def migration(self, object_id: str) -> dict | None:
        with self._db() as con:
            row = con.execute('SELECT * FROM storage_migrations WHERE object_id=? ORDER BY rowid DESC LIMIT 1', (object_id,)).fetchone()
            return dict(row) if row else None

    def _phase(self, operation, phase, *, reason='', destination_identity=None):
        with self._db() as con:
            con.execute('UPDATE storage_migrations SET phase=?,reason=?,destination_identity=COALESCE(?,destination_identity) WHERE id=?',
                        (phase, reason, json.dumps(destination_identity) if destination_identity else None, operation))

    @contextmanager
    def _operation_lock(self, object_id):
        directory = self.root / '.storage-locks'
        _plain_path(directory).mkdir(mode=0o700, exist_ok=True)
        path = directory / (hashlib.sha256(object_id.encode()).hexdigest() + '.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise StorageError('migration_already_owned') from error
            yield
        finally:
            os.close(fd)

    def evict(self, object_id: str, *, consumers_qualified=False,
              eligibility_verified=False) -> dict:
        """Dispose only policy-qualified private cache or completed scratch."""
        with self._operation_lock(object_id), media_access(self.root):
            record = self.object(object_id)
            prior = self.migration(object_id)
            if prior and prior['phase'] == 'evicted':
                return {'state': 'evicted', 'object_id': object_id, 'idempotent': True}
            reason = ''
            if record['classification'] not in {'private_cache', 'completed_scratch'}:
                reason = 'classification_eviction_unqualified'
            elif not consumers_qualified:
                reason = 'consumer_compatibility_unqualified'
            elif not eligibility_verified:
                reason = 'policy_eligibility_unverified'
            if reason:
                return {'state': 'held', 'reason': reason, 'object_id': object_id}
            source = _plain_path(Path(record['active_path']))
            if not any(source.is_relative_to(root) for root in self.source_roots):
                raise StorageError('source_outside_owned_roots')
            if prior and prior['phase'] != 'evicting':
                raise StorageError('migration_configuration_changed')
            identity = json.loads(prior['source_identity']) if prior else file_identity(source)
            if identity[6] != os.getuid():
                raise StorageError('unowned_source')
            with self._db() as con:
                self._object_available(con, record)
                if con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone():
                    return {'state': 'held', 'reason': 'active_pin', 'object_id': object_id}
            if source.exists():
                if file_identity(source) != identity or self._digest(source) != record['sha256'] or file_identity(source) != identity:
                    raise StorageError('source_changed')
            elif not prior:
                raise StorageError('media_placement_unavailable')
            operation = hashlib.sha256(('evict:' + object_id + str(source)).encode()).hexdigest()
            if not prior:
                with self._db() as con:
                    con.execute('INSERT INTO storage_migrations(id,object_id,source,destination,temporary,phase,source_identity,guard) VALUES(?,?,?,?,?,?,?,?)',
                                (operation, object_id, str(source), '', '', 'evicting', json.dumps(identity), '{}'))
            if self.checkpoint:
                self.checkpoint()
            with self._db() as con:
                self._object_available(con, record)
                if con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone():
                    return {'state': 'held', 'reason': 'active_pin', 'object_id': object_id}
                if source.exists():
                    if file_identity(source) != identity or self._digest(source) != record['sha256'] or file_identity(source) != identity:
                        raise StorageError('source_changed')
                    source.unlink()
                directory = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                con.execute("UPDATE storage_objects SET active_path='',guard='{}' WHERE id=?", (object_id,))
                con.execute("UPDATE storage_migrations SET phase='evicted',reason='' WHERE id=?", (operation,))
            return {'state': 'evicted', 'object_id': object_id}

    def migrate(self, object_id: str, *, independent_replica: Path | None = None,
                replica_failure_domain: str = '', replica_custody: str = '', consumers_qualified=False,
                retire=True) -> dict:
        with self._operation_lock(object_id):
            return self._migrate(object_id, independent_replica, replica_failure_domain,
                                 consumers_qualified, retire, replica_custody)

    def _migrate(self, object_id, independent_replica, replica_domain, consumers_qualified, retire, replica_custody):
        record = self.object(object_id)
        prior = self.migration(object_id)
        if prior and prior['phase'] == 'retired':
            self.resolve(object_id)
            return {'state': 'retired', 'object_id': object_id, 'path': record['active_path'], 'idempotent': True}
        if self.cold_root is None:
            raise StorageError('cold_root_required')
        guard = self._guard(self.cold_root)
        self.cold_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        source = _plain_path(Path(prior['source'] if prior else record['active_path']))
        if not any(source.is_relative_to(root) for root in self.source_roots):
            raise StorageError('source_outside_owned_roots')
        suffix = source.suffix if re.fullmatch(r'\.[A-Za-z0-9]{1,12}', source.suffix) else '.bin'
        category = 'quarantine' if record['classification'] == 'unclassified' else 'objects'
        destination = self.cold_root / category / record['sha256'][:2] / (object_id + '-' + record['sha256'] + suffix)
        _plain_path(destination.parent).mkdir(mode=0o700, parents=True, exist_ok=True)
        operation = hashlib.sha256((object_id + str(destination)).encode()).hexdigest()
        temporary = destination.with_name('.' + destination.name + '.' + operation + '.partial')
        with media_access(self.root):
            with self._db() as con:
                self._object_available(con, record)
                if con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone():
                    return {'state': 'held', 'reason': 'active_pin', 'object_id': object_id}
                if not prior:
                    identity = file_identity(source)
                    if self._digest(source) != record['sha256'] or file_identity(source) != identity:
                        raise StorageError('source_changed')
                    con.execute('INSERT INTO storage_migrations(id,object_id,source,destination,temporary,phase,source_identity,guard) VALUES(?,?,?,?,?,?,?,?)',
                                (operation, object_id, str(source), str(destination), str(temporary), 'inventoried', json.dumps(identity), json.dumps(guard)))
                else:
                    if prior['id'] != operation:
                        raise StorageError('migration_configuration_changed')
                    identity = json.loads(prior['source_identity'])
                    if guard['identity'] != json.loads(prior['guard'])['identity']:
                        raise StorageError('cold_mount_identity_mismatch')
        # Recover atomic no-overwrite publication interrupted after link(), before unlink().
        if destination.exists() and temporary.exists():
            left, right = file_identity(destination), file_identity(temporary)
            if left[:2] != right[:2] or left[8] != 2 or left[6] != os.getuid():
                raise StorageError('conflicting_partial_publication')
            if self._digest(destination, expected_links=2) != record['sha256']:
                raise StorageError('destination_hash_conflict')
            if file_identity(destination) != left or file_identity(temporary) != right:
                raise StorageError('destination_changed')
            temporary.unlink()
        if not destination.exists():
            if file_identity(source) != identity:
                raise StorageError('source_changed')
            self._phase(operation, 'copying')
            if temporary.exists():
                _plain_path(temporary)
                if temporary.stat().st_nlink != 1 or temporary.stat().st_uid != os.getuid():
                    raise StorageError('unowned_partial_copy')
                temporary.unlink()
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                with os.fdopen(fd, 'wb') as target:
                    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(source_fd, 'rb') as original:
                        if file_identity(source) != identity:
                            raise StorageError('source_changed')
                        for chunk in iter(lambda: original.read(4 * 1024 * 1024), b''):
                            if self.checkpoint:
                                self.checkpoint(write_bytes=len(chunk), read_bytes=len(chunk))
                            target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                if file_identity(source) != identity or self._digest(temporary) != record['sha256']:
                    raise StorageError('copy_verification_failed')
                self._guard(self.cold_root, guard)
                # The operation lock owns this destination; never overwrite other content.
                os.link(temporary, destination)
                temporary.unlink()
                directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                if temporary.exists() and temporary.stat().st_nlink == 1:
                    temporary.unlink()
                raise
        destination_identity = file_identity(destination)
        if self._digest(destination) != record['sha256'] or file_identity(destination) != destination_identity:
            self._phase(operation, 'quarantined', reason='destination_hash_conflict')
            raise StorageError('destination_hash_conflict')
        if self.checkpoint:
            self.checkpoint()
        self._phase(operation, 'verified', destination_identity=destination_identity)
        with media_access(self.root):
            with self._db() as con:
                self._object_available(con, record)
                if con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone():
                    return {'state': 'held', 'reason': 'active_pin', 'object_id': object_id}
                self._guard(self.cold_root, guard)
                if source.exists() and file_identity(source) != identity:
                    raise StorageError('source_changed')
                con.execute('UPDATE storage_objects SET active_path=?,guard=? WHERE id=?',
                            (str(destination), json.dumps(guard), object_id))
                con.execute('INSERT OR IGNORE INTO storage_aliases VALUES(?,?)', (str(destination), object_id))
                exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='blobs'").fetchone()
                if exists:
                    con.execute('UPDATE blobs SET stored_path=? WHERE id=?', (str(destination), object_id))
                con.execute("UPDATE storage_migrations SET phase='published' WHERE id=?", (operation,))
        reason = ''
        replica_proof = None
        if not retire or not consumers_qualified:
            reason = 'consumer_compatibility_unqualified' if retire else 'copy_only'
        elif record['classification'] not in {'canonical_source', 'source_recording', 'processing_derivative', 'historical_derivative', 'closed_evidence', 'qualification_payload', 'unresolved'}:
            reason = 'classification_retirement_unqualified'
        elif record['classification'] in {'canonical_source', 'source_recording'}:
            if independent_replica is None or not replica_domain or not replica_custody.strip() or replica_domain == guard['failure_domain']:
                reason = 'independent_replica_required'
            else:
                replica = _plain_path(Path(independent_replica))
                if replica in {source, destination} or replica.is_relative_to(self.cold_root):
                    reason = 'independent_replica_required'
                elif guard.get('uuid') and replica.stat().st_dev == destination.stat().st_dev:
                    reason = 'independent_replica_same_filesystem'
                else:
                    proof = file_identity(replica)
                    if (replica.stat().st_dev, replica.stat().st_ino) == tuple(identity[:2]) or self._digest(replica) != record['sha256'] or file_identity(replica) != proof:
                        reason = 'independent_replica_invalid'
                    else:
                        replica_proof = {'sha256': record['sha256'], 'size': record['size'],
                            'path': str(replica), 'file_identity': proof,
                            'failure_domain': replica_domain, 'custody': replica_custody,
                            'verification_state': 'verified'}
                        if prior and prior['replica'] and json.loads(prior['replica']) != replica_proof:
                            raise StorageError('independent_replica_changed')
                        with self._db() as con:
                            con.execute('UPDATE storage_migrations SET replica=? WHERE id=?',
                                        (json.dumps(replica_proof), operation))
        if reason:
            self._phase(operation, 'retirement-pending', reason=reason)
            return {'state': 'held', 'reason': reason, 'object_id': object_id, 'path': str(destination), 'published': True}
        if source.exists() and (file_identity(source) != identity or self._digest(source) != record['sha256']):
            raise StorageError('source_changed')
        if self.checkpoint:
            self.checkpoint()
        with media_access(self.root):
            with self._db() as con:
                self._object_available(con, record)
                if con.execute('SELECT 1 FROM storage_pins WHERE object_id=?', (object_id,)).fetchone():
                    return {'state': 'held', 'reason': 'active_pin', 'object_id': object_id}
                self._guard(self.cold_root, guard)
                if file_identity(destination) != destination_identity:
                    raise StorageError('destination_changed')
                if source.exists():
                    if file_identity(source) != identity:
                        raise StorageError('source_changed')
                    if replica_proof:
                        replica_path = Path(replica_proof['path'])
                        if (file_identity(replica_path) != replica_proof['file_identity']
                                or self._digest(replica_path) != replica_proof['sha256']
                                or file_identity(replica_path) != replica_proof['file_identity']):
                            raise StorageError('independent_replica_changed')
                    source.unlink()
                    directory = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                con.execute("UPDATE storage_migrations SET phase='retired',reason='' WHERE id=?", (operation,))
        return {'state': 'retired', 'object_id': object_id, 'path': str(destination)}
