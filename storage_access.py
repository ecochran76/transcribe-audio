"""Placement-aware access with explicit store scope and immutable legacy identity.

Callers doing I/O hold ``media_access(root)`` for its full duration. Resolution
never discovers a user's store implicitly or treats a failed catalog read as a
legacy-path fallback.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

from storage_catalog import StorageCatalog, StorageError


def catalog_for(root):
    if root is None:
        return None
    root = Path(root).expanduser().absolute()
    database = root / 'transcripts.sqlite3'
    if database.is_symlink():
        raise StorageError('private_catalog_required')
    if not database.exists():
        return None
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='storage_objects'").fetchone():
            return None
    return StorageCatalog(root)


def media_record(root, reference):
    catalog = catalog_for(root)
    if catalog is None:
        return None
    try:
        return catalog.object(reference)
    except LookupError:
        # Published paths are locations too; they are not legacy aliases.
        absolute = str(Path(reference).expanduser().absolute())
        with sqlite3.connect(f'file:{catalog.filename}?mode=ro', uri=True) as con:
            matched = con.execute('SELECT id FROM storage_objects WHERE active_path=?', (absolute,)).fetchone()
        return catalog.object(matched[0]) if matched else None


def resolve_media(root, reference):
    """Resolve a catalog object/alias, or retain strict legacy path checks."""
    record = media_record(root, reference)
    if record:
        return catalog_for(root).resolve(record['id'])
    path = Path(reference).expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise StorageError('symlink_or_uncontained_path')
    return path.resolve(strict=True)


def logical_media_path(root, reference):
    record = media_record(root, reference)
    if record:
        return Path(record['logical_path'])
    return Path(reference).expanduser().absolute()


# Public derivative operations accept storage_root as an access scope, keeping it
# outside historical manifests, recipe hashes and generation keys.
from contextvars import ContextVar
from functools import wraps

_ACCESS_ROOT = ContextVar('recording_storage_root', default=None)


def access_root():
    return _ACCESS_ROOT.get()


def storage_scoped(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        from recording_source_guard import media_access
        root = kwargs.pop('storage_root', kwargs.get('store_root', access_root()))
        token = _ACCESS_ROOT.set(root)
        try:
            with media_access(root):
                return function(*args, **kwargs)
        finally:
            _ACCESS_ROOT.reset(token)
    return scoped


def erasure_catalog_paths(root, digest):
    """Exact journal-owned placements, without bypassing source-owner limits.

    Tombstones deliberately do not hide an erasure owner's own inventory. The
    owner still verifies every file and the independent bounded erasure protocol.
    An unfinished copier is never interpreted as quiescent custody.
    """
    import json
    import storage_catalog
    catalog = catalog_for(root)
    if catalog is None:
        return set(), set()
    paths, retired_aliases = set(), set()
    for record in catalog.objects():
        if record['sha256'] != digest and catalog.origin(record['id']) != digest:
            continue
        if catalog.is_pinned(record['id']):
            raise ValueError('speech_owner_storage_reader_active')
        migration = catalog.migration(record['id'])
        if not migration:
            continue
        if migration['phase'] in ('inventoried', 'copying'):
            raise ValueError('speech_owner_storage_copy_active')
        guard = json.loads(migration['guard'])
        destination = Path(migration['destination'])
        if not guard or not guard.get('identity'):
            raise ValueError('speech_owner_storage_identity_unavailable')
        observed = storage_catalog.native_storage_identity(destination.parent, guard.get('uuid', ''))
        if observed['identity'] != guard['identity']:
            raise ValueError('speech_owner_storage_identity_mismatch')
        for field in ('destination', 'temporary'):
            path = Path(migration[field])
            if path.parent != destination.parent or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError('speech_owner_storage_path_unavailable')
            if path.exists():
                paths.add(str(path))
        old = Path(migration['source'])
        if not old.exists() and not old.is_symlink() and migration['phase'] == 'retired':
            retired_aliases.add(str(old))
    return paths, retired_aliases
