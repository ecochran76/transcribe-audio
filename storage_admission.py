"""Producer admission tied to an explicit catalog store, outside frozen inputs."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from storage_access import catalog_for
from storage_catalog import StorageError


def storage_policy_path(root):
    if root is None or catalog_for(root) is None:
        return None
    config = Path(root) / 'storage-maintenance.json'
    if not config.exists():
        raise StorageError('storage_configuration_required')
    return config


@contextmanager
def admit_media_write(root, owner, maximum_bytes, *, transient_path=None):
    """Legacy stores remain compatible; a catalog store requires installed policy."""
    config = storage_policy_path(root)
    if config is None:
        yield None
        return
    from storage_maintenance import plain, private_json, reserve_transient
    if transient_path is not None:
        policy = private_json(config)
        if not any(plain(transient_path).is_relative_to(plain(path))
                   for path in policy.get('transient_roots', [])):
            raise StorageError('transient_destination_outside_owned_roots')
    with reserve_transient(config, owner, maximum_bytes) as lease:
        yield lease


def copy_admitted_media(root, source, destination, *, owner, sha256):
    """Copy within a reserved bound and publish only the expected complete bytes."""
    maximum = source.stat().st_size
    with admit_media_write(root, owner, maximum) as lease:
        fd, name = tempfile.mkstemp(prefix='.media-', dir=destination.parent)
        try:
            digest = hashlib.sha256()
            written = 0
            with os.fdopen(fd, 'wb') as target, source.open('rb') as original:
                for chunk in iter(lambda: original.read(1024 * 1024), b''):
                    if written + len(chunk) > maximum:
                        raise StorageError('source_exceeds_reserved_size')
                    if lease:
                        lease.checkpoint(write_bytes=len(chunk))
                    target.write(chunk)
                    written += len(chunk)
                    digest.update(chunk)
                if digest.hexdigest() != sha256:
                    raise StorageError('source_changed_during_copy')
                target.flush()
                os.fsync(target.fileno())
            shutil.copystat(source, name, follow_symlinks=False)
            os.chmod(name, 0o600)
            os.replace(name, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)
