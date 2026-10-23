"""Reentrant process and OS-store exclusion for registered recording I/O."""

from contextlib import contextmanager
from functools import wraps
from inspect import signature
import sqlite3
import os
from pathlib import Path
import stat
from threading import RLock, local

MEDIA_LOCK = RLock()
_LOCAL = local()


@contextmanager
def media_access(root=None):
    with MEDIA_LOCK:
        if root is None:
            yield
            return
        root = Path(root).resolve()
        key = str(root)
        held = getattr(_LOCAL, "held", {})
        if key in held:
            yield
            return
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        filename = root / ".recording-source-io.sqlite"
        if filename.is_symlink():
            raise ValueError("speech_owner_storage_unavailable")
        connection = sqlite3.connect(filename, timeout=900, isolation_level=None)
        try:
            metadata = filename.stat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or hasattr(os, "getuid")
                and metadata.st_uid != os.getuid()
            ):
                raise ValueError("speech_owner_storage_unavailable")
            filename.chmod(0o600)
            connection.execute("CREATE TABLE IF NOT EXISTS media_owner(id INTEGER)")
            connection.execute("BEGIN EXCLUSIVE")
            _LOCAL.held = {**held, key: connection}
            try:
                yield
            finally:
                _LOCAL.held = held
                connection.rollback()
        finally:
            connection.close()


def guard_media(function):
    parameters = signature(function)

    @wraps(function)
    def guarded(*args, **kwargs):
        arguments = parameters.bind_partial(*args, **kwargs).arguments
        receiver = arguments.get("self")
        if receiver is not None:
            root = getattr(
                receiver, "source_store_root", getattr(receiver, "store_root", None)
            )
        else:
            from transcript_store import store_dir

            root = store_dir(arguments.get("root"))
        with media_access(root):
            return function(*args, **kwargs)

    return guarded
