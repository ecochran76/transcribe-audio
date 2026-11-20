#!/usr/bin/env python3
"""Prepare an immutable storage controller generation; activation is explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_maintenance import config_digest, plain, private_json, write_receipt

# The installed consumer set is explicit. No private data, .git or virtualenv is
# copied and installed code does not resolve imports through a mutable checkout.
FILES = ('storage_maintenance.py', 'storage_catalog.py', 'storage_policy.py',
         'storage_backups.py', 'recording_source_guard.py',
         'scripts/storage_maintenance.py', 'systemd/transcribe-storage-maintenance.service',
         'systemd/transcribe-storage-maintenance.timer')


def install(config_path, destination, python, *, activate=False, unit_root=None):
    config_path, destination = map(plain, (config_path, destination))
    # A virtualenv interpreter normally is a symlink; preserve its venv entry
    # path while restricting private configuration and release roots.
    python = Path(python).expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('executable Python interpreter required')
    private_json(config_path)
    source = Path(__file__).resolve().parents[1]
    files = {name: config_digest(source / name) for name in FILES}
    generation = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    release = destination / generation
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.stat().st_uid != os.getuid() or destination.stat().st_mode & 0o077:
        raise ValueError('private installation directory required')
    if not release.exists():
        release.mkdir(mode=0o700)
        for name in FILES:
            target = release / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            target.chmod(0o400)
    if any(config_digest(release / name) != digest for name, digest in files.items()):
        raise ValueError('immutable release identity mismatch')
    unit_root = plain(unit_root or (destination / 'units'))
    unit_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    def quote(value):
        # systemd specifiers and argument parsing are separate from shell quoting.
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
    service = (release / 'systemd/transcribe-storage-maintenance.service').read_text()
    service = service.replace('@PYTHON@', quote(python)).replace('@SCRIPT@', quote(release / 'scripts/storage_maintenance.py')).replace('@CONFIG@', quote(config_path))
    (unit_root / 'transcribe-storage-maintenance.service').write_text(service)
    shutil.copyfile(release / 'systemd/transcribe-storage-maintenance.timer', unit_root / 'transcribe-storage-maintenance.timer')
    source_commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    receipt = {'version': 1, 'source_commit': source_commit, 'source_generation': generation, 'config_sha256': config_digest(config_path),
               'release': str(release), 'files': {str(release / name): digest for name, digest in files.items()},
               'units': str(unit_root), 'activated': False, 'consumer_qualification': 'not_proven'}
    if activate:
        configured = private_json(config_path)
        if configured.get('source_generation') != generation:
            raise ValueError('config must bind prepared source_generation before activation')
        from storage_maintenance import Maintenance
        if not Maintenance(configured, config_path=config_path).qualified():
            raise ValueError('installed consumer qualification required before activation')
        active_units = plain(Path.home() / '.config/systemd/user')
        active_units.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ('transcribe-storage-maintenance.service', 'transcribe-storage-maintenance.timer'):
            shutil.copyfile(unit_root / name, active_units / name)
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        subprocess.run(['systemctl', '--user', 'enable', '--now', 'transcribe-storage-maintenance.timer'], check=True)
        receipt['activated'] = True
    write_receipt(destination / 'installation.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(install(args.config, args.destination, args.python, activate=args.activate), sort_keys=True))


if __name__ == '__main__':
    main()
