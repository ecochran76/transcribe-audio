"""Private operator storage controller. Initial and hourly passes share this path.

Configuration v1 uses explicit owned roots; no directory scanning crosses them.
Inventories are reviewed ownership records, refreshed each pass. Registered live
blobs are rediscovered each pass, but unclassified roles never gain retirement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import http.client
import ipaddress
import json
import math
import os
import re
import subprocess
from pathlib import Path
import shutil
import sqlite3
import ssl
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from recording_source_guard import media_access
from storage_backups import BackupManager, BackupError, TrustedTime
from storage_catalog import StorageCatalog, StorageError, native_storage_identity, file_identity
from storage_policy import (GIB, POLICY_VERSION, BudgetState, ObjectState,
                            TimeObservation, decide, reserve)


class MaintenanceHeld(RuntimeError):
    """An explicit safe checkpoint, not a failed transcription job."""


def plain(path):
    path = Path(path).expanduser().absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise MaintenanceHeld('private_plain_path_required')
    return path


def private_json(path):
    path = plain(path)
    info = path.stat()
    if not path.is_file() or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
        raise MaintenanceHeld('private_json_required')
    return json.loads(path.read_text())


def write_receipt(path, value):
    path = plain(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid() or path.parent.stat().st_mode & 0o077:
        raise MaintenanceHeld('private_state_root_required')
    temporary = path.with_name('.' + uuid4().hex)
    try:
        with temporary.open('x') as stream:
            temporary.chmod(0o600)
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def config_digest(path):
    return hashlib.sha256(plain(path).read_bytes()).hexdigest()


class StableTimeClock:
    """Supported authenticated HTTPS /v1/time, without proxies or redirects.

    No receipt-time UTC claim, local counter bridge or guest-clock projection.
    A fresh external observation is requested at every safe I/O checkpoint.
    """
    def __init__(self, client_config):
        self.config = private_json(client_config)
        self.last_measurement = None

    @staticmethod
    def decode(value):
        try:
            UUID(value['epoch'])
            estimate, policy = value['estimate'], value['counter_policy']
            operators = value['operators']
            measurement = value['measurement']
            if (value['contract_version'] != '2.0' or value['bound'] != 'finite'
                    or value['status'] not in ('qualified_live', 'qualified_holdover')
                    or value['authentication'] != 'NTS-verified'
                    or value['time_scale'] != 'UTC-step; announced-leaps-unavailable'
                    or value.get('reason') is not None or value.get('best_effort') is not None
                    or len(set(operators)) < 2 or any(not x.strip() for x in operators)
                    or not 0 <= policy['rate_ppm'] < 1000000
                    or not 0 < policy['maximum_age_ns'] <= 300000000000
                    or not 0 <= value['acquisition_age_ns'] <= policy['maximum_age_ns']
                    or value['evidence_generation'] <= 0
                    or not str(measurement['counter']).isdigit()
                    or not str(measurement['frequency_hz']).isdigit()
                    or int(measurement['frequency_hz']) <= 0):
                raise ValueError()
            points = [datetime.fromisoformat(estimate[key].replace('Z', '+00:00'))
                      for key in ('earliest_utc', 'utc', 'latest_utc')]
            if any(p.tzinfo is None or p.utcoffset().total_seconds() != 0 for p in points):
                raise ValueError()
            low, mid, high = [p.timestamp() for p in points]
            error = estimate['maximum_error_ns'] / 1e9
            if (not all(math.isfinite(x) for x in (low, mid, high, error))
                    or error < 0 or not low <= mid <= high
                    or abs(max(mid-low, high-mid)-error) > 0.000002
                    or estimate['maximum_error_ms'] != estimate['maximum_error_ns'] / 1e6):
                raise ValueError()
            return TimeObservation(mid, max(error, mid-low, high-mid))
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
            raise MaintenanceHeld('stable_time_client_invalid_evidence') from error

    def __call__(self):
        config = self.config
        url = urlsplit(config['url'])
        try:
            loopback = url.hostname == 'localhost' or ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            loopback = False
        if url.scheme != 'https' or not loopback or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
            raise MaintenanceHeld('stable_time_client_invalid_endpoint')
        token_path = plain(config['token_file'])
        if token_path.stat().st_uid != os.getuid() or token_path.stat().st_mode & 0o077:
            raise MaintenanceHeld('stable_time_client_private_token_required')
        try:
            context = ssl.create_default_context(cafile=str(plain(config['ca_file'])))
            connection = http.client.HTTPSConnection(url.hostname, url.port, context=context, timeout=10)
            try:
                connection.request('GET', '/v1/time', headers={'Authorization': 'Bearer ' + token_path.read_text().strip()})
                response = connection.getresponse()
                if response.status != 200:
                    raise MaintenanceHeld('stable_time_service_http_' + str(response.status))
                payload = response.read(65537)
                if len(payload) > 65536:
                    raise MaintenanceHeld('stable_time_client_oversize_response')
                value = json.loads(payload)
            finally:
                connection.close()
        except (OSError, http.client.HTTPException) as error:
            raise MaintenanceHeld('stable_time_service_transport_unavailable') from error
        observation = self.decode(value)
        marker = (value['epoch'], int(value['measurement']['counter']))
        if self.last_measurement and (marker[0] != self.last_measurement[0] or marker[1] <= self.last_measurement[1]):
            raise MaintenanceHeld('stable_time_epoch_changed_or_replayed')
        self.last_measurement = marker
        return observation


class PassBudget:
    def __init__(self, clock, *, write_limit=16 * GIB, account=None, clock_quantum_bytes=0):
        if type(clock_quantum_bytes) is not int or not 0 <= clock_quantum_bytes <= 64 * 1024 * 1024:
            raise ValueError('invalid_time_checkpoint_quantum')
        self.clock, self.limit, self.account = clock, write_limit, account
        self.clock_quantum_bytes, self.unobserved_io = clock_quantum_bytes, 0
        self.written = self.read = 0
        self.started = clock()
        if self.started is None or not self.started.valid:
            raise MaintenanceHeld('trusted_time_unavailable')
        self.last_observation = self.started

    def checkpoint(self, *, write_bytes=0, read_bytes=0):
        if write_bytes < 0 or read_bytes < 0 or self.written + write_bytes > self.limit:
            raise MaintenanceHeld('pass_write_budget')
        io_bytes = write_bytes + read_bytes
        # Byte limits and durable accounting apply to every chunk. Time is read
        # at bounded I/O checkpoints and every explicit phase boundary, never
        # projected with a guest clock or represented as receipt-time UTC.
        observe = (not io_bytes or not self.clock_quantum_bytes
                   or self.unobserved_io + io_bytes >= self.clock_quantum_bytes)
        now = self.clock() if observe else self.last_observation
        if now is None or not now.valid:
            raise MaintenanceHeld('trusted_time_unavailable')
        if now.utc_seconds + now.uncertainty_seconds - self.started.utc_seconds + self.started.uncertainty_seconds >= 1800:
            raise MaintenanceHeld('pass_time_budget')
        if now.utc_seconds + now.uncertainty_seconds < self.started.utc_seconds - self.started.uncertainty_seconds:
            raise MaintenanceHeld('qualified_time_regressed')
        self.last_observation = now
        self.unobserved_io = 0 if observe else self.unobserved_io + io_bytes
        self.written += write_bytes
        self.read += read_bytes
        if self.account:
            self.account(self)
        return now


class Maintenance:
    def __init__(self, config, *, config_path=None, catalog=None, clock=None, free_bytes=None):
        self.config, self.config_path = config, config_path
        if config.get('version') != 1 or not config.get('source_roots'):
            raise MaintenanceHeld('explicit_v1_owned_roots_required')
        self.state = plain(config['state_root'])
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.state.stat().st_uid != os.getuid() or self.state.stat().st_mode & 0o077:
            raise MaintenanceHeld('private_state_root_required')
        self.catalog = catalog or StorageCatalog(Path(config['store_root']),
            cold_root=Path(config['cold_root']), expected_cold_uuid=config['cold_uuid'],
            source_roots=[Path(p) for p in config['source_roots']])
        self._external_clock = None
        self.clock = clock or self.external_clock
        self.free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)
        self.budget = None
        self.metadata = {}
        self.reservations = {}
        self.item_states = {}

    def external_clock(self):
        if self._external_clock is None:
            try:
                self._external_clock = StableTimeClock(self.config['stable_time_client'])
            except (OSError, KeyError, ValueError) as error:
                raise MaintenanceHeld('stable_time_client_configuration_unavailable') from error
        return self._external_clock()

    @contextmanager
    def worker(self):
        path = plain(self.state / 'worker.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise MaintenanceHeld('worker_already_running') from error
            # The flock proves any old process exited. Old accounting is retained
            # in pass receipts; no concurrent reservations survive this owner.
            yield
        finally:
            os.close(fd)

    def qualified(self):
        try:
            receipt = private_json(self.config['consumer_receipt'])
            if 'api_pid' in receipt:
                pid = receipt['api_pid']
                if type(pid) is not int or pid <= 0:
                    return False
                process = Path('/proc') / str(pid)
                start = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
                if (start != receipt['api_process_start_ticks']
                        or receipt['qualified_consumer'].encode() not in (process / 'cmdline').read_bytes().split(b'\0')
                        or (process / 'stat').read_text().split(') ', 1)[1].split()[19] != start):
                    return False
            return (receipt['version'] == 1 and receipt['status'] == 'qualified'
                    and receipt['source_generation'] == self.config['source_generation']
                    and receipt['config_sha256'] == config_digest(self.config_path)
                    and receipt['scope_roots'] == self.config['source_roots']
                    and receipt['ownership_inventory'] == 'transcribe-owned-v1'
                    and bool(receipt['files'])
                    and all(config_digest(p) == digest for p, digest in receipt['files'].items()))
        except (KeyError, OSError, TypeError, ValueError, IndexError, MaintenanceHeld):
            return False

    def discover(self):
        self.metadata, self.deferred = {}, {}
        qualified = self.qualified()

        def defer(item, reason):
            path = plain(item['path'])
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.deferred[item['id']] = {'id': item['id'], 'sha256': item['sha256'],
                'classification': item.get('classification', 'unclassified'), 'size': size,
                'logical_path': str(path), 'active_path': str(path), 'guard': '{}',
                'deferred_reason': reason, 'content_verified': False}

        for filename in self.config.get('inventories', []):
            inventory = private_json(filename)
            if (not qualified or inventory.get('version') != 1
                    or inventory.get('source_generation') != self.config['source_generation']
                    or inventory.get('ownership_inventory') != 'transcribe-owned-v1'):
                continue
            for item in inventory['objects']:
                self.metadata[item['id']] = item
                if (item.get('classification') in ('canonical_source', 'source_recording')
                        and item['id'] not in self.config.get('replica_proofs', {})
                        and item['id'] not in self.config.get('copy_only_ids', [])):
                    defer(item, 'independent_replica_custody_unqualified')
                    continue
                try:
                    existing = self.catalog.object(item['id'])
                    origin = item.get('source_sha256', item.get('related_source_sha256'))
                    if origin and self.catalog.origin(item['id']) != origin:
                        self.catalog.register(item['id'], Path(existing['logical_path']),
                            classification=existing['classification'], expected_sha256=existing['sha256'], source_sha256=origin)
                except LookupError:
                    self.catalog.register(item['id'], Path(item['path']),
                        classification=item.get('classification', 'unclassified'), expected_sha256=item['sha256'],
                        source_sha256=item.get('source_sha256', item.get('related_source_sha256')))
        # Query authoritative registration each pass; no recursive scan or frozen
        # one-off list can accidentally confer ownership on arbitrary files.
        with closing(sqlite3.connect(Path(self.config['live_database']).as_uri() + '?mode=ro', uri=True)) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='blobs'").fetchone():
                for identity, path, digest, role in db.execute('SELECT id,stored_path,sha256,role FROM blobs'):
                    if not any(plain(path).is_relative_to(plain(p)) for p in self.config['source_roots']):
                        continue
                    if identity in self.deferred:
                        continue
                    try:
                        self.catalog.object(identity)
                    except LookupError:
                        # Preserve a visible hold without hashing/copying bytes
                        # whose publication/custody has not been qualified.
                        defer({'id': identity, 'path': path, 'sha256': digest},
                              'registered_blob_custody_unqualified')
        records = {item['id']: item for item in self.catalog.objects()}
        for identity, item in self.deferred.items():
            records.setdefault(identity, item)
        return list(records.values())

    def plan(self):
        try:
            now = self.clock()
        except MaintenanceHeld:
            now = None
        result = []
        for item in self.discover():
            if item['id'] in self.deferred:
                result.append({**item, 'action': 'hold', 'reason': self.deferred[item['id']]['deferred_reason']})
                continue
            metadata = self.metadata.get(item['id'], {})
            def observation(key):
                value = metadata.get(key)
                return TimeObservation(**value) if value and metadata.get('time_provenance') == 'stable-time-v2' else None
            decision = decide(ObjectState(item['classification'],
                ownership_known=metadata.get('ownership_known', False),
                active=metadata.get('active', True), pinned=self.catalog.is_pinned(item['id']),
                consumers_qualified=self.qualified(), completed_at=observation('completed_at'),
                inactive_since=observation('inactive_since'),
                publication_verified=metadata.get('publication_verified', False),
                recovery_verified=metadata.get('recovery_verified', False)), now)
            if item['classification'] == 'historical_derivative' and not self.catalog.origin(item['id']):
                from storage_policy import Decision
                decision = Decision('hold', 'source_lineage_unqualified')
            if not self.qualified():
                from storage_policy import Decision
                decision = Decision('hold', 'consumer_compatibility_unqualified')
            result.append({**item, 'action': decision.action, 'reason': decision.reason})
        return result

    def account(self, budget):
        write_receipt(self.state / 'accounting.json', {'version': 1,
            'written_bytes': budget.written, 'read_bytes': budget.read,
            'reservations': self.reservations, 'started_at': budget.started.utc_seconds})

    def _pause_attempt(self, identity, attempt, reason, states_path):
        if self.item_states.get(identity, {}).get('state') == 'attempting':
            self.item_states[identity] = {'state': 'held', 'attempts': attempt - 1, 'reason': reason}
            write_receipt(states_path, self.item_states)

    def admit(self, identity, payload, transient=0):
        if identity in self.reservations:
            raise MaintenanceHeld('reservation_already_owned')
        now = self.budget.checkpoint()
        outstanding = sum(self.reservations.values())
        # reserve() currently requires nonoverlapping observations. Waiting on
        # the external authority is the safe path; never narrow uncertainty.
        for _ in range(4):
            if now.utc_seconds - now.uncertainty_seconds >= self.budget.started.utc_seconds + self.budget.started.uncertainty_seconds:
                break
            import time
            time.sleep(0.5)
            now = self.budget.checkpoint()
        decision = reserve(BudgetState(outstanding, self.free_bytes(self.catalog.root),
                           self.budget.written, self.budget.started),
            transient_bytes=transient, nvme_write_bytes=0, payload_write_bytes=payload,
            duration_seconds=0, now=now)
        if not decision.admitted:
            raise MaintenanceHeld(decision.reason)
        self.reservations[identity] = transient
        self.account(self.budget)

    def replica(self, item):
        proof = self.config.get('replica_proofs', {}).get(item['id'])
        if not proof:
            return None, '', 'independent_replica_custody_unqualified'
        try:
            path = plain(proof['path'])
            cold = proof['cold_backing_drive'].upper()
            drive = proof['replica_drive'].upper()
            if (not re.fullmatch('[A-Z]', cold) or not re.fullmatch('[A-Z]', drive)
                    or cold != self.config['cold_backing_drive'].upper()
                    or not path.is_relative_to(Path('/mnt') / drive.lower())
                    or not proof['custody'] or proof['sha256'] != item['sha256']
                    or file_identity(path) != proof['identity']):
                raise ValueError()
            # Caller-supplied names never establish physical independence. Query
            # current Windows disk identity for both mounted drive letters.
            command = "@('" + cold + "','" + drive + "') | ForEach-Object { $p=Get-Partition -DriveLetter $_ -ErrorAction Stop; $d=$p|Get-Disk -ErrorAction Stop; [pscustomobject]@{drive=$_;disk=$d.UniqueId} } | ConvertTo-Json -Compress"
            result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                    capture_output=True, text=True, timeout=15, check=True)
            disks = {row['drive']: row['disk'].strip() for row in json.loads(result.stdout)}
            if not disks[cold] or not disks[drive] or disks[cold] == disks[drive]:
                return None, '', 'independent_replica_same_host_disk'
            return path, 'windows-physical-disk:' + disks[drive], ''
        except (OSError, KeyError, ValueError, TypeError, subprocess.SubprocessError):
            return None, '', 'independent_replica_custody_unqualified'

    def backup_manager(self):
        def guard(path):
            native_storage_identity(path, self.config['cold_uuid'])
            if self.free_bytes(path) < 1024 * 1024:
                raise MaintenanceHeld('backup_destination_capacity')
            return True
        def clock():
            now = self.clock()
            return TrustedTime(datetime.fromtimestamp(now.utc_seconds, timezone.utc), now.uncertainty_seconds)
        return BackupManager(Path(self.config['live_database']), Path(self.config['backup_root']),
            guard, clock, checkpoint=self.budget.checkpoint)

    def reconcile_erasure(self, restored):
        authority = plain(self.config['erasure_authority'])
        if authority != plain(self.config['live_database']):
            raise MaintenanceHeld('explicit_live_erasure_authority_required')
        # Guard serializes current owner erasure alongside the authoritative read
        # and restore reconciliation. Never treat an unknown owner schema as empty.
        sources = [authority, *[plain(p) for p in self.config.get('owner_fences', [])]]
        digests = set()
        with media_access(self.catalog.root):
            for source in sources:
                with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as current:
                    # The live application lazily creates this table at first
                    # erasure. Its verified absence is an empty current set.
                    if current.execute("SELECT 1 FROM sqlite_master WHERE name='recording_source_erasures'").fetchone():
                        digests.update(row[0] for row in current.execute('SELECT sha256 FROM recording_source_erasures'))
                    elif source != authority:
                        raise MaintenanceHeld('owner_erasure_schema_unavailable')
            restored.execute('CREATE TABLE IF NOT EXISTS recording_source_erasures(sha256 TEXT PRIMARY KEY,owner TEXT,request TEXT)')
            for digest in digests:
                restored.execute('INSERT OR IGNORE INTO recording_source_erasures(sha256,owner,request) VALUES(?,?,?)',
                                 (digest, 'current-live-authority', 'isolated-restore'))
                if restored.execute("SELECT 1 FROM sqlite_master WHERE name='storage_objects'").fetchone():
                    restored.execute("UPDATE storage_objects SET active_path='' WHERE sha256=?", (digest,))
                    if restored.execute("SELECT 1 FROM sqlite_master WHERE name='storage_origins'").fetchone():
                        restored.execute("UPDATE storage_objects SET active_path='' WHERE id IN (SELECT object_id FROM storage_origins WHERE source_sha256=?)", (digest,))
                if restored.execute("SELECT 1 FROM sqlite_master WHERE name='blobs'").fetchone():
                    restored.execute("UPDATE blobs SET stored_path='' WHERE sha256=?", (digest,))
        return {'current_tombstones_applied': True, 'source': str(authority), 'fences': len(digests)}

    def check_references(self, db):
        failures = []
        for identity, digest, classification in db.execute('SELECT id,sha256,classification FROM storage_objects'):
            self.budget.checkpoint()
            migration = self.catalog.migration(identity)
            if (classification in ('private_cache', 'completed_scratch')
                    and migration and migration['phase'] == 'evicted'):
                continue
            origin = self.catalog.origin(identity)
            if db.execute('SELECT 1 FROM recording_source_erasures WHERE sha256 IN (?,?)', (digest, origin or digest)).fetchone():
                continue
            try:
                self.catalog.resolve(identity)
            except (OSError, StorageError, LookupError):
                failures.append('unresolved media: ' + identity)
        return failures

    def recover_media(self, identity, remaining):
        with self.catalog.pin(identity, owner='restore-drill') as path:
            size = path.stat().st_size
            if size > remaining:
                raise MaintenanceHeld('media_recovery_bound')
            chunks = []
            with path.open('rb') as stream:
                while chunk := stream.read(min(1024 * 1024, remaining + 1)):
                    if len(chunk) > remaining:
                        raise MaintenanceHeld('media_recovery_bound')
                    self.budget.checkpoint(read_bytes=len(chunk))
                    chunks.append(chunk)
                    remaining -= len(chunk)
            return b''.join(chunks)

    def run(self, command, *, snapshot=None, destination=None):
        if command == 'status':
            path = self.state / 'latest.json'
            return private_json(path) if path.exists() else {'status': 'never_run', 'policy_version': POLICY_VERSION}
        with self.worker():
            states_path = self.state / 'item-states.json'
            self.item_states = private_json(states_path) if states_path.exists() else {}
            if command == 'inventory':
                return {'version': 1, 'objects': self.discover(), 'policy_version': POLICY_VERSION}
            if command == 'plan':
                return {'version': 1, 'items': self.plan(), 'policy_version': POLICY_VERSION}
            result = {'version': 1, 'command': command, 'policy_version': POLICY_VERSION,
                      'source_generation': self.config.get('source_generation'),
                      'config_sha256': config_digest(self.config_path) if self.config_path else None,
                      'items': [], 'status': 'running'}
            try:
                self.budget = PassBudget(self.clock, account=self.account, clock_quantum_bytes=64 * 1024 * 1024)
                self.catalog.checkpoint = self.budget.checkpoint
                if command in ('backup', 'maintenance', 'restore'):
                    manager = self.backup_manager()
                    size = Path(self.config['live_database']).stat().st_size
                    self.admit('database', size * 5 + 1048576, size * 3)
                    if self.free_bytes(Path(self.config['backup_root'])) < size * 3 + 1048576:
                        raise MaintenanceHeld('backup_destination_capacity')
                    if command == 'restore':
                        if not snapshot or not destination:
                            raise MaintenanceHeld('snapshot_and_isolated_destination_required')
                        # Isolated restore writes use the same native guard; no
                        # restore can consume the NVMe admission reserve.
                        native_storage_identity(plain(destination).parent, self.config['cold_uuid'])
                        result['restore'] = str(manager.restore(snapshot, destination, self.reconcile_erasure))
                    else:
                        result['backup'] = manager.snapshot()
                        result['rotation'] = manager.rotate()
                        if command == 'maintenance':
                            samples = [{'media_id': p['id'], 'sha256': p['sha256']}
                                       for p in self.catalog.objects() if p['active_path'] and 0 < p['size'] <= 4 * 1024 * 1024][:16]
                            result['drill'] = manager.monthly_restore_drill(result['backup']['snapshot_id'],
                                Path(self.config['backup_root']) / ('isolated-' + uuid4().hex),
                                reconcile_erasure=self.reconcile_erasure, check_references=self.check_references,
                                recover_media=self.recover_media, samples=samples)
                    self.reservations.pop('database', None)
                planned = []
                if command in ('migrate', 'reconcile', 'maintenance'):
                    planned = sorted(self.plan(), key=lambda item: (
                        item['classification'] != 'historical_derivative', item['id']))
                    for item in planned:
                        outcome = {'object_id': item['id'], 'state': 'held', 'reason': item['reason']}
                        previous_migration = self.catalog.migration(item['id'])
                        if previous_migration and previous_migration['phase'] == 'retired':
                            # resolve checks the recorded destination identity;
                            # do not recopy or rehash a completed retirement.
                            self.catalog.resolve(item['id'])
                            result['items'].append({'object_id': item['id'], 'state': 'retired', 'reason': 'already_retired'})
                            continue
                        if previous_migration and previous_migration['phase'] == 'evicted':
                            result['items'].append({'object_id': item['id'], 'state': 'evicted', 'reason': 'already_evicted'})
                            continue
                        disposal = item['action'] in ('evict_cache', 'evict_scratch')
                        canonical = item['classification'] in ('canonical_source', 'source_recording')
                        if canonical:
                            replica, domain, replica_reason = self.replica(item)
                            if replica is None and item['id'] not in self.config.get('copy_only_ids', []):
                                result['items'].append({**outcome, 'reason': replica_reason})
                                continue
                            if replica is None and previous_migration and previous_migration['phase'] in ('published', 'retirement-pending'):
                                self.catalog.resolve(item['id'])
                                result['items'].append({'object_id': item['id'], 'state': 'copy-only', 'reason': replica_reason})
                                continue
                        if item['action'] not in ('cold_move', 'quarantine', 'evict_cache', 'evict_scratch'):
                            result['items'].append(outcome)
                            continue
                        if not disposal and item['size'] > self.budget.limit:
                            result['items'].append({'object_id': item['id'], 'state': 'held', 'reason': 'individual_object_exceeds_pass_bound'})
                            continue
                        previous = self.item_states.get(item['id'], {})
                        if previous.get('state') == 'quarantined' or previous.get('attempts', 0) >= 3:
                            result['items'].append({'object_id': item['id'], 'state': previous.get('state', 'held'),
                                'reason': previous.get('reason', 'object_retry_exhausted'), 'attempts': previous.get('attempts', 0)})
                            continue
                        # Unknown media can be copied to quarantine, never retired.
                        for attempt in range(previous.get('attempts', 0) + 1, 4):
                            try:
                                self.admit(item['id'], 0 if disposal else item['size'], 0)
                                if not disposal:
                                    self.catalog._guard(self.catalog.cold_root)
                                old_source = Path(item['logical_path'])
                                before = old_source.stat() if old_source.exists() else None
                                if not disposal and self.free_bytes(self.catalog.cold_root) < item['size'] + 1024 * 1024:
                                    raise MaintenanceHeld('cold_destination_capacity')
                                # Native independent replica proofs are separate
                                # evidence, never an arbitrary configured domain.
                                retire = item['classification'] not in ('canonical_source', 'source_recording', 'unclassified') and item['action'] != 'quarantine'
                                self.item_states[item['id']] = {'state': 'attempting', 'attempts': attempt, 'reason': 'interrupted_attempt'}
                                write_receipt(states_path, self.item_states)
                                replica, domain, replica_reason = (None, '', '')
                                if item['classification'] in ('canonical_source', 'source_recording'):
                                    replica, domain, replica_reason = self.replica(item)
                                    retire = replica is not None and item['action'] != 'quarantine'
                                if disposal:
                                    outcome = self.catalog.evict(item['id'], consumers_qualified=self.qualified(),
                                        eligibility_verified=True)
                                else:
                                    custody = self.config.get('replica_proofs', {}).get(item['id'], {}).get('custody', '')
                                    outcome = self.catalog.migrate(item['id'], consumers_qualified=self.qualified(), retire=retire,
                                        independent_replica=replica, replica_failure_domain=domain, replica_custody=custody)
                                if item['action'] == 'quarantine':
                                    outcome.update(state='quarantined', reason=item['reason'])
                                if replica_reason and outcome['state'] != 'retired':
                                    outcome['reason'] = replica_reason
                                outcome['attempts'] = attempt
                                outcome['exclusive_blocks_recovered_bytes'] = (before.st_blocks * 512
                                    if before and before.st_nlink == 1 and not old_source.exists() and outcome['state'] in ('retired', 'evicted') else 0)
                                break
                            except MaintenanceHeld as error:
                                self._pause_attempt(item['id'], attempt, str(error), states_path)
                                outcome.update(reason=str(error), attempts=attempt)
                                result['items'].append(outcome)
                                raise
                            except OSError as error:
                                if error.errno in (errno.EIO, errno.ENODEV, errno.ESTALE, errno.ENOENT):
                                    self._pause_attempt(item['id'], attempt, 'systemic_storage_outage', states_path)
                                    raise MaintenanceHeld('systemic_storage_outage') from error
                                outcome['reason'] = 'object_transient_' + type(error).__name__
                                outcome['attempts'] = attempt
                            except StorageError as error:
                                reason = str(error)
                                if any(word in reason for word in ('mount', 'native_cold', 'storage_identity')):
                                    self._pause_attempt(item['id'], attempt, reason, states_path)
                                    raise MaintenanceHeld(reason) from error
                                outcome.update(state='quarantined' if any(w in reason for w in ('conflict', 'changed', 'hash')) else 'held', reason=reason)
                                break
                            finally:
                                self.reservations.pop(item['id'], None)
                                self.account(self.budget)
                        self.item_states[item['id']] = {**outcome, 'attempts': outcome.get('attempts', 0) if outcome.get('reason', '').startswith('object_transient_') else 0}
                        write_receipt(states_path, self.item_states)
                        result['items'].append(outcome)
                result['status'] = 'complete'
            except (MaintenanceHeld, StorageError, BackupError, OSError, sqlite3.Error, KeyError, ValueError, TypeError, LookupError) as error:
                result.update(status='held', reason=str(error) if isinstance(error, (MaintenanceHeld, StorageError, BackupError)) else type(error).__name__)
            finally:
                if result['status'] == 'running':
                    result.update(status='held', reason='interrupted_operation')
                if result['status'] == 'held':
                    seen = {p['object_id'] for p in result['items']}
                    for item in locals().get('planned', []):
                        if item['id'] not in seen:
                            result['items'].append({'object_id': item['id'], 'state': 'held', 'reason': result['reason']})
                if self.budget:
                    result.update(written_bytes=self.budget.written, read_bytes=self.budget.read)
                self.catalog.checkpoint = None
                self.reservations.clear()
                if self.budget:
                    self.account(self.budget)
                result['receipt_id'] = uuid4().hex
                write_receipt(self.state / (result['receipt_id'] + '.json'), result)
                write_receipt(self.state / 'latest.json', result)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('command', choices=('inventory', 'plan', 'migrate', 'reconcile', 'status', 'backup', 'restore', 'maintenance'))
    parser.add_argument('--snapshot')
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args(argv)
    try:
        config = private_json(args.config)
        outcome = Maintenance(config, config_path=args.config).run(args.command,
                    snapshot=args.snapshot, destination=args.destination)
    except (MaintenanceHeld, OSError, KeyError, ValueError) as error:
        outcome = {'status': 'held', 'reason': str(error) if isinstance(error, MaintenanceHeld) else type(error).__name__}
    print(json.dumps(outcome, sort_keys=True))
    return 2 if outcome.get('status') == 'held' else 0



@contextmanager
def reserve_transient(config_path, owner, payload_bytes):
    """Shared producer admission, independent of maintenance's cold-copy budget.

    Foreground writers reserve their worst-case NEW output before decoding or
    staging, call lease.checkpoint(write_bytes=N) before writes, and use a hard
    output bound for subprocess decoders. Existing over-budget data remains debt;
    this context grants no eviction authority. Explicit transient_roots enumerate
    owned staging directories, never Speech or package/model stores.
    """
    config = private_json(config_path)
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise MaintenanceHeld('invalid_transient_reservation')
    roots = [plain(p) for p in config.get('transient_roots', [])]
    if not roots or not owner:
        raise MaintenanceHeld('explicit_transient_ownership_required')
    owned = [plain(p) for p in config['source_roots']]
    if any(not any(root.is_relative_to(parent) for parent in owned) for root in roots):
        raise MaintenanceHeld('transient_root_outside_owned_scope')
    state = plain(config['state_root'])
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.stat().st_uid != os.getuid() or state.stat().st_mode & 0o077:
        raise MaintenanceHeld('private_state_root_required')
    lock_path = state / 'transient-admission.lock'
    lock = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    lease_id = uuid4().hex
    lease_path = state / ('lease-' + lease_id)
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lease_fd, fcntl.LOCK_EX)
    filename = plain(state / 'transient-reservations.sqlite3')
    database = sqlite3.connect(filename)
    filename.chmod(0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        database.execute('CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY,owner TEXT,bytes INTEGER)')
        database.execute('BEGIN IMMEDIATE')
        for identity, in database.execute('SELECT id FROM reservations').fetchall():
            old_path = plain(state / ('lease-' + identity))
            try:
                old_fd = os.open(old_path, os.O_RDWR | os.O_NOFOLLOW)
            except FileNotFoundError:
                database.execute('DELETE FROM reservations WHERE id=?', (identity,))
                continue
            try:
                try:
                    fcntl.flock(old_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                database.execute('DELETE FROM reservations WHERE id=?', (identity,))
                old_path.unlink()
            finally:
                os.close(old_fd)
        used, seen = 0, set()
        for root in roots:
            if not root.exists():
                continue
            for directory, dirs, files in os.walk(root, followlinks=False):
                for name in dirs + files:
                    path = plain(Path(directory) / name)
                    info = path.stat()
                    if path.is_file() and (info.st_dev, info.st_ino) not in seen:
                        if info.st_uid != os.getuid():
                            raise MaintenanceHeld('transient_ownership_unqualified')
                        seen.add((info.st_dev, info.st_ino))
                        used += info.st_size
        outstanding = database.execute('SELECT COALESCE(sum(bytes),0) FROM reservations').fetchone()[0]
        if used + outstanding + payload_bytes > 20 * GIB:
            raise MaintenanceHeld('transient_budget')
        if shutil.disk_usage(plain(config['store_root'])).free - outstanding - payload_bytes < 100 * GIB:
            raise MaintenanceHeld('nvme_reserve')
        database.execute('INSERT INTO reservations VALUES(?,?,?)', (lease_id, str(owner), payload_bytes))
        database.commit()
        fcntl.flock(lock, fcntl.LOCK_UN)
        class Lease:
            written = 0
            limit = payload_bytes

            def checkpoint(self, *, write_bytes):
                if type(write_bytes) is not int or write_bytes < 0 or self.written + write_bytes > self.limit:
                    raise MaintenanceHeld('transient_reservation_exceeded')
                if shutil.disk_usage(plain(config['store_root'])).free - write_bytes < 100 * GIB:
                    raise MaintenanceHeld('nvme_reserve')
                self.written += write_bytes
        yield Lease()
    finally:
        database.rollback()
        fcntl.flock(lock, fcntl.LOCK_EX)
        database.execute('DELETE FROM reservations WHERE id=?', (lease_id,))
        database.commit()
        database.close()
        lease_path.unlink(missing_ok=True)
        os.close(lease_fd)
        os.close(lock)


if __name__ == '__main__':
    raise SystemExit(main())
