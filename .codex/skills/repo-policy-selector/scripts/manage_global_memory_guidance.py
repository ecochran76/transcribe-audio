#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEADING = 'Graphiti Discovery And Memory'
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = SCRIPT_ROOT / 'templates' / 'global-graphiti-guidance.md'
DEFAULT_TARGET = Path.home() / '.codex' / 'AGENTS.md'
DEFAULT_RECEIPTS = Path.home() / '.graphiti-openclaw' / 'state' / 'closeout-memory'
DEFAULT_JOB_STORE = Path.home() / '.graphiti-openclaw' / 'state' / 'memory_jobs.json'
DISPOSITIONS = {'queued', 'duplicate_noop', 'not_durable', 'forbidden', 'unavailable'}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'expected JSON object: {path}')
    return value


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def install_guidance_text(current: str, template: str) -> str:
    replacement = template.strip() + '\n'
    pattern = re.compile(
        rf'(?ms)^## {re.escape(HEADING)}\n.*?(?=^## |\Z)'
    )
    if pattern.search(current):
        updated = pattern.sub(replacement + '\n', current, count=1)
    else:
        updated = current.rstrip() + '\n\n' + replacement
    return updated.rstrip() + '\n'


def install_guidance(target: Path, template_path: Path, *, check: bool) -> dict[str, Any]:
    template = template_path.read_text(encoding='utf-8')
    current = target.read_text(encoding='utf-8') if target.exists() else ''
    expected = install_guidance_text(current, template)
    changed = expected != current
    if check:
        return {'target': str(target), 'changed': changed, 'status': 'drift' if changed else 'current'}
    if changed:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(expected)
            if target.exists():
                temporary.chmod(target.stat().st_mode)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {'target': str(target), 'changed': changed, 'status': 'installed'}


def audit_closeouts(
    *,
    receipts_dir: Path,
    job_store: Path,
    since: str,
    expected_closeouts: int | None,
) -> dict[str, Any]:
    since_time = parse_timestamp(since)
    if since_time is None:
        raise ValueError('since is required')

    receipts: list[dict[str, Any]] = []
    if receipts_dir.exists():
        for path in sorted(receipts_dir.glob('*-remember.json')):
            receipt = read_json(path)
            recorded_at = parse_timestamp(receipt.get('recorded_at') or receipt.get('queued_at'))
            if recorded_at is None:
                recorded_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if recorded_at >= since_time and receipt.get('disposition') in DISPOSITIONS:
                receipts.append({**receipt, '_path': str(path), '_recorded_at': recorded_at.isoformat()})

    disposition_counts = Counter(receipt['disposition'] for receipt in receipts)
    receipted_job_ids = {
        receipt['job_id']
        for receipt in receipts
        if receipt.get('disposition') == 'queued' and receipt.get('job_id')
    }
    jobs_value = read_json(job_store) if job_store.exists() else {'jobs': []}
    jobs = []
    for job in jobs_value.get('jobs', []):
        if not isinstance(job, dict):
            continue
        queued_at = parse_timestamp(job.get('queued_at'))
        if queued_at is not None and queued_at >= since_time:
            jobs.append(job)

    jobs_by_id = {job.get('job_id'): job for job in jobs if job.get('job_id')}
    queued_delivery = Counter(
        jobs_by_id[job_id].get('status', 'missing') if job_id in jobs_by_id else 'missing'
        for job_id in receipted_job_ids
    )
    unreceipted = sorted(
        job_id for job_id in jobs_by_id if job_id not in receipted_job_ids
    )
    observed = len(receipts)
    missing = None if expected_closeouts is None else max(expected_closeouts - observed, 0)
    return {
        'schema_version': 'memory_closeout_observability.v0',
        'since': since_time.isoformat(),
        'expected_closeout_count': expected_closeouts,
        'observed_disposition_count': observed,
        'missing_disposition_count': missing,
        'disposition_counts': dict(sorted(disposition_counts.items())),
        'queued_delivery_status_counts': dict(sorted(queued_delivery.items())),
        'unreceipted_memory_job_ids': unreceipted,
        'evidence_boundaries': {
            'policy_compliance': 'observed disposition receipts versus an explicit closeout denominator',
            'queue_acceptance': 'queued receipts with job ids',
            'terminal_processing': 'job-store status for receipted queued jobs',
            'visibility_and_retrieval': 'not measured by this audit',
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    install_parser = subparsers.add_parser('install')
    install_parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    install_parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    install_parser.add_argument('--check', action='store_true')

    audit_parser = subparsers.add_parser('audit')
    audit_parser.add_argument('--receipts-dir', type=Path, default=DEFAULT_RECEIPTS)
    audit_parser.add_argument('--job-store', type=Path, default=DEFAULT_JOB_STORE)
    audit_parser.add_argument('--since', required=True)
    audit_parser.add_argument('--expected-closeouts', type=int)

    args = parser.parse_args()
    if args.command == 'install':
        report = install_guidance(args.target, args.template, check=args.check)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if args.check and report['changed'] else 0

    if args.expected_closeouts is not None and args.expected_closeouts < 0:
        parser.error('--expected-closeouts must be non-negative')
    report = audit_closeouts(
        receipts_dir=args.receipts_dir,
        job_store=args.job_store,
        since=args.since,
        expected_closeouts=args.expected_closeouts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
