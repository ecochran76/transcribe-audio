#!/usr/bin/env python3
"""Watcher adapter for the existing manager-owned diarized recording pipeline."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import managed_recording_cli as managed


def run(source, config, *, ingress_directory=None, recording_id=None,
        conversation_id=None, existing_transcript=None):
    """Persist ingress identity before submitting; resume the same CLI generation."""
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('explicit media file required')
    source_sha = managed._hash_file(source)
    identity = dict(source=str(source), sha256=source_sha)
    key = hashlib.sha256(managed._json(identity).encode()).hexdigest()
    if ingress_directory:
        ingress_root = Path(ingress_directory).expanduser().resolve()
    else:
        state = managed._private_directory(
            Path(config['processor']['state_directory']).expanduser().resolve())
        ingress_root = state/'watcher-ingress'
    directory = managed._private_directory(ingress_root)
    old = {}
    old_ref = None
    if existing_transcript:
        existing_transcript = Path(existing_transcript).expanduser().resolve(strict=True)
        raw = managed._private_read(existing_transcript)
        old = json.loads(raw)
        if not isinstance(old, dict):
            raise ValueError('existing transcript object required')
        old_ref = dict(path=str(existing_transcript), sha256=hashlib.sha256(raw).hexdigest())
    supplied = dict(recording_id=recording_id, conversation_id=conversation_id)
    for name, value in supplied.items():
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value)>256):
            raise ValueError('bounded '+name+' required')
        if value and old.get(name) and value != old[name]:
            raise ValueError('existing transcript '+name+' conflict')
    fd = os.open(directory/(key+'.lock'), os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('managed recording ingress already owned') from None
        path = directory/(key+'.json')
        if path.exists():
            ingress = json.loads(managed._private_read(path))
            if ingress['source'] != identity or ingress['existing_transcript'] != old_ref:
                raise ValueError('immutable ingress metadata conflict')
            for name, value in supplied.items():
                if value is not None and value != ingress[name]:
                    raise ValueError('immutable ingress '+name+' conflict')
        else:
            ingress = dict(schema_version=1, source=identity, existing_transcript=old_ref)
            for name, value in supplied.items():
                ingress[name] = value or old.get(name) or str(uuid4())
            managed._save(path, ingress)
        if managed._hash_file(source) != source_sha:
            raise RuntimeError('source changed before managed submission')
        receipt = managed.run(source, config, recording_id=ingress['recording_id'],
                              conversation_id=ingress['conversation_id'],
                              existing_transcript=existing_transcript)
        paths = receipt.get('output_paths', {})
        if (receipt.get('published') is False or receipt.get('state') not in (None, 'complete')
                or not receipt.get('generation')
                or any(not isinstance(paths.get(kind), str) or not Path(paths[kind]).is_file()
                       for kind in ('docx','txt','json'))):
            raise RuntimeError('managed recording has not published complete artifacts')
        artifact = json.loads(Path(paths['json']).read_text())
        if (any(artifact.get(name) != ingress[name] for name in supplied)
                or artifact.get('output_paths') != paths
                or managed._hash_file(source) != source_sha):
            raise RuntimeError('published recording identity or source conflict')
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--config', type=Path, required=True, help='Existing private manager recording configuration')
    parser.add_argument('--ingress-directory', type=Path)
    parser.add_argument('--recording-id')
    parser.add_argument('--conversation-id')
    parser.add_argument('--existing-transcript', type=Path)
    parser.add_argument('--text-output', action='store_true', help='Compatible flag; managed publication always includes TXT')
    args = parser.parse_args(argv)
    try:
        config = json.loads(managed._private_read(args.config))
        receipt = run(args.media, config, ingress_directory=args.ingress_directory,
                      recording_id=args.recording_id, conversation_id=args.conversation_id,
                      existing_transcript=args.existing_transcript)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        print('Managed recording failed: '+str(error), file=sys.stderr)
        return 1
    print(managed._json(receipt))
    print('TRANSCRIPT_ARTIFACT_JSON='+receipt['output_paths']['json'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
