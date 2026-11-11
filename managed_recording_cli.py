#!/usr/bin/env python3
"""Explicit CPU media preparation and managed diarized recording publication."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import stat
import subprocess
import sys
import tempfile
import time
import wave

from managed_recording import RecordingProcessor, RecordingProcessingError


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _private_read(path, limit=4*1024*1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('private owned file required')
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('private file exceeds byte bound')
    return raw


def _private_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('private owned directory required')
    return path


def _save(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.manifest-')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(_json(value)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _bounded(config, name, default, minimum, maximum):
    value = config.get(name, default)
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('invalid bounded '+name)
    return value


def attribute_words(words, turns):
    """Return millisecond utterances; uncovered/competing words remain UNKNOWN.

    A word must be covered in full by turns of a single resolved speaker. Merely
    having its midpoint near a turn does not establish attribution across a gap.
    """
    utterances = []
    for word in sorted(words, key=lambda item: (item['start_seconds'], item['end_seconds'])):
        start, end = word['start_seconds'], word['end_seconds']
        overlap = [turn for turn in turns if
                   (turn['start_seconds'] < end and turn['end_seconds'] > start) or
                   (start == end and turn['start_seconds'] <= start < turn['end_seconds'])]
        speakers = {turn['speaker'] for turn in overlap}
        covered = start
        for turn in sorted(overlap, key=lambda item: item['start_seconds']):
            if turn['start_seconds'] > covered + 1e-9:
                break
            covered = max(covered, turn['end_seconds'])
        speaker = next(iter(speakers)) if len(speakers) == 1 and None not in speakers and covered >= end and overlap else 'UNKNOWN'
        row = dict(text=word['text'], start=round(start*1000, 3), end=round(end*1000, 3))
        if utterances and utterances[-1]['speaker'] == speaker and start*1000-utterances[-1]['end'] <= 500:
            utterances[-1]['text'] += ' '+word['text']
            utterances[-1]['end'] = max(utterances[-1]['end'], row['end'])
            utterances[-1]['words'].append(row)
        else:
            utterances.append(dict(speaker=speaker, text=word['text'], start=row['start'], end=row['end'], words=[row]))
    return utterances


def _decode(source, destination, timeout):
    if destination.exists():
        raise RecordingProcessingError('uncheckpointed decoded artifact requires reconciliation')
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.wav')
    os.close(fd)
    try:
        result = subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
            '-c:a', 'pcm_s16le', '-bitexact', '-f', 'wav', temporary,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise RecordingProcessingError('CPU media decode failed')
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish(output, key, source, config, old):
    from docx import Document

    root = _private_directory(Path(config['output_directory']).expanduser().resolve())
    generation = root/('recording-'+key)
    generation.mkdir(mode=0o700)  # Never replace an existing generation.
    paths = {kind: str(generation/('transcript.'+kind)) for kind in ('docx', 'txt', 'json')}
    utterances = attribute_words(output['words'], output['turns'])
    duration = max(turn['end_seconds'] for turn in output['turns']) if output['turns'] else 0
    duration = max(duration, max((word['end_seconds'] for word in output['words']), default=0))
    # The complete WAV duration includes leading/trailing silence and gaps.
    with wave.open(output['source'], 'rb') as audio:
        duration = audio.getnframes()/audio.getframerate()
    end = old.get('recording_end') or datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()
    start = old.get('recording_start') or (datetime.fromisoformat(end)-timedelta(seconds=duration)).isoformat()
    artifact = dict(schema_version=2, recording_id=output['recording_id'], conversation_id=output['conversation_id'],
                    source_media_path=str(source), working_media_path=output['source'],
                    backend='managed-recording', duration_seconds=duration,
                    recording_start=start, recording_end=end,
                    transcript_window_start_seconds=0, transcript_window_end_seconds=duration,
                    transcript_title=old.get('transcript_title') or source.stem,
                    event=old.get('event'), utterance_count=len(utterances), utterances=utterances,
                    transcript_text='\n'.join(row['text'] for row in utterances), output_paths=paths,
                    prior_artifacts=output['prior_artifacts'], managed_recording=output)
    document = Document()
    document.add_heading(artifact['transcript_title'], 0)
    document.add_paragraph('Recording: '+artifact['recording_id'])
    document.add_paragraph('Conversation: '+artifact['conversation_id'])
    document.add_paragraph('Source: '+str(source))
    if artifact['event']:
        document.add_paragraph('Calendar: '+_json(artifact['event']))
    lines = []
    for row in utterances:
        line = f"{row['speaker']} [{row['start']/1000:.3f}s - {row['end']/1000:.3f}s]: {row['text']}"
        lines.append(line)
        document.add_paragraph(line)
    document.save(paths['docx'])
    Path(paths['txt']).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    Path(paths['json']).write_text(_json(artifact)+'\n', encoding='utf-8')
    for path in paths.values():
        os.chmod(path, 0o400)
        with open(path, 'rb') as stream:
            os.fsync(stream.fileno())
    _sync_directory(generation)
    _sync_directory(root)
    receipt = dict(generation=key, recording_id=output['recording_id'], conversation_id=output['conversation_id'],
                   output_paths=paths, sha256={kind: _hash_file(Path(path)) for kind, path in paths.items()})
    if config.get('transcript_store'):
        from transcript_store import ingest_artifact
        store = config['transcript_store']
        receipt['store'] = ingest_artifact(Path(paths['json']), root=Path(store['root']),
                                          embedding_provider=store['embedding_provider'],
                                          embedding_model=store.get('embedding_model', 'debug-hash')).to_dict()
    return receipt


def run(source, config, *, recording_id=None, conversation_id=None, existing_transcript=None):
    """Execute explicit media through actual ManagedClient; return guarded receipt."""
    from inference_manager.service_client import ManagedClient

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError('explicit media file required')
    old_raw = _private_read(Path(existing_transcript).expanduser()) if existing_transcript else b'{}'
    old = json.loads(old_raw)
    if not isinstance(old, dict):
        raise ValueError('existing transcript object required')
    recording_id = recording_id or old.get('recording_id')
    conversation_id = conversation_id or old.get('conversation_id')
    for name, value in (('recording_id', recording_id), ('conversation_id', conversation_id)):
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError('explicit bounded '+name+' required')
        if old.get(name) and old[name] != value:
            raise ValueError('existing transcript '+name+' conflict')
    wait = _bounded(config, 'wait_seconds', 300, .01, 86400)
    poll = _bounded(config, 'poll_seconds', .25, .01, 60)
    decode_timeout = _bounded(config, 'decode_timeout_seconds', 300, 1, 3600)
    attempts = config.get('max_stage_attempts', 1)
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError('max_stage_attempts must be 1..10')
    store = config.get('transcript_store')
    if store and (not store.get('root') or store.get('embedding_provider') not in ('hash', 'debug-hash')):
        raise ValueError('optional store requires explicit CPU debug-hash embedding provider')
    token = _private_read(Path(config['token_file']).expanduser(), 4096).decode().strip()
    tls = ssl.create_default_context(cafile=config.get('ca_file'))
    client = ManagedClient(config['endpoint'], token, ssl_context=tls,
                           timeout_seconds=config.get('request_timeout_seconds', 10),
                           allow_insecure_loopback=config.get('allow_insecure_loopback', False))
    processor = RecordingProcessor(config['processor'], client=client)
    manifests = _private_directory(processor.state/'cli')
    decoded = _private_directory(manifests/'media')
    source_hash = _hash_file(source)
    identity = dict(source=str(source), sha256=source_hash, recording_id=recording_id,
                    conversation_id=conversation_id)
    key = hashlib.sha256(_json(identity).encode()).hexdigest()
    frozen = dict(identity=identity, config=config, existing_transcript=old,
                  existing_transcript_sha256=hashlib.sha256(old_raw).hexdigest())
    manifest_path = manifests/(key+'.json')
    fd = os.open(manifests/(key+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RecordingProcessingError('recording CLI already owned') from None
        if manifest_path.exists():
            manifest = json.loads(_private_read(manifest_path))
            if manifest['frozen'] != frozen:
                raise RecordingProcessingError('immutable CLI configuration/metadata conflict')
        else:
            manifest = dict(frozen=frozen)
            _save(manifest_path, manifest)
        derivative = decoded/(key+'.wav')
        if 'decoded_sha256' not in manifest:
            _decode(source, derivative, decode_timeout)
            if _hash_file(source) != source_hash:
                raise RecordingProcessingError('original source changed during CPU decode')
            manifest['decoded_sha256'] = _hash_file(derivative)
            _save(manifest_path, manifest)
        if derivative.is_symlink() or _hash_file(derivative) != manifest['decoded_sha256']:
            raise RecordingProcessingError('immutable decoded media conflict')
        processor_key = processor.prepare(
            derivative, recording_id=recording_id, conversation_id=conversation_id,
            metadata=dict(original_source=identity, existing_transcript=old),
            options=config.get('options', {}),
            prior_artifacts=list((old.get('output_paths') or {}).values()),
        )
        if manifest.get('processor_key', processor_key) != processor_key:
            raise RecordingProcessingError('processor generation conflict')
        manifest['processor_key'] = processor_key
        _save(manifest_path, manifest)
        deadline = time.monotonic()+wait
        while time.monotonic() < deadline:
            try:
                checkpoint = processor.advance(processor_key)
            except RecordingProcessingError:
                checkpoint = processor.snapshot(processor_key)
                terminal = [(index, stage, entry) for index, unit in enumerate(checkpoint['units'])
                            for stage, entry in unit['stages'].items()
                            if entry['state'] in ('cancelled', 'failed') and 'result' not in entry]
                if terminal:
                    for index, stage, entry in terminal:
                        if entry['attempt']+1 >= attempts:
                            raise RecordingProcessingError('bounded stage attempts exhausted')
                        processor.retry_stage(processor_key, index, stage)
                elif not any(entry['state'] in ('unknown', 'submitting', 'pending')
                             for unit in checkpoint['units'] for entry in unit['stages'].values()):
                    raise
                time.sleep(min(poll, max(0, deadline-time.monotonic())))
                continue
            if checkpoint['state'] == 'ready':
                if _hash_file(source) != source_hash:
                    raise RecordingProcessingError('original source changed before publication')
                return processor.publish(processor_key, lambda output: _publish(output, processor_key, source, config, old))
            time.sleep(min(poll, max(0, deadline-time.monotonic())))
        raise RecordingProcessingError('bounded wait expired; checkpoint retained without replay')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording-id')
    parser.add_argument('--conversation-id')
    parser.add_argument('--existing-transcript', type=Path)
    args = parser.parse_args(argv)
    try:
        config = json.loads(_private_read(args.config))
        receipt = run(args.media, config, recording_id=args.recording_id,
                      conversation_id=args.conversation_id, existing_transcript=args.existing_transcript)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print('Managed recording failed: '+str(error), file=sys.stderr)
        return 1
    print(_json(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
