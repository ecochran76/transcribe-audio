#!/usr/bin/env python3
"""Explicit CPU media preparation and managed diarized recording publication."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import errno
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
        offset = 0
        while True:
            for attempt in range(3):
                try:
                    stream.seek(offset)
                    block = stream.read(1024*1024)
                    break
                except OSError as error:
                    # A transient filesystem ENOMEM is not a GPU retry. Reset to
                    # the last hashed offset: failed buffered reads can advance.
                    if error.errno != errno.ENOMEM or attempt == 2:
                        raise
            if not block:
                break
            digest.update(block)
            offset += len(block)
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


def _decode(source, destination, timeout, *, maximum_bytes=None):
    if destination.exists():
        raise RecordingProcessingError('uncheckpointed decoded artifact requires reconciliation')
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.wav')
    os.close(fd)
    try:
        command = [
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
            '-c:a', 'pcm_s16le', '-bitexact', '-f', 'wav',
        ]
        if maximum_bytes is not None:
            # prlimit execs ffmpeg with a kernel hard bound; -fs alone may overshoot.
            command = ['prlimit', '--fsize=' + str(maximum_bytes) + ':' + str(maximum_bytes), '--',
                       *command, '-fs', str(maximum_bytes - 65536)]
        result = subprocess.run([*command, temporary], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode or (maximum_bytes is not None
                                 and Path(temporary).stat().st_size >= maximum_bytes - 65536):
            raise RecordingProcessingError('CPU media decode failed')
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _decoded_byte_bound(source, timeout):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                            '-show_entries', 'stream=duration:format=duration', '-of', 'json', str(source)],
                           capture_output=True, text=True, timeout=min(timeout, 30), check=True)
    result = json.loads(probe.stdout)
    values = [item.get('duration') for item in result.get('streams', [])]
    values.append(result.get('format', {}).get('duration'))
    durations = []
    for value in values:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        raise RecordingProcessingError('bounded audio duration required for storage admission')
    # PCM payload plus bounded container/decoder padding. Hitting either output
    # limit is a failed decode, never an accepted truncated recording.
    return math.ceil(max(durations) * 32000) + 2 * 1024 * 1024


def _publish(output, key, source, config, old, *, storage_root=None, logical_source=None):
    from storage_access import resolve_media
    from docx import Document

    root = _private_directory(Path(config['output_directory']).expanduser().resolve())
    generation = root/('recording-'+key)
    generation.mkdir(mode=0o700)  # Never replace an existing generation.
    paths = {kind: str(generation/('transcript.'+kind)) for kind in ('docx', 'txt', 'json')}
    utterances = attribute_words(output['words'], output['turns'])
    duration = max(turn['end_seconds'] for turn in output['turns']) if output['turns'] else 0
    duration = max(duration, max((word['end_seconds'] for word in output['words']), default=0))
    # The complete WAV duration includes leading/trailing silence and gaps.
    with wave.open(str(resolve_media(storage_root, output['source'])), 'rb') as audio:
        duration = audio.getnframes()/audio.getframerate()
    source_mtime = source.stat().st_mtime
    from storage_access import media_record, catalog_for
    placement = media_record(storage_root, source)
    if placement and str(source) != placement['logical_path']:
        migration = catalog_for(storage_root).migration(placement['id'])
        if migration:
            source_mtime = json.loads(migration['source_identity'])[3] / 1_000_000_000
    end = old.get('recording_end') or datetime.fromtimestamp(source_mtime, timezone.utc).isoformat()
    start = old.get('recording_start') or (datetime.fromisoformat(end)-timedelta(seconds=duration)).isoformat()
    artifact = dict(schema_version=2, recording_id=output['recording_id'], conversation_id=output['conversation_id'],
                    source_media_path=str(logical_source or source), working_media_path=output['source'],
                    backend='managed-recording', duration_seconds=duration,
                    recording_start=start, recording_end=end,
                    transcript_window_start_seconds=0, transcript_window_end_seconds=duration,
                    transcript_title=old.get('transcript_title') or (logical_source or source).stem,
                    event=old.get('event'), utterance_count=len(utterances), utterances=utterances,
                    transcript_text='\n'.join(row['text'] for row in utterances), output_paths=paths,
                    prior_artifacts=output['prior_artifacts'], managed_recording=output)
    document = Document()
    document.add_heading(artifact['transcript_title'], 0)
    document.add_paragraph('Recording: '+artifact['recording_id'])
    document.add_paragraph('Conversation: '+artifact['conversation_id'])
    document.add_paragraph('Source: '+str(logical_source or source))
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


def run(source, config, *, recording_id=None, conversation_id=None, existing_transcript=None,
        max_stage_results=None, storage_root=None):
    """Keep storage scope outside frozen configuration and checkpoint identity."""
    from recording_source_guard import media_access
    with media_access(storage_root):
        return _run(source, config, recording_id=recording_id,
                    conversation_id=conversation_id, existing_transcript=existing_transcript,
                    max_stage_results=max_stage_results, storage_root=storage_root)


def _run(source, config, *, recording_id=None, conversation_id=None, existing_transcript=None,
         max_stage_results=None, storage_root=None):
    """Execute explicit media through actual ManagedClient; return guarded receipt."""
    if max_stage_results is not None and (type(max_stage_results) is not int or not 1 <= max_stage_results <= 10000):
        raise ValueError('max_stage_results must be an integer1..10000')
    from inference_manager.service_client import ManagedClient

    from storage_access import resolve_media, logical_media_path
    logical_source = logical_media_path(storage_root, source)
    source = resolve_media(storage_root, source)
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
    processor = RecordingProcessor(config['processor'], client=client, storage_root=storage_root)
    manifests = _private_directory(processor.state/'cli')
    decoded = _private_directory(manifests/'media')
    source_hash = _hash_file(source)
    identity = dict(source=str(logical_source), sha256=source_hash, recording_id=recording_id,
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
            from storage_admission import admit_media_write, storage_policy_path
            if storage_policy_path(storage_root) is None:
                _decode(source, derivative, decode_timeout)
            else:
                bound = _decoded_byte_bound(source, decode_timeout)
                with admit_media_write(storage_root, 'decode-' + key, bound,
                                       transient_path=derivative) as lease:
                    lease.checkpoint(write_bytes=bound)
                    _decode(source, derivative, decode_timeout, maximum_bytes=lease.limit)
            if _hash_file(source) != source_hash:
                raise RecordingProcessingError('original source changed during CPU decode')
            manifest['decoded_sha256'] = _hash_file(derivative)
            _save(manifest_path, manifest)
        derivative_reference = derivative
        derivative = resolve_media(storage_root, derivative)
        if derivative.is_symlink() or _hash_file(derivative) != manifest['decoded_sha256']:
            raise RecordingProcessingError('immutable decoded media conflict')
        processor_key = processor.prepare(
            derivative_reference, recording_id=recording_id, conversation_id=conversation_id,
            metadata=dict(original_source=identity, existing_transcript=old),
            options=config.get('options', {}),
            prior_artifacts=list((old.get('output_paths') or {}).values()),
        )
        if manifest.get('processor_key', processor_key) != processor_key:
            raise RecordingProcessingError('processor generation conflict')
        manifest['processor_key'] = processor_key
        _save(manifest_path, manifest)
        def durable_results(checkpoint):
            return [dict(unit_index=index, stage=stage, request_id=entry['request']['id'],
                         attempt=entry['attempt'])
                    for index, unit in enumerate(checkpoint['units'])
                    for stage, entry in unit['stages'].items() if 'result' in entry]
        initial_results = len(durable_results(processor.snapshot(processor_key)))
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
            completed = durable_results(checkpoint)
            new_results = len(completed)-initial_results
            all_complete = all('result' in entry for unit in checkpoint['units']
                               for entry in unit['stages'].values())
            if max_stage_results is not None and new_results >= max_stage_results:
                if all_complete and checkpoint['state'] != 'ready':
                    # advance now only reconciles durable results; no new work.
                    checkpoint = processor.advance(processor_key)
                elif not all_complete:
                    return dict(state='partial', published=False, resumable=True,
                                reason='max_stage_results', recording_id=recording_id,
                                conversation_id=conversation_id, processor_key=processor_key,
                                checkpoint_path=str(processor.state/(processor_key+'.json')),
                                cli_manifest_path=str(manifest_path), new_stage_results=new_results,
                                completed_stage_results=completed)
            if checkpoint['state'] == 'ready':
                if _hash_file(source) != source_hash:
                    raise RecordingProcessingError('original source changed before publication')
                return processor.publish(processor_key, lambda output: _publish(output, processor_key, source, config, old, storage_root=storage_root, logical_source=logical_source))
            time.sleep(min(poll, max(0, deadline-time.monotonic())))
        raise RecordingProcessingError('bounded wait expired; checkpoint retained without replay')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--recording-id')
    parser.add_argument('--conversation-id')
    parser.add_argument('--existing-transcript', type=Path)
    parser.add_argument('--storage-root', type=Path,
                        default=os.environ.get('TRANSCRIBE_STORAGE_ROOT') or None,
                        help='Existing catalog root; defaults to TRANSCRIBE_STORAGE_ROOT outside frozen configuration')
    parser.add_argument('--max-stage-results', type=int, help='Checkpoint after1..10000 new durable unit-stage results')
    args = parser.parse_args(argv)
    try:
        config = json.loads(_private_read(args.config))
        receipt = run(args.media, config, recording_id=args.recording_id,
                      conversation_id=args.conversation_id, existing_transcript=args.existing_transcript,
                      max_stage_results=args.max_stage_results, storage_root=args.storage_root)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print('Managed recording failed: '+str(error), file=sys.stderr)
        return 1
    print(_json(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
