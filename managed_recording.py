"""Durable, CPU-only recording units over an injected managed service client."""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import wave

from recording_source_guard import guard_media
from storage_access import resolve_media, logical_media_path


class RecordingProcessingError(RuntimeError):
    """Work remains pending, uncertain, or requires explicit reconciliation."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


class RecordingProcessor:
    """One durable recording generation; ``advance`` performs at most one stage.

    Config requires private state/spool paths, unit_seconds, profiles keyed by
    asr/diarization (profile and pinned identity), speaker_similarity and
    speaker_margin. Thresholds must be reviewed by the caller, not inferred here.
    """

    def __init__(self, config, *, client, storage_root=None):
        self.source_store_root = storage_root
        self.config = copy.deepcopy(config)
        self.client = client
        self.state = Path(config['state_directory'])
        self.spool = Path(config['spool_root'])
        for path in (self.state, self.spool):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if path.is_symlink() or path.stat().st_mode & 0o077:
                raise ValueError('private managed directory required')
        if 'max_workers' in config and (type(config['max_workers']) is not int or not 1 <= config['max_workers'] <= 64):
            raise ValueError('bounded requested concurrency required')
        if set(config['profiles']) != {'asr', 'diarization'}:
            raise ValueError('exactly ASR and diarization profiles required')
        for stage in ('asr', 'diarization'):
            profile = config['profiles'][stage]
            if not profile['profile'] or not re.fullmatch('[a-f0-9]{64}', profile['identity']):
                raise ValueError('explicit pinned stage profile required')
        for name in ('unit_seconds', 'speaker_similarity', 'speaker_margin'):
            value = config[name]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite processing controls required')
        if not 0 < config['unit_seconds'] <= 3600:
            raise ValueError('unit_seconds must be in (0, 3600]')
        if not -1 <= config['speaker_similarity'] <= 1 or not 0 < config['speaker_margin'] <= 2:
            raise ValueError('reviewed cosine threshold and positive margin required')

    @contextlib.contextmanager
    def _locked(self, key):
        if not re.fullmatch('[a-f0-9]{64}', key):
            raise ValueError('invalid recording generation key')
        fd = os.open(self.state / (key+'.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RecordingProcessingError('recording already owned') from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _save(self, key, record):
        fd, name = tempfile.mkstemp(dir=self.state, prefix='.recording-')
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(_json(record))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.state/(key+'.json'))
            fd = os.open(self.state, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _load(self, key):
        with (self.state/(key+'.json')).open() as stream:
            record = json.load(stream)
        if record['identity']['config'] != self.config:
            raise RecordingProcessingError('checkpoint profile/options configuration conflict')
        return record

    def snapshot(self, key):
        """Return a detached checkpoint, including raw completed stage evidence."""
        with self._locked(key):
            return self._load(key)

    @guard_media
    def prepare(self, source, *, recording_id, conversation_id, metadata=None,
                options=None, prior_artifacts=None):
        """Stage bounded immutable PCM16 mono16k WAV units without model effects."""
        logical_source = logical_media_path(self.source_store_root, source)
        source = resolve_media(self.source_store_root, source)
        source_hash = hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                source_hash.update(block)
        identity = dict(source=str(logical_source), sha256=source_hash.hexdigest(),
                        recording_id=recording_id, conversation_id=conversation_id,
                        metadata=metadata or {}, options=options or {},
                        prior_artifacts=prior_artifacts or [], config=self.config)
        key = _digest(_json(identity).encode())
        with self._locked(key):
            if (self.state/(key+'.json')).exists():
                return key
            units = []
            with wave.open(str(source), 'rb') as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, 'NONE'):
                    raise ValueError('PCM16 mono16k WAV input required')
                frames_per_unit = int(self.config['unit_seconds']*16000)
                if frames_per_unit < 1:
                    raise ValueError('unit must include a frame')
                from storage_admission import admit_media_write
                frame_count = audio.getnframes()
                if frame_count < 1:
                    raise ValueError('empty recording')
                maximum_bytes = 2 * frame_count + 44 * ((frame_count + frames_per_unit - 1) // frames_per_unit)
                with admit_media_write(self.source_store_root, 'stage-' + key, maximum_bytes) as lease:
                    offset = 0
                    while True:
                        frames = audio.readframes(frames_per_unit)
                        if not frames:
                            break
                        data = io.BytesIO()
                        with wave.open(data, 'wb') as output:
                            output.setparams((1, 2, 16000, 0, 'NONE', ''))
                            output.writeframes(frames)
                        content = data.getvalue()
                        sha = _digest(content)
                        artifact = sha+'.wav'
                        path = self.spool/artifact
                        try:
                            if lease:
                                lease.checkpoint(write_bytes=len(content))
                            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
                        except FileExistsError:
                            if path.is_symlink() or _digest(path.read_bytes()) != sha:
                                raise RecordingProcessingError('staged artifact conflict')
                        else:
                            with os.fdopen(fd, 'wb') as out:
                                out.write(content)
                                out.flush()
                                os.fsync(out.fileno())
                        unit = dict(offset_seconds=offset/16000, duration_seconds=len(frames)/32000,
                                    artifact_id=artifact, sha256=sha, stages={})
                        for stage, profile in self.config['profiles'].items():
                            request_id = stage+'-'+_digest(_json([key, len(units), stage, 0]).encode())
                            request = dict(id=request_id, **profile, workload_class='historical',
                                           payload=dict(artifact_id=artifact, sha256=sha,
                                                        options=identity['options']))
                            if 'max_workers' in self.config:
                                request['max_workers'] = self.config['max_workers']
                            unit['stages'][stage] = dict(state='pending', attempt=0, request=request, history=[])
                        units.append(unit)
                        offset += len(frames)//2
            verification = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    verification.update(block)
            if verification.hexdigest() != identity['sha256']:
                raise RecordingProcessingError('source changed during staging')
            if not units:
                raise ValueError('empty recording')
            self._save(key, dict(schema_version=1, key=key, identity=identity,
                                 state='processing', units=units))
        return key

    @guard_media
    def advance(self, key):
        """Submit once or observe one unfinished stage; never replay unknown custody."""
        with self._locked(key):
            record = self._load(key)
            # Finish each model cohort before changing residency. Checkpoints and
            # uncertain requests retain their original IDs and observation rules.
            for stage in ('asr', 'diarization'):
                for unit in record['units']:
                    entry = unit['stages'][stage]
                    if 'result' in entry:
                        continue
                    if entry['state'] in ('failed', 'cancelled'):
                        raise RecordingProcessingError('explicit new attempt required')
                    artifact = resolve_media(self.source_store_root, self.spool/unit['artifact_id'])
                    if artifact.is_symlink() or _digest(artifact.read_bytes()) != unit['sha256']:
                        raise RecordingProcessingError('staged artifact changed')
                    if entry['state'] == 'pending':
                        entry['state'] = 'submitting'
                        self._save(key, record)
                        try:
                            view = self.client.submit(entry['request'])
                        except Exception as error:
                            known_rejection = (getattr(error, 'code', None), getattr(error, 'status', None)) in {('queue_full', 429), ('retention_full', 503)}
                            entry['state'] = 'pending' if known_rejection else 'unknown'
                            self._save(key, record)
                            raise RecordingProcessingError('submission retained; retry only known rejection') from None
                    else:
                        try:
                            view = self.client.observe(entry['request']['id'])
                        except Exception:
                            raise RecordingProcessingError('observation unavailable; no replay') from None
                    entry['state'] = view['state']
                    self._save(key, record)
                    if view['state'] == 'succeeded':
                        response = self.client.result(entry['request']['id'])
                        if response['state'] != 'succeeded':
                            raise RecordingProcessingError('result not available')
                        self._validate(response['result'], stage, unit, entry['request']['identity'])
                        entry['result'] = response['result']
                        self._save(key, record)
                    return record
            if 'output' not in record:
                record['output'] = self._reconcile(record)
                record['state'] = 'ready'
                self._save(key, record)
            return record

    def _validate(self, result, stage, unit, fingerprint):
        if result.get('stage') != stage or result.get('model', {}).get('identity') != fingerprint:
            raise RecordingProcessingError('stage model identity conflict')
        duration = result['duration_seconds']
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or abs(duration-unit['duration_seconds']) > .001:
            raise RecordingProcessingError('stage duration conflict')
        for row in result['words' if stage == 'asr' else 'turns']:
            start, end = row['start_seconds'], row['end_seconds']
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not 0 <= start <= end <= duration:
                raise RecordingProcessingError('invalid unit timestamps')
        for vector in result.get('speaker_vectors', {}).values():
            if not vector or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or sum(v*v for v in vector) == 0:
                raise RecordingProcessingError('invalid speaker evidence')

    def retry_stage(self, key, unit_index, stage, *, replacement_profile=None):
        """Retry a confirmed terminal attempt, optionally on an explicit pinned repair.

        Completed results and original frozen configuration remain unchanged.
        The attempt history retains the original request/profile; only this new
        attempt uses the replacement. Unknown outcomes never permit replacement.
        """
        if replacement_profile is not None:
            if (not isinstance(replacement_profile, dict)
                    or set(replacement_profile) != {'profile', 'identity'}
                    or not isinstance(replacement_profile['profile'], str)
                    or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', replacement_profile['profile'])
                    or not isinstance(replacement_profile['identity'], str)
                    or not re.fullmatch(r'[a-f0-9]{64}', replacement_profile['identity'])):
                raise ValueError('explicit pinned replacement profile required')
        with self._locked(key):
            record = self._load(key)
            entry = record['units'][unit_index]['stages'][stage]
            view = self.client.observe(entry['request']['id'])
            if view['state'] not in ('failed', 'cancelled') or 'result' in entry or 'output' in record:
                raise RecordingProcessingError('retry requires confirmed terminal outcome')
            entry['history'].append(dict(request=entry['request'], state=view['state']))
            entry['attempt'] += 1
            entry['request'] = copy.deepcopy(entry['request'])
            entry['request']['id'] = stage+'-'+_digest(_json([key, unit_index, stage, entry['attempt']]).encode())
            if replacement_profile is not None:
                entry['request'].update(copy.deepcopy(replacement_profile))
            entry['state'] = 'pending'
            self._save(key, record)
            return record

    def _reconcile(self, record):
        output = copy.deepcopy(record['identity'])
        output.update(words=[], turns=[], unresolved=[])
        speakers = {}
        for index, unit in enumerate(record['units']):
            asr = unit['stages']['asr']['result']
            diar = unit['stages']['diarization']['result']
            mapping = {}
            for label in sorted({turn['speaker'] for turn in diar['turns']}):
                vector = diar.get('speaker_vectors', {}).get(label)
                candidates = []
                if vector:
                    for speaker, reference in speakers.items():
                        if len(vector) == len(reference):
                            score = sum(a*b for a, b in zip(vector, reference))/math.sqrt(sum(a*a for a in vector)*sum(b*b for b in reference))
                            candidates.append((score, speaker))
                candidates.sort(reverse=True)
                if vector and not speakers:
                    speaker = 'speaker-'+str(len(speakers)+1)
                    speakers[speaker] = vector
                elif candidates and candidates[0][0] >= self.config['speaker_similarity'] and (len(candidates) == 1 or candidates[0][0]-candidates[1][0] >= self.config['speaker_margin']):
                    speaker = candidates[0][1]
                elif vector and candidates and candidates[0][0] < self.config['speaker_similarity']:
                    speaker = 'speaker-'+str(len(speakers)+1)
                    speakers[speaker] = vector
                else:
                    speaker = None
                    output['unresolved'].append(dict(unit=index, local_speaker=label, reason='missing_or_ambiguous_evidence', candidates=candidates))
                if speaker is not None and speaker in mapping.values():
                    speaker = None
                    output['unresolved'].append(dict(unit=index, local_speaker=label,
                                                     reason='competing_local_labels', candidates=candidates))
                mapping[label] = speaker
            for rows, destination in ((asr['words'], 'words'), (diar['turns'], 'turns')):
                for row in rows:
                    item = dict(row, unit=index)
                    for field in ('start_seconds', 'end_seconds'):
                        item[field] += unit['offset_seconds']
                    if destination == 'turns':
                        item['local_speaker'] = row['speaker']
                        item['speaker'] = mapping[row['speaker']]
                    output[destination].append(item)
        return output

    def publish(self, key, callback):
        """Call callback(output) once; it must retain prior versions and return a receipt.

        The callback owns canonical DOCX/TXT/JSON/store transactions. A crash or
        exception is uncertain and blocks all replay, including after restart.
        """
        with self._locked(key):
            record = self._load(key)
            if record.get('publication_state') == 'complete':
                return record['publication_receipt']
            if record.get('publication_state') is not None:
                raise RecordingProcessingError('publication uncertain; explicit external reconciliation required')
            if record['state'] != 'ready':
                raise RecordingProcessingError('recording reconciliation incomplete')
            record['publication_state'] = 'publishing'
            self._save(key, record)
            receipt = callback(copy.deepcopy(record['output']))
            if not receipt:
                raise RecordingProcessingError('publication receipt required; outcome uncertain')
            _json(receipt)
            record.update(publication_state='complete', publication_receipt=receipt)
            self._save(key, record)
            return receipt
