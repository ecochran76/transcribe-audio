import wave

import pytest

from managed_recording import RecordingProcessor, RecordingProcessingError


def audio(path):
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 16000, 0, 'NONE', ''))
        out.writeframes(b'\x00\x00' * 32000)
    return path


def config(tmp_path):
    return dict(state_directory=str(tmp_path/'state'), spool_root=str(tmp_path/'spool'),
                unit_seconds=1, profiles={
                    'asr': dict(profile='asr', identity='a'*64),
                    'diarization': dict(profile='diar', identity='b'*64)},
                speaker_similarity=.9, speaker_margin=.1)


class Client:
    def __init__(self):
        self.requests = []
        self.lost = True

    def submit(self, request):
        self.requests.append(request)
        if self.lost:
            self.lost = False
            raise TimeoutError()
        return dict(state='succeeded')

    def observe(self, request_id):
        return dict(state='succeeded')

    def result(self, request_id):
        request = next(r for r in self.requests if r['id'] == request_id)
        stage = 'asr' if request['profile'] == 'asr' else 'diarization'
        result = dict(stage=stage, duration_seconds=1,
                      model=dict(identity=request['identity']))
        if stage == 'asr':
            result['words'] = [dict(text='hello', start_seconds=0, end_seconds=.5)]
        else:
            result.update(turns=[dict(speaker='0', start_seconds=0, end_seconds=1)],
                          speaker_vectors={'0': [1, 0]})
        return dict(state='succeeded', result=result)


def test_restart_observes_lost_ack_and_publishes_one_generation(tmp_path):
    client = Client()
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r',
                            conversation_id='c', metadata={'calendar': 'original'},
                            prior_artifacts=['old.docx'])
    with pytest.raises(RecordingProcessingError):
        processor.advance(key)
    processor = RecordingProcessor(config(tmp_path), client=client)
    for _ in range(5):
        snapshot = processor.advance(key)
    assert snapshot['state'] == 'ready'
    assert len(client.requests) == 4
    assert snapshot['output']['words'][1]['start_seconds'] == 1
    assert {t['speaker'] for t in snapshot['output']['turns']} == {'speaker-1'}
    calls = []
    processor.publish(key, lambda output: calls.append(output) or {'artifact': 'new.docx'})
    processor.publish(key, lambda output: pytest.fail('duplicate publication'))
    assert calls[0]['metadata'] == {'calendar': 'original'}
    assert calls[0]['prior_artifacts'] == ['old.docx']


def test_uncertain_publication_never_replays_after_restart(tmp_path):
    client = Client()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    for _ in range(5):
        processor.advance(key)
    def lost(output):
        raise TimeoutError('artifact committed but ack lost')
    with pytest.raises(TimeoutError):
        processor.publish(key, lost)
    resumed = RecordingProcessor(config(tmp_path), client=client)
    with pytest.raises(RecordingProcessingError, match='uncertain'):
        resumed.publish(key, lambda output: pytest.fail('duplicate'))


def test_profile_conflict_cannot_complete_stage(tmp_path):
    class Conflict(Client):
        def result(self, request_id):
            response = super().result(request_id)
            response['result']['model']['identity'] = 'c'*64
            return response
    client = Conflict()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    with pytest.raises(RecordingProcessingError, match='identity conflict'):
        processor.advance(key)
    assert 'result' not in processor.snapshot(key)['units'][0]['stages']['asr']


def test_different_vector_dimensions_are_explicitly_unresolved(tmp_path):
    class Conflict(Client):
        def result(self, request_id):
            response = super().result(request_id)
            if len(self.requests) == 4:
                response['result']['speaker_vectors'] = {'0': [1, 0, 0]}
            return response
    client = Conflict()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    for _ in range(5):
        result = processor.advance(key)
    assert result['output']['turns'][1]['speaker'] is None
    assert len(result['output']['unresolved']) == 1


def test_same_unit_competing_labels_remain_ambiguous(tmp_path):
    class Conflict(Client):
        def result(self, request_id):
            response = super().result(request_id)
            result = response['result']
            if result['stage'] == 'diarization':
                result['turns'].append(dict(speaker='1', start_seconds=.5, end_seconds=1))
                result['speaker_vectors']['1'] = [1, 0]
            return response
    client = Conflict()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    for _ in range(5):
        result = processor.advance(key)
    assert result['output']['turns'][1]['speaker'] is None


def test_retry_requires_observed_cancellation_and_preserves_lineage(tmp_path):
    client = Client()
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    with pytest.raises(RecordingProcessingError):
        processor.advance(key)
    with pytest.raises(RecordingProcessingError, match='confirmed'):
        processor.retry_stage(key, 0, 'asr')
    old_id = client.requests[0]['id']
    client.observe = lambda request_id: dict(state='cancelled')
    record = processor.retry_stage(key, 0, 'asr')
    stage = record['units'][0]['stages']['asr']
    assert stage['request']['id'] != old_id
    assert stage['history'][0]['request']['id'] == old_id
    processor.advance(key)
    assert len(client.requests) == 2


def test_restarted_processor_rejects_configuration_drift(tmp_path):
    client = Client()
    original = config(tmp_path)
    processor = RecordingProcessor(original, client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    changed = dict(original, unit_seconds=2)
    resumed = RecordingProcessor(changed, client=client)
    with pytest.raises(RecordingProcessingError, match='configuration conflict'):
        resumed.advance(key)
    assert client.requests == []


@pytest.mark.parametrize('field', ['model', 'stage'])
def test_missing_stage_provenance_never_accepts_result(tmp_path, field):
    class Missing(Client):
        def result(self, request_id):
            response = super().result(request_id)
            del response['result'][field]
            return response
    client = Missing()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    with pytest.raises(RecordingProcessingError, match='identity conflict'):
        processor.advance(key)
    assert 'result' not in processor.snapshot(key)['units'][0]['stages']['asr']


def test_restart_finishes_asr_cohort_before_loading_diarization(tmp_path):
    client = Client()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    first = processor.advance(key)
    first_asr = first['units'][0]['stages']['asr'].copy()
    processor = RecordingProcessor(config(tmp_path), client=client)
    for _ in range(4):
        final = processor.advance(key)
    assert [request['profile'] for request in client.requests] == ['asr', 'asr', 'diar', 'diar']
    assert final['units'][0]['stages']['asr'] == first_asr
    assert final['state'] == 'ready'


def test_explicit_retention_rejection_waits_with_same_request_id(tmp_path):
    class RetentionFull(Exception):
        code='retention_full'
        status=503
    class RejectedOnce(Client):
        def __init__(self):
            super().__init__();self.lost=False;self.rejected=False
        def submit(self,request):
            if not self.rejected:
                self.rejected=True;self.requests.append(request)
                raise RetentionFull()
            return super().submit(request)
    client=RejectedOnce();processor=RecordingProcessor(config(tmp_path),client=client)
    key=processor.prepare(audio(tmp_path/'source.wav'),recording_id='r',conversation_id='c')
    with pytest.raises(RecordingProcessingError):processor.advance(key)
    entry=processor.snapshot(key)['units'][0]['stages']['asr']
    assert entry['state']=='pending'
    snapshot=processor.advance(key)
    assert snapshot['units'][0]['stages']['asr']['state']=='succeeded'
    assert len(client.requests)==2
    assert client.requests[0]['id']==client.requests[1]['id']
    assert snapshot['units'][0]['stages']['asr']['attempt']==0


def test_explicit_concurrency_is_frozen_in_managed_submission(tmp_path):
    settings = config(tmp_path)
    settings['max_workers'] = 2
    client = Client()
    client.lost = False
    processor = RecordingProcessor(settings, client=client)
    key = processor.prepare(audio(tmp_path/'input.wav'), recording_id='r', conversation_id='c')
    processor.advance(key)
    assert client.requests[0].get('max_workers') == 2
    settings['max_workers'] = 1
    with pytest.raises(RecordingProcessingError, match='configuration conflict'):
        RecordingProcessor(settings, client=client).snapshot(key)


def test_explicit_profile_repair_preserves_results_and_blocks_uncertain_retry(tmp_path):
    client = Client()
    client.lost = False
    processor = RecordingProcessor(config(tmp_path), client=client)
    key = processor.prepare(audio(tmp_path/'source.wav'), recording_id='r', conversation_id='c')
    processor.advance(key)  # One durable ASR result must survive the repair.
    before = processor.snapshot(key)
    replacement = dict(profile='diar-fixed', identity='c'*64)
    client.observe = lambda request_id: dict(state='unknown')
    with pytest.raises(RecordingProcessingError, match='confirmed'):
        processor.retry_stage(key, 0, 'diarization', replacement_profile=replacement)
    assert processor.snapshot(key) == before
    client.observe = lambda request_id: dict(state='failed')
    repaired = processor.retry_stage(key, 0, 'diarization', replacement_profile=replacement)
    entry = repaired['units'][0]['stages']['diarization']
    assert repaired['units'][0]['stages']['asr'] == before['units'][0]['stages']['asr']
    assert entry['history'][-1]['request'] == before['units'][0]['stages']['diarization']['request']
    assert entry['request']['profile'] == replacement['profile']
    assert entry['request']['identity'] == replacement['identity']
    assert entry['request']['payload'] == before['units'][0]['stages']['diarization']['request']['payload']
    assert entry['request']['id'] != entry['history'][-1]['request']['id']
    resumed = RecordingProcessor(config(tmp_path), client=client)
    assert resumed.snapshot(key)['units'][0]['stages']['diarization'] == entry
