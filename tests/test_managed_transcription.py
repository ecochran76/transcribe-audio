"""Managed transcription through client and output commit interfaces."""
import json
from pathlib import Path
import tempfile
import wave

from managed_transcription import ManagedTranscriber


class Client:
    def __init__(self): self.requests=[]; self.observations=[]
    def submit(self, request): self.requests.append(request);return {'id':request['id'],'state':'running'}
    def observe(self, job_id): self.observations.append(job_id);return {'id':job_id,'state':'succeeded'}
    def result(self, job_id):
        return {'id':job_id,'state':'succeeded','result':{'segments':[{'start':0.,'end':.1,'text':'Useful test.','words':[{'start':0.,'end':.1,'word':'Useful','probability':.9}]}],'duration':.1,'language':'en','language_probability':1.}}


def audio(path):
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(b'\0\0'*1600)


def config(root):
    return {'endpoint':'https://example.invalid','profile':'fw','identity':'a'*64,
        'spool_root':str(root/'spool'),'state_directory':str(root/'state'),
        'max_workers':2,'model':'large-v3','device':'cuda','compute_type':'float16','batched':True,
        'poll_seconds':.01,'wait_seconds':1}


def test_transcription_stages_and_preserves_controls_and_recovers_result(tmp_path):
    source=tmp_path/'input.wav';audio(source);client=Client()
    transcriber=ManagedTranscriber(config(tmp_path),client=client)
    segments,info=transcriber.transcribe(str(source),beam_size=5,batch_size=8,word_timestamps=False)
    segments=list(segments)
    assert segments[0].words[0].word=='Useful'
    assert info.duration==.1
    request=client.requests[0]
    artifact=tmp_path/'spool'/request['payload']['artifact_id']
    assert artifact.stat().st_mode&0o777==0o600
    assert request['payload']['options']=={'beam_size':5,'batch_size':8,'word_timestamps':False}
    assert '/' not in request['payload']['artifact_id']
    recovered=ManagedTranscriber(config(tmp_path),client=client)
    assert list(recovered.transcribe(str(source),beam_size=5,batch_size=8,word_timestamps=False)[0])[0].text=='Useful test.'
    assert len(client.requests)==1


def test_unknown_submission_is_observed_without_replay_and_exports_once(tmp_path):
    import pytest
    from managed_transcription import ManagedTranscriptionError
    source=tmp_path/'input.wav';audio(source);client=Client()
    original=client.submit
    def uncertain(request):
        original(request)
        raise OSError('network lost after acceptance')
    client.submit=uncertain
    first=ManagedTranscriber(config(tmp_path),client=client)
    with pytest.raises(ManagedTranscriptionError):first.transcribe(str(source),beam_size=5)
    recovered=ManagedTranscriber(config(tmp_path),client=client)
    list(recovered.transcribe(str(source),beam_size=5)[0])
    outputs=[]
    assert recovered.commit_outputs(source,lambda:(outputs.append('canonical'),print('ARTIFACT: /synthetic/result.json'),True)[-1])
    again=ManagedTranscriber(config(tmp_path),client=client)
    list(again.transcribe(str(source),beam_size=5)[0])
    assert again.commit_outputs(source,lambda:outputs.append('duplicate'))
    assert outputs==['canonical']
    assert len(client.requests)==1


def test_cli_managed_mode_reuses_original_output_pipeline_without_engine(tmp_path,monkeypatch):
    import faster_whisper_transcribe as fw
    import managed_transcription
    source=tmp_path/'input.wav';audio(source);client=Client()
    settings=tmp_path/'managed.json';settings.write_text(json.dumps(config(tmp_path)));settings.chmod(0o600)
    original=managed_transcription.ManagedTranscriber
    monkeypatch.setattr(managed_transcription,'ManagedTranscriber',lambda conf:original(conf,client=client))
    monkeypatch.setattr(fw,'load_runtime',lambda args:(_ for _ in ()).throw(AssertionError('local engine initialized')))
    output=tmp_path/'outputs'
    assert fw.main([str(source),'--managed-config',str(settings),'--no-speaker-labels',
        '--text-output','--output-dir',str(output)])==0
    assert list(output.glob('*.docx'))
    assert 'Useful test.' in next(output.glob('*.txt')).read_text()


def test_managed_watcher_rejects_fallback_configuration():
    import pytest
    from watch_transcriptions import parse_cli_args_by_backend,WatcherError
    job={'cli_args_by_backend':{'faster_whisper':['--managed-config','/private/settings.json']}}
    with pytest.raises(WatcherError):parse_cli_args_by_backend(job,['faster_whisper','assembly'])
    assert '--managed-config' in parse_cli_args_by_backend(job,['faster_whisper'])['faster_whisper']


def test_known_busy_retries_same_request_but_output_unknown_never_replays(tmp_path):
    import pytest
    from managed_transcription import ManagedTranscriptionError
    from inference_manager.service_client import ManagedClientError
    source=tmp_path/'input.wav';audio(source);client=Client();attempts=[]
    original=client.submit
    def busy_once(request):
        attempts.append(request['id'])
        if len(attempts)==1:raise ManagedClientError('queue_full',status=429,retry_after=.01)
        return original(request)
    client.submit=busy_once
    transcriber=ManagedTranscriber(config(tmp_path),client=client,sleep=lambda _:None)
    list(transcriber.transcribe(str(source),beam_size=5)[0])
    assert attempts[0]==attempts[1]
    with pytest.raises(RuntimeError):
        transcriber.commit_outputs(source,lambda:(_ for _ in ()).throw(RuntimeError('export fault')))
    with pytest.raises(ManagedTranscriptionError):transcriber.commit_outputs(source,lambda:True)
