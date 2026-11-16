import hashlib
import json
import pytest


def test_preflight_rejects_source_and_environment_before_effects(tmp_path):
    from managed_operations import preflight
    source = tmp_path/'source.wav'
    source.write_bytes(b'audio')
    plan = dict(schema_version=1, mode='publication', source=str(source),
                source_sha256='0'*64, checkpoint=str(tmp_path/'missing.json'),
                required_commands=['definitely-missing-transcribe-command'],
                resources=dict(workers=2, gpu_budget_bytes=12*1024**3,
                               headroom_bytes=2*1024**3, system_ram_bytes=12*1024**3))
    with pytest.raises(ValueError, match='source'):
        preflight(plan)
    plan['source_sha256'] = hashlib.sha256(b'audio').hexdigest()
    with pytest.raises(ValueError, match='command'):
        preflight(plan)


def test_publication_only_reuses_complete_receipt_without_inference(tmp_path):
    from managed_operations import execute
    source = tmp_path/'source.wav';source.write_bytes(b'audio')
    state = tmp_path/'state';state.mkdir(mode=0o700)
    spool = tmp_path/'spool';spool.mkdir(mode=0o700)
    cfg = dict(state_directory=str(state), spool_root=str(spool), unit_seconds=60,
               profiles={s:dict(profile=s,identity=('a' if s=='asr' else 'b')*64)
                         for s in ('asr','diarization')},speaker_similarity=.9,speaker_margin=.1)
    paths = {}
    for kind in ('json','txt','docx'):
        p=tmp_path/('out.'+kind);p.write_bytes(b'output');paths[kind]=str(p)
    receipt=dict(output_paths=paths,sha256={k:hashlib.sha256(b'output').hexdigest()for k in paths})
    key='c'*64;cp=state/(key+'.json');cp.write_text(json.dumps(dict(key=key,identity=dict(config=cfg,recording_id='r',conversation_id='c'),state='ready',units=[dict(stages={s:dict(result={})for s in ('asr','diarization')})],publication_state='complete',publication_receipt=receipt)))
    plan=dict(schema_version=1,mode='publication',source=str(source),source_sha256=hashlib.sha256(b'audio').hexdigest(),checkpoint=str(cp),required_commands=[],resources=dict(workers=2,gpu_budget_bytes=12*1024**3,headroom_bytes=2*1024**3,system_ram_bytes=12*1024**3))
    assert execute(plan)==receipt


def test_private_script_preflight_catches_missing_name_without_execution(tmp_path):
    from managed_operations import check_script
    path=tmp_path/'operation.py'
    path.write_text('print(undefined_name)\n')
    r=check_script(path)
    assert not r['passed']
    assert r['findings'][0]['code']=='F821'


def test_failure_receipt_excludes_message_and_locals():
    from managed_operations import failure_receipt
    try:
        raise OSError(12,'PRIVATE-SECRET-TRANSCRIPT')
    except OSError as error:
        receipt=failure_receipt(error,operation='publication')
    assert 'PRIVATE-SECRET' not in json.dumps(receipt)
    assert receipt['diagnostic']['errno']==12


def test_resume_observes_last_stage_after_transport_failure_without_resubmitting(tmp_path):
    import wave
    from managed_recording import RecordingProcessor
    from managed_operations import execute
    from tests.test_managed_recording import Client, config
    source=tmp_path/'source.wav'
    with wave.open(str(source),'wb')as out:
        out.setparams((1,2,16000,0,'NONE',''));out.writeframes(b'\0\0'*16000)
    cfg=config(tmp_path);client=Client();client.lost=False
    original_submit=client.submit
    def submit(request):
        original_submit(request)
        return dict(state='running') if request['profile']=='diar' else dict(state='succeeded')
    client.submit=submit
    processor=RecordingProcessor(cfg,client=client);key=processor.prepare(source,recording_id='r',conversation_id='c')
    processor.advance(key);processor.advance(key)
    observations=[]
    def observe(request_id):
        observations.append(request_id)
        if len(observations)==1:raise TimeoutError()
        return dict(state='succeeded')
    client.observe=observe
    plan=dict(schema_version=1,mode='resume',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(tmp_path/'state'/(key+'.json')),required_commands=[],resources=dict(workers=2,gpu_budget_bytes=12*1024**3,headroom_bytes=2*1024**3,system_ram_bytes=12*1024**3),wait_seconds=2,poll_seconds=.01)
    result=execute(plan,client=client)
    assert result['state']=='ready'
    assert len(client.requests)==2
    assert len(observations)==2
