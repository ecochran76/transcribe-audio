"""Watcher opt-in uses manager HTTP/owned CPU fixtures, never local models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from watch_transcriptions import backend_script_path, parse_backends, parse_cli_args_by_backend, extract_artifact_paths, WatcherError
from tests.test_managed_recording_cli import recording_service as recording_service


def test_managed_backend_is_explicit_without_cross_route_fallback():
    assert parse_backends({}) == ['assembly']
    assert parse_backends({'backend':'managed_recording'}) == ['managed_recording']
    assert backend_script_path('managed_recording').name == 'managed_recording_transcribe.py'
    with pytest.raises(WatcherError,match='only'):
        parse_cli_args_by_backend({},['assembly','managed_recording'])


def invoke(source, settings, *extra, umask=-1):
    env=dict(os.environ,PATH='/home/linuxbrew/.linuxbrew/bin:'+os.environ.get('PATH',''))
    return subprocess.run([sys.executable,str(Path(__file__).parents[1]/'managed_recording_transcribe.py'),str(source),'--config',str(settings),*extra],capture_output=True,text=True,timeout=25,env=env,umask=umask)


def test_manager_exports_replay_through_watcher_artifact_seam(recording_service):
    source,settings,root=recording_service
    first=invoke(source,settings,'--text-output')
    assert first.returncode == 0,first.stderr
    paths=extract_artifact_paths(first.stdout)
    assert len(paths)==1
    artifact=json.loads(Path(paths[0]).read_text())
    assert artifact['backend']=='managed-recording'
    assert all(Path(p).is_file() for p in artifact['output_paths'].values())
    second=invoke(source,settings)
    assert second.returncode==0,second.stderr
    assert extract_artifact_paths(second.stdout)==paths
    assert len((root/'starts').read_text().splitlines())==2
    assert len(list((root/'state/watcher-ingress').glob('*.json')))==1


def test_failure_restart_reuses_ids_and_pending_requests(recording_service,monkeypatch,capsys):
    import managed_recording_transcribe as adapter
    source,settings,root=recording_service
    monkeypatch.setenv('PATH','/home/linuxbrew/.linuxbrew/bin:'+os.environ.get('PATH',''))
    original=adapter.managed.run
    def checkpoint_one(*args,**kwargs):
        return original(*args,**kwargs,max_stage_results=1)
    # Stop at the existing public durable stage-result boundary, without a
    # scheduling-dependent tiny timeout or a fabricated successful response.
    monkeypatch.setattr(adapter.managed,'run',checkpoint_one)
    assert adapter.main([str(source),'--config',str(settings)])==1
    assert extract_artifact_paths(capsys.readouterr().out)==[]
    manifests=list((root/'state/watcher-ingress').glob('*.json'))
    assert len(manifests)==1
    identity=json.loads(manifests[0].read_text())
    monkeypatch.setattr(adapter.managed,'run',original)
    assert adapter.main([str(source),'--config',str(settings)])==0
    paths=extract_artifact_paths(capsys.readouterr().out)
    artifact=json.loads(Path(paths[0]).read_text())
    assert artifact['recording_id']==identity['recording_id']
    assert artifact['conversation_id']==identity['conversation_id']
    assert len((root/'starts').read_text().splitlines())==2


def test_unsupported_calendar_flags_refuse_before_manager_requests(recording_service):
    source,settings,root=recording_service
    result=invoke(source,settings,'--use-calendar')
    assert result.returncode!=0
    assert 'unrecognized arguments' in result.stderr
    assert not (root/'starts').exists()
    assert not (root/'state/watcher-ingress').exists()


def test_ingress_prepares_private_state_under_normal_user_umask(recording_service):
    source,settings,root=recording_service
    result=invoke(source,settings,umask=0o022)
    assert result.returncode==0,result.stderr
    assert (root/'state').stat().st_mode & 0o077 == 0


def test_new_content_gets_new_ingress_ids_preserving_old_exports(recording_service):
    import wave
    source,settings,root=recording_service
    first=invoke(source,settings)
    assert first.returncode==0,first.stderr
    original_path=Path(extract_artifact_paths(first.stdout)[0])
    old=original_path.read_bytes()
    original=json.loads(old)
    with wave.open(str(source),'wb') as out:
        out.setparams((2,2,22050,0,'NONE',''));out.writeframes(b'\x01\0\x01\0'*22050)
    second=invoke(source,settings)
    assert second.returncode==0,second.stderr
    new=json.loads(Path(extract_artifact_paths(second.stdout)[0]).read_text())
    assert new['recording_id']!=original['recording_id']
    assert new['conversation_id']!=original['conversation_id']
    assert original_path.read_bytes()==old
    assert len(list((root/'state/watcher-ingress').glob('*.json')))==2


def test_partial_receipt_cannot_emit_watcher_success(recording_service,monkeypatch,capsys):
    import managed_recording_transcribe as adapter
    source,settings,root=recording_service
    monkeypatch.setattr(adapter.managed,'run',lambda *args,**kwargs:dict(state='partial',published=False))
    assert adapter.main([str(source),'--config',str(settings)])==1
    assert extract_artifact_paths(capsys.readouterr().out)==[]
    assert not (root/'starts').exists()


def test_real_watcher_backend_marks_success_and_ingests_store(recording_service,monkeypatch):
    import time
    from watch_transcriptions import WatchJob,JobState,process_file,should_retry
    source,settings,root=recording_service
    monkeypatch.setenv('PATH','/home/linuxbrew/.linuxbrew/bin:'+os.environ.get('PATH',''))
    job=WatchJob(name='managed-optin',watch_dir=root,glob='*.wav',backends=['managed_recording'],recursive=False,settle_seconds=0,min_age_seconds=0,scan_interval=30,failure_retry_seconds=900,cli_args={'managed_recording':['--config',str(settings),'--text-output']},notify_on_success=False,notify_on_failure=False,slack_channel=None,store_enabled=True,store_dir=root/'library',store_embedding_provider='debug-hash',store_embedding_model='debug-hash')
    state=JobState(processed={},candidates={})
    assert process_file(job,source,time.time(),state)
    record=state.processed[str(source)]
    assert record.status=='success'
    assert record.attempted_backends==['managed_recording']
    assert len(record.artifact_paths)==1
    assert len(record.store_paths)==1
    assert Path(record.store_paths[0]).is_file()
    assert not should_retry(record,time.time(),record.fingerprint,record.size)
    assert len((root/'starts').read_text().splitlines())==2


def test_existing_transcript_identity_and_metadata_are_preserved(recording_service):
    source,settings,root=recording_service
    old=root/'existing.json'
    old.write_text(json.dumps(dict(recording_id='recording-retained',conversation_id='conversation-retained',event={'summary':'Accepted calendar metadata'},output_paths={})))
    old.chmod(0o600)
    result=invoke(source,settings,'--existing-transcript',str(old))
    assert result.returncode==0,result.stderr
    artifact=json.loads(Path(extract_artifact_paths(result.stdout)[0]).read_text())
    assert artifact['recording_id']=='recording-retained'
    assert artifact['conversation_id']=='conversation-retained'
    assert artifact['event']=={'summary':'Accepted calendar metadata'}
    rejected=invoke(source,settings,'--existing-transcript',str(old),'--recording-id','wrong')
    assert rejected.returncode==1
    assert extract_artifact_paths(rejected.stdout)==[]
    assert len((root/'starts').read_text().splitlines())==2
