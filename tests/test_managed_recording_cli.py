"""Public CLI over real authenticated HTTP and owned scripted CPU workers."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

from docx import Document
import pytest

from inference_manager import Capacity, Principal, Profile
from inference_manager.adapters import ProcessAdapter
from inference_manager.service_runtime import ServiceRuntime
from inference_manager.shared_service import ConsumerPolicy, SharedService


@pytest.fixture
def recording_service(tmp_path):
    source = tmp_path/'original name.wav'
    with wave.open(str(source), 'wb') as out:
        out.setparams((2, 2, 22050, 0, 'NONE', ''))
        out.writeframes(b'\0\0\0\0'*22050)
    script = tmp_path/'engine.py'
    script.write_text('''import json,sys,wave
from pathlib import Path
stage,identity,spool,log=sys.argv[1:]
print(json.dumps({'type':'ready'}),flush=True)
for line in sys.stdin:
    packet=json.loads(line)
    if packet['type']!='start': continue
    Path(log).open('a').write(packet['job_id']+'\\n')
    if packet['payload']['options'].get('cancel_once') and stage=='asr' and sum('asr-' in row for row in Path(log).read_text().splitlines())==1:
        print(json.dumps(dict(type='result',job_id=packet['job_id'],outcome='cancelled')),flush=True)
        continue
    with wave.open(str(Path(spool)/packet['payload']['artifact_id'])) as audio:
        duration=audio.getnframes()/audio.getframerate()
    result=dict(stage=stage,duration_seconds=duration,model={'identity':identity})
    if stage=='asr': result['words']=[dict(text='Hello',start_seconds=.1,end_seconds=.3),dict(text='gap',start_seconds=.6,end_seconds=.8)]
    else: result.update(turns=[dict(speaker='0',start_seconds=0,end_seconds=.4)],speaker_vectors={'0':[1,0]})
    print(json.dumps(dict(type='result',job_id=packet['job_id'],outcome='succeeded',result=result)),flush=True)
''')
    owner = Principal('fixture', 'recording')
    profiles = {stage: Profile(stage, 1, {'model': 'synthetic-'+stage})
                for stage in ('asr', 'diarization')}
    service = SharedService(tmp_path/'service', budget_bytes=4,
                            capacity=lambda: Capacity(8, time.time()), operator=Principal('fixture', 'operator'),
                            consumers=[ConsumerPolicy(owner, tuple(profiles), ('historical',))])
    for stage, profile in profiles.items():
        worker = ProcessAdapter([sys.executable, str(script), stage, profile.fingerprint,
                                 str(tmp_path/'spool'), str(tmp_path/'starts')],
                                tmp_path/('worker-'+stage))
        service.register_pool(profile, [worker])
    runtime = ServiceRuntime(service, {'fixture-token': owner}, allow_insecure_loopback=True)
    runtime.start()
    token = tmp_path/'token'
    token.write_text('fixture-token')
    token.chmod(0o600)
    config = dict(endpoint=f'http://127.0.0.1:{runtime.address[1]}', token_file=str(token),
                  allow_insecure_loopback=True, output_directory=str(tmp_path/'exports'),
                  wait_seconds=10, poll_seconds=.01,
                  processor=dict(state_directory=str(tmp_path/'state'), spool_root=str(tmp_path/'spool'),
                                 unit_seconds=1, speaker_similarity=.9, speaker_margin=.1,
                                 profiles={stage: dict(profile=p.name, identity=p.fingerprint)
                                           for stage, p in profiles.items()}))
    settings = tmp_path/'config.json'
    settings.write_text(json.dumps(config))
    settings.chmod(0o600)
    try:
        yield source, settings, tmp_path
    finally:
        runtime.close()


def invoke(source, settings, *extra):
    env = dict(os.environ)
    env['PATH'] = '/home/linuxbrew/.linuxbrew/bin:'+env.get('PATH', '')
    return subprocess.run([sys.executable, str(Path(__file__).parents[1]/'managed_recording_cli.py'),
                           str(source), '--config', str(settings), '--recording-id', 'recording-1',
                           '--conversation-id', 'conversation-1', *extra],
                          env=env, capture_output=True, text=True, timeout=25)


def test_real_cli_exports_readable_versions_and_reuses_publication(recording_service):
    source, settings, root = recording_service
    original = source.read_bytes()
    old = root/'old.json'
    old.write_text(json.dumps(dict(event={'summary': 'Original calendar'},
                                   recording_id='recording-1', conversation_id='conversation-1',
                                   output_paths={'docx': str(root/'old.docx')})))
    old.chmod(0o600)
    (root/'old.docx').write_bytes(b'previous canonical artifact')
    first = invoke(source, settings, '--existing-transcript', str(old))
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    artifact = json.loads(Path(receipt['output_paths']['json']).read_text())
    assert artifact['source_media_path'] == str(source)
    assert artifact['conversation_id'] == 'conversation-1'
    assert artifact['event'] == {'summary': 'Original calendar'}
    assert [u['speaker'] for u in artifact['utterances']] == ['speaker-1', 'UNKNOWN']
    assert 'Hello' in '\n'.join(p.text for p in Document(receipt['output_paths']['docx']).paragraphs)
    assert 'UNKNOWN' in Path(receipt['output_paths']['txt']).read_text()
    repeated = invoke(source, settings, '--existing-transcript', str(old))
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout) == receipt
    assert len((root/'starts').read_text().splitlines()) == 2
    assert len(list((root/'exports').rglob('*.docx'))) == 1
    assert (root/'old.docx').read_bytes() == b'previous canonical artifact'
    assert source.read_bytes() == original
    assert json.loads(old.read_text())['event']['summary'] == 'Original calendar'


def test_restarted_cli_rejects_profile_and_output_configuration_drift(recording_service):
    source, settings, root = recording_service
    first = invoke(source, settings)
    assert first.returncode == 0, first.stderr
    config = json.loads(settings.read_text())
    config['processor']['unit_seconds'] = 2
    config['output_directory'] = str(root/'different-exports')
    settings.write_text(json.dumps(config))
    second = invoke(source, settings)
    assert second.returncode == 1
    assert 'configuration/metadata conflict' in second.stderr
    assert len((root/'starts').read_text().splitlines()) == 2
    assert not (root/'different-exports').exists()


def test_uncertain_store_publication_keeps_exports_and_cannot_replay(recording_service):
    source, settings, root = recording_service
    blocked = root/'blocked-store'
    blocked.write_text('not a directory')
    config = json.loads(settings.read_text())
    config['transcript_store'] = dict(root=str(blocked), embedding_provider='debug-hash')
    settings.write_text(json.dumps(config))
    first = invoke(source, settings)
    assert first.returncode == 1
    outputs = list((root/'exports').rglob('transcript.*'))
    assert len(outputs) == 3
    retained = {str(path): path.read_bytes() for path in outputs}
    second = invoke(source, settings)
    assert second.returncode == 1
    assert 'publication uncertain' in second.stderr
    assert {str(path): path.read_bytes() for path in outputs} == retained
    assert len((root/'starts').read_text().splitlines()) == 2


def test_explicit_cpu_store_ingestion_returns_same_receipt_on_restart(recording_service):
    source, settings, root = recording_service
    config = json.loads(settings.read_text())
    config['transcript_store'] = dict(root=str(root/'library'), embedding_provider='debug-hash')
    settings.write_text(json.dumps(config))
    first = invoke(source, settings)
    assert first.returncode == 0, first.stderr
    receipt = json.loads(first.stdout)
    assert receipt['store']['kind'] == 'transcript'
    assert Path(receipt['store']['stored_path']).is_file()
    second = invoke(source, settings)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout) == receipt
    assert len((root/'starts').read_text().splitlines()) == 2


def test_new_source_generation_retains_previous_canonical_files(recording_service):
    source, settings, root = recording_service
    first = invoke(source, settings)
    assert first.returncode == 0, first.stderr
    previous = json.loads(first.stdout)
    retained = {path: Path(path).read_bytes() for path in previous['output_paths'].values()}
    with wave.open(str(source), 'wb') as out:
        out.setparams((2, 2, 22050, 0, 'NONE', ''))
        out.writeframes(b'\x01\0\x01\0'*22050)
    second = invoke(source, settings)
    assert second.returncode == 0, second.stderr
    current = json.loads(second.stdout)
    assert current['generation'] != previous['generation']
    assert len(list((root/'exports').rglob('*.docx'))) == 2
    assert {path: Path(path).read_bytes() for path in retained} == retained


def test_word_attribution_requires_full_coverage_without_competing_turns():
    from managed_recording_cli import attribute_words
    words = [dict(text='gap', start_seconds=0, end_seconds=1),
             dict(text='overlap', start_seconds=1, end_seconds=2)]
    turns = [dict(speaker='s1', start_seconds=0, end_seconds=.4),
             dict(speaker='s1', start_seconds=.6, end_seconds=2),
             dict(speaker='s2', start_seconds=1.5, end_seconds=2)]
    utterances = attribute_words(words, turns)
    assert {row['speaker'] for row in utterances} == {'UNKNOWN'}
    assert utterances[0]['text'] == 'gap overlap'


def test_confirmed_cancellation_retries_only_with_bounded_new_attempt(recording_service):
    source, settings, root = recording_service
    config = json.loads(settings.read_text())
    config.update(max_stage_attempts=2, options={'cancel_once': True})
    settings.write_text(json.dumps(config))
    result = invoke(source, settings)
    assert result.returncode == 0, result.stderr
    starts = (root/'starts').read_text().splitlines()
    assert len(starts) == 3
    assert len(set(starts)) == 3
    checkpoints = list((root/'state').glob('*.json'))
    checkpoint = json.loads(checkpoints[0].read_text())
    assert checkpoint['units'][0]['stages']['asr']['history'][0]['state'] == 'cancelled'


def test_unknown_submission_is_observed_without_replay_across_cli_restarts(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    observations = []
    class Unknown(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.connection.close()  # Accepted, but acknowledgement was lost.
        def do_GET(self):
            observations.append(self.path)
            body = json.dumps(dict(state='unknown')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Unknown)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = tmp_path/'input.wav'
    with wave.open(str(source), 'wb') as out:
        out.setparams((1, 2, 16000, 0, 'NONE', ''))
        out.writeframes(b'\0\0'*16000)
    token = tmp_path/'token'
    token.write_text('fixture-token')
    token.chmod(0o600)
    config = dict(endpoint=f'http://127.0.0.1:{server.server_port}', token_file=str(token),
                  allow_insecure_loopback=True, output_directory=str(tmp_path/'exports'),
                  wait_seconds=.1, poll_seconds=.01,
                  processor=dict(state_directory=str(tmp_path/'state'), spool_root=str(tmp_path/'spool'),
                                 unit_seconds=1, speaker_similarity=.9, speaker_margin=.1,
                                 profiles={stage: dict(profile=stage, identity='a'*64)
                                           for stage in ('asr', 'diarization')}))
    settings = tmp_path/'config.json'
    settings.write_text(json.dumps(config))
    settings.chmod(0o600)
    try:
        for _ in range(2):
            result = invoke(source, settings)
            assert result.returncode == 1
            assert 'bounded wait expired' in result.stderr
        assert len(requests) == 1
        assert observations
        assert not (tmp_path/'exports').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_cli_budget_checkpoints_before_dia_and_same_ids_finish(recording_service):
    from transcript_artifact_access import ResolvedTranscriptArtifact, read_resolved_transcript
    source, settings, root = recording_service
    config_bytes = settings.read_bytes()
    first = invoke(source, settings, '--max-stage-results', '1')
    assert first.returncode == 0, first.stderr
    partial = json.loads(first.stdout)
    assert partial['state'] == 'partial'
    assert partial['published'] is False and partial['resumable'] is True
    assert partial['new_stage_results'] == 1
    assert 'output_paths' not in partial
    assert (root/'starts').read_text().splitlines() == ['fixture/recording/' + partial['completed_stage_results'][0]['request_id']]
    assert partial['completed_stage_results'][0]['stage'] == 'asr'
    assert not (root/'worker-diarization/custody.json').exists()
    checkpoint = json.loads(Path(partial['checkpoint_path']).read_text())
    asr = checkpoint['units'][0]['stages']['asr']
    frozen_asr = json.loads(json.dumps(asr))
    assert checkpoint['units'][0]['stages']['diarization']['state'] == 'pending'
    second = invoke(source, settings, '--max-stage-results', '1')
    assert second.returncode == 0, second.stderr
    receipt = json.loads(second.stdout)
    assert receipt['generation'] == partial['processor_key']
    assert 'output_paths' in receipt and 'state' not in receipt
    final = json.loads(Path(partial['checkpoint_path']).read_text())
    assert final['units'][0]['stages']['asr'] == frozen_asr
    assert len((root/'starts').read_text().splitlines()) == 2
    assert settings.read_bytes() == config_bytes
    artifact = json.loads(Path(receipt['output_paths']['json']).read_text())
    txt = Path(receipt['output_paths']['txt']).read_text()
    docx = '\n'.join(p.text for p in Document(receipt['output_paths']['docx']).paragraphs)
    for row in artifact['utterances']:
        assert row['text'] in txt and row['text'] in docx
    public = read_resolved_transcript(ResolvedTranscriptArtifact(
        path=Path(receipt['output_paths']['json']), location='source',
        expected_sha256=receipt['sha256']['json'],
        actual_sha256=hashlib.sha256(Path(receipt['output_paths']['json']).read_bytes()).hexdigest()))
    assert public == artifact
    repeated = invoke(source, settings, '--max-stage-results', '1')
    assert repeated.returncode == 0 and json.loads(repeated.stdout) == receipt
    assert len((root/'starts').read_text().splitlines()) == 2


def test_budget_counts_only_new_results_across_multi_unit_resumption(recording_service):
    source, settings, root = recording_service
    with wave.open(str(source), 'wb') as out:
        out.setparams((2, 2, 22050, 0, 'NONE', ''))
        out.writeframes(b'\0\0\0\0' * (22050*2))
    first = invoke(source, settings, '--max-stage-results', '2')
    assert first.returncode == 0, first.stderr
    initial = json.loads(first.stdout)
    assert initial['new_stage_results'] == 2
    assert [x['stage'] for x in initial['completed_stage_results']] == ['asr', 'asr']
    assert not (root/'worker-diarization/custody.json').exists()
    second = invoke(source, settings, '--max-stage-results', '1')
    assert second.returncode == 0, second.stderr
    partial = json.loads(second.stdout)
    assert partial['processor_key'] == initial['processor_key']
    assert partial['new_stage_results'] == 1
    assert len(partial['completed_stage_results']) == 3
    assert len((root/'starts').read_text().splitlines()) == 3
    third = invoke(source, settings, '--max-stage-results', '1')
    assert third.returncode == 0, third.stderr
    assert json.loads(third.stdout)['generation'] == initial['processor_key']
    assert len((root/'starts').read_text().splitlines()) == 4


def test_invalid_stage_result_budgets_refuse_api_and_cli_before_effects(recording_service):
    from managed_recording_cli import run
    source, settings, root = recording_service
    config = json.loads(settings.read_text())
    for value in (0, -1, True, 1.0, '1', 10001):
        with pytest.raises(ValueError, match='max_stage_results'):
            run(source, config, recording_id='recording-1', conversation_id='conversation-1',
                max_stage_results=value)
    for value in ('0', '-1', '10001', 'not-an-int'):
        result = invoke(source, settings, '--max-stage-results', value)
        assert result.returncode != 0
    assert not (root/'starts').exists()
    assert not (root/'state').exists()
    assert not (root/'exports').exists()
    assert not (root/'worker-asr/custody.json').exists()
    assert not (root/'worker-diarization/custody.json').exists()
