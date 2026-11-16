

def test_reconcile_after_restart_keeps_pause_and_resumes_only_observation(tmp_path):
    from managed_service_intent import save_intent, reconcile
    path=tmp_path/'intent.json'
    save_intent(path, batch_service='batch.service', paused_services=['speech.service'], continue_batch=True)
    calls=[]
    states={'batch.service':'inactive','speech.service':'active'}
    def control(action, unit):
        calls.append((action,unit));states[unit]='active' if action=='start' else 'inactive'
    receipt=reconcile(path,observe=lambda unit:states[unit],control=control)
    assert calls==[('stop','speech.service'),('start','batch.service')]
    assert receipt['state']=='reconciled'
    calls.clear()
    assert reconcile(path,observe=lambda unit:states[unit],control=control)['state']=='reconciled'
    assert calls==[]


def test_uncertain_pause_prevents_batch_start(tmp_path):
    from managed_service_intent import save_intent, reconcile
    path=tmp_path/'intent.json';save_intent(path,batch_service='batch.service',paused_services=['speech.service'],continue_batch=True)
    calls=[]
    def control(action,unit):calls.append((action,unit))
    r=reconcile(path,observe=lambda unit:'active' if unit=='speech.service' else 'inactive',control=control)
    assert r['state']=='held'
    assert calls==[('stop','speech.service')]


def test_terminal_queue_disarms_continuation_but_keeps_pause(tmp_path):
    import json
    from managed_service_intent import save_intent,reconcile
    queue=tmp_path/'queue.json';queue.write_text(json.dumps({'items':[{'status':'succeeded'},{'status':'failed'}]}));queue.chmod(0o600)
    path=tmp_path/'intent.json';save_intent(path,batch_service='batch.service',paused_services=['speech.service'],continue_batch=True,queue_file=queue)
    calls=[]
    receipt=reconcile(path,observe=lambda unit:'inactive',control=lambda *args:calls.append(args))
    assert receipt['continue_batch'] is False
    assert json.loads(path.read_text())['continue_batch'] is False
    assert calls==[]


def test_missing_queue_holds_batch_but_still_restores_pause(tmp_path):
    from managed_service_intent import save_intent,reconcile
    path=tmp_path/'intent.json';save_intent(path,batch_service='batch.service',paused_services=['speech.service'],continue_batch=True,queue_file=tmp_path/'missing.json')
    states={'speech.service':'active','batch.service':'inactive'};actions=[]
    def control(action,unit):
        actions.append((action,unit));states[unit]='inactive'
    receipt=reconcile(path,observe=lambda unit:states[unit],control=control)
    assert actions==[('stop','speech.service')]
    assert receipt['state']=='held' and receipt['reason']=='queue_unavailable'
