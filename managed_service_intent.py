"""Persist explicit service intent and reconcile restart without inference replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from managed_recording_cli import _private_directory, _private_read, _save


def _validate(intent):
    if (set(intent) != {'schema_version','batch_service','paused_services','continue_batch','queue_file'}
            or type(intent['schema_version']) is not int or intent['schema_version'] != 1
            or type(intent['continue_batch']) is not bool
            or not isinstance(intent['paused_services'],list)
            or len(intent['paused_services']) > 8):
        raise ValueError('explicit bounded service intent required')
    if intent['queue_file'] is not None and (not isinstance(intent['queue_file'],str)
            or not Path(intent['queue_file']).is_absolute()):
        raise ValueError('absolute queue path required')
    units=[intent['batch_service'],*intent['paused_services']]
    if (len(set(units))!=len(units) or any(not isinstance(u,str)
            or not re.fullmatch(r'[A-Za-z0-9_.@-]{1,128}\.service',u) for u in units)):
        raise ValueError('distinct explicit service names required')
    return intent


def save_intent(path, *, batch_service, paused_services, continue_batch, queue_file=None):
    intent=_validate(dict(schema_version=1,batch_service=batch_service,
                          paused_services=paused_services,continue_batch=continue_batch,
                          queue_file=str(queue_file) if queue_file is not None else None))
    path=Path(path);_private_directory(path.parent);_save(path,intent)
    return intent


def system_state(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],
                          capture_output=True,text=True,timeout=10,check=True)
    return result.stdout.strip()


def system_control(action,unit):
    subprocess.run(['systemctl','--user',action,unit],capture_output=True,timeout=30,check=True)


def reconcile(path, *, observe=system_state, control=system_control):
    intent=_validate(json.loads(_private_read(Path(path))))
    if intent['continue_batch'] and intent['queue_file']:
        queue=json.loads(_private_read(Path(intent['queue_file']),64*1024*1024))
        items=queue['items']
        if (not isinstance(items,list) or not items
                or any(item.get('status') not in ('succeeded','failed','pending','running') for item in items)):
            raise ValueError('bounded known queue states required')
        if all(item['status'] in ('succeeded','failed') for item in items):
            intent['continue_batch']=False
            _save(Path(path),intent)
    actions=[]
    for unit in intent['paused_services']:
        state=observe(unit)
        if state!='inactive':
            if state not in ('active','activating','deactivating','failed'):
                return dict(state='held',reason='pause_state_unknown',actions=actions)
            control('stop',unit);actions.append(dict(action='stop',unit=unit))
            if observe(unit)!='inactive':
                return dict(state='held',reason='pause_not_confirmed',actions=actions)
    batch=intent['batch_service'];state=observe(batch)
    if intent['continue_batch']:
        if state in ('inactive','failed'):
            control('start',batch);actions.append(dict(action='start',unit=batch))
            if observe(batch) not in ('active','activating'):
                return dict(state='held',reason='batch_start_not_confirmed',actions=actions)
        elif state not in ('active','activating'):
            return dict(state='held',reason='batch_state_unknown',actions=actions)
    elif state not in ('inactive','failed'):
        # Completing a batch changes intent, not ownership of arbitrary processes.
        return dict(state='held',reason='completed_batch_still_active',actions=actions)
    return dict(state='reconciled',actions=actions,continue_batch=intent['continue_batch'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        receipt=reconcile(args.intent)
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        from managed_operations import failure_receipt
        receipt=failure_receipt(error,operation='service-intent-reconcile')
    _private_directory(args.receipt.parent);_save(args.receipt,receipt)
    print(json.dumps(dict(state=receipt['state'])))
    return 0 if receipt['state']=='reconciled' else 1


if __name__=='__main__':
    raise SystemExit(main())
