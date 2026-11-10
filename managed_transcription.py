"""Explicit managed ASR client; application retains canonical output ownership."""
from __future__ import annotations

import hashlib
import contextlib
import io
import sys
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from types import SimpleNamespace
import wave


class ManagedTranscriptionError(RuntimeError):
    """Managed inference is unavailable or requires explicit reconciliation."""


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _private_directory(path):
    path.mkdir(parents=True,exist_ok=True,mode=0o700)
    info=path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
        raise ValueError('private owned managed directory required')
    return path


def _read_private(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_nlink!=1:
            raise ValueError('private owned managed file required')
        value=stream.read(limit+1)
    if len(value)>limit:raise ValueError('managed file exceeds bound')
    return value


def load_config(path):
    value=json.loads(_read_private(Path(path).expanduser(),65536))
    if not isinstance(value,dict):raise ValueError('managed settings object required')
    return value


class ManagedTranscriber:
    def __init__(self,config,*,client=None,clock=time.monotonic,sleep=time.sleep):
        self.config=dict(config);self.clock=clock;self.sleep=sleep
        self.spool=_private_directory(Path(config['spool_root']).expanduser())
        self.state=_private_directory(Path(config['state_directory']).expanduser())
        if not re.fullmatch(r'[a-f0-9]{64}',config['identity']):raise ValueError('pinned profile fingerprint required')
        self.max_workers=config.get('max_workers',2)
        if type(self.max_workers) is not int or not 1<=self.max_workers<=2:raise ValueError('pilot worker ceiling must be1or2')
        self.poll=config.get('poll_seconds',.25);self.wait=config.get('wait_seconds',3600)
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (self.poll,self.wait)):
            raise ValueError('bounded positive managed polling/wait required')
        if self.poll>60 or self.wait>86400:raise ValueError('managed wait exceeds bound')
        self._records={}
        if client is None:
            from inference_manager.service_client import ManagedClient
            token=_read_private(Path(config['token_file']).expanduser(),4096).decode().strip()
            import ssl
            tls=ssl.create_default_context(cafile=str(Path(config['ca_file']).expanduser()) if config.get('ca_file') else None)
            client=ManagedClient(config['endpoint'],token,ssl_context=tls,
                timeout_seconds=config.get('request_timeout_seconds',10),
                allow_insecure_loopback=config.get('allow_insecure_loopback',False))
        self.client=client

    def _save(self,key,record):
        fd,name=tempfile.mkstemp(prefix='.managed-',dir=self.state)
        try:
            with os.fdopen(fd,'w') as out:
                out.write(_json(record));out.flush();os.fsync(out.fileno())
            os.replace(name,self.state/(key+'.json'))
            directory=os.open(self.state,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            if os.path.exists(name):os.unlink(name)

    def _stage(self,source):
        fd,name=tempfile.mkstemp(prefix='.audio-',suffix='.wav',dir=self.spool);os.close(fd)
        temporary=Path(name)
        try:
            try:
                with wave.open(str(source),'rb') as original:
                    already_pcm=(original.getnchannels()==1 and original.getsampwidth()==2 and original.getframerate()==16000 and original.getcomptype()=='NONE')
            except (wave.Error,EOFError):already_pcm=False
            if already_pcm:
                temporary.write_bytes(source.read_bytes())
            else:
                subprocess.run([self.config.get('ffmpeg','ffmpeg'),'-nostdin','-y','-v','error','-i',str(source),'-map','0:a:0',
                    '-ac','1','-ar','16000','-c:a','pcm_s16le',str(temporary)],check=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
            raw=temporary.read_bytes()
            if len(raw)>self.config.get('max_audio_bytes',64*1024**2):raise ValueError('audio exceeds pilot byte bound')
            with wave.open(str(temporary),'rb') as stream:
                if stream.getnframes()/stream.getframerate()>self.config.get('max_audio_seconds',300):
                    raise ManagedTranscriptionError('recording exceeds selected whole-file pilot duration; retained pending')
            digest=hashlib.sha256(raw).hexdigest();artifact=self.spool/(digest+'.wav')
            try:
                fd=os.open(artifact,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            except FileExistsError:
                if _read_private(artifact,64*1024**2)!=raw:raise ValueError('staged artifact identity conflict')
            else:
                with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
            return artifact.name,digest
        finally:temporary.unlink(missing_ok=True)

    @contextlib.contextmanager
    def _locked(self,key):
        fd=os.open(self.state/(key+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a+') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_nlink!=1:
                raise ValueError('private managed custody lock required')
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ManagedTranscriptionError('another process owns this managed source; retain pending') from None
            try:yield
            finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

    def transcribe(self,audio_path,**options):
        source=Path(audio_path).expanduser().resolve()
        # Include original source bytes so retries preserve source/profile/options identity.
        digest_state=hashlib.sha256()
        with source.open('rb') as audio:
            for block in iter(lambda:audio.read(1024**2),b''):digest_state.update(block)
        digest=digest_state.hexdigest()
        key=hashlib.sha256(_json([str(source),digest,self.config['profile'],self.config['identity'],options]).encode()).hexdigest()
        with self._locked(key):return self._transcribe(source,key,options)

    def _transcribe(self,source,key,options):
        record_path=self.state/(key+'.json')
        record=json.loads(_read_private(record_path,10*1024**2)) if record_path.exists() else None
        if record is None:
            artifact,sha=self._stage(source)
            request=dict(id='asr-'+key,profile=self.config['profile'],identity=self.config['identity'],
                workload_class='background',max_workers=self.max_workers,
                payload=dict(artifact_id=artifact,sha256=sha,options=options))
            record=dict(state='pending',request=request)
            self._save(key,record)
        end=self.clock()+self.wait
        if record.get('result') is None:
            while self.clock()<end:
                if record['state']=='pending':
                    record['state']='submitting';self._save(key,record)
                    try:view=self.client.submit(record['request'])
                    except Exception as error:
                        if getattr(error,'code',None)=='queue_full' and getattr(error,'status',None)==429:
                            record['state']='pending';self._save(key,record)
                            retry=getattr(error,'retry_after',None)
                            self.sleep(min(max(retry if type(retry) in (int,float) else 1,.01),60));continue
                        record['state']='unknown';self._save(key,record)
                        raise ManagedTranscriptionError('managed submission uncertain; retained for observation, never replayed') from None
                else:
                    try:view=self.client.observe(record['request']['id'])
                    except Exception:
                        raise ManagedTranscriptionError('managed job observation unavailable; retained without replay') from None
                record['state']=view['state'];self._save(key,record)
                if view['state']=='succeeded':
                    result=self.client.result(record['request']['id'])
                    if result['state']!='succeeded':raise ManagedTranscriptionError('managed result not ready')
                    record['result']=result['result'];self._save(key,record);break
                if view['state'] in ('unknown','failed','cancelled','unavailable'):
                    raise ManagedTranscriptionError('managed '+view['state']+'; retained without route fallback')
                self.sleep(self.poll)
            if record.get('result') is None:raise ManagedTranscriptionError('managed wait expired; job retained without cancellation/replay')
        self._records[str(source)]=(key,record)
        result=record['result']
        segments=[]
        for row in result['segments']:
            item=dict(row);item['words']=[SimpleNamespace(**w) for w in item.get('words') or []]
            segments.append(SimpleNamespace(**item))
        return iter(segments),SimpleNamespace(**{k:v for k,v in result.items() if k!='segments'})

    def commit_outputs(self,source,commit):
        """Guard nontransactional exports; uncertain output effects never replay."""
        key,_=self._records[str(Path(source).expanduser().resolve())]
        with self._locked(key):
            record=json.loads(_read_private(self.state/(key+'.json'),10*1024**2))
            return self._commit_outputs(key,record,commit)

    def _commit_outputs(self,key,record,commit):
        marker=record.get('output_state')
        if marker=='complete':
            print(record.get('output_stdout',''),end='')
            return True
        if marker is not None:
            raise ManagedTranscriptionError('canonical output outcome uncertain; reconcile before replay')
        record['output_state']='exporting';self._save(key,record)
        captured=io.StringIO()
        original_stdout=sys.stdout
        class Tee:
            def write(self,value):captured.write(value);return original_stdout.write(value)
            def flush(self):original_stdout.flush()
        try:
            with contextlib.redirect_stdout(Tee()):success=bool(commit())
        except BaseException:
            record['output_state']='unknown';self._save(key,record);raise
        record['output_state']='complete' if success else 'unknown'
        record['output_stdout']=captured.getvalue();self._save(key,record)
        return success
