"""Recording-owner erasure at real transcript/blob stores; no provider or inference."""

import hashlib
import json
from pathlib import Path

import pytest

import transcript_api
import transcript_store

OWNER = {"tenant": "fixture", "app": "meeting", "account": "recordings"}
PRINCIPAL = {"tenant": "fixture", "app": "operator"}


def fixture(tmp_path):
    media = tmp_path / "originals" / "meeting.wav"
    media.parent.mkdir()
    media.write_bytes(b"RIFF-synthetic-original")
    artifact = tmp_path / "meeting.transcript.json"
    artifact.write_text(
        json.dumps(
            {
                "transcript_title": "Fixture",
                "source_media_path": str(media),
                "working_media_path": str(media),
                "transcript_text": "Fixture",
                "backend": "assembly",
                "utterances": [{"speaker": "A", "start": 0, "end": 1000}],
            }
        )
    )
    root = tmp_path / "store"
    document = transcript_store.ingest_artifact(
        artifact,
        root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    with transcript_store.connect(root) as con:
        blob = dict(
            con.execute(
                "SELECT * FROM blobs WHERE sha256=?",
                (hashlib.sha256(media.read_bytes()).hexdigest(),),
            ).fetchone()
        )
    source = {
        "service": "transcribe-audio",
        "account": OWNER["account"],
        "id": "ta-recording-"
        + __import__("speech_harvesting").digest(
            [OWNER["account"], document.id, blob["id"]]
        )[:40],
        "revision": blob["sha256"],
        "sha256": blob["sha256"],
    }
    return root, media, blob, source


def test_owner_requires_separate_authority_then_fences_and_purges_exact_original_copies_across_restart(
    tmp_path,
):
    from speech_source_erasure import RecordingSourceOwner

    root, original, blob, source = fixture(tmp_path)
    second_original = original.with_name("second-original.wav")
    second_original.write_bytes(original.read_bytes())
    second_artifact = tmp_path / "second.transcript.json"
    second_artifact.write_text(
        json.dumps(
            {
                "transcript_title": "Same owned recording",
                "source_media_path": str(second_original),
                "working_media_path": str(second_original),
                "transcript_text": "Fixture",
                "backend": "assembly",
                "utterances": [{"speaker": "A", "start": 0, "end": 1000}],
            }
        )
    )
    transcript_store.ingest_artifact(
        second_artifact,
        root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    options = dict(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="independent-owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    service = RecordingSourceOwner(**options)
    with pytest.raises(ValueError, match="forbidden"):
        service.request(
            "inventory",
            {"principal": PRINCIPAL, "source": source},
            "Bearer source-export-token",
        )
    with pytest.raises(ValueError, match="forbidden"):
        service.request(
            "inventory",
            {"principal": {**PRINCIPAL, "tenant": "other"}, "source": source},
            "Bearer independent-owner-token",
        )
    inventory = service.request(
        "inventory",
        {"principal": PRINCIPAL, "source": source},
        "Bearer independent-owner-token",
    )
    assert inventory["inventory"]["available"] is True
    assert len(inventory["inventory"]["items"]) == 3
    intent = {
        "principal": PRINCIPAL,
        "requestId": "erase-1",
        "source": source,
        "expectedFingerprint": inventory["inventory"]["fingerprint"],
    }
    assert (
        service.request("fence", intent, "Bearer independent-owner-token")["receipt"][
            "quiescent"
        ]
        is True
    )
    with pytest.raises(transcript_store.TranscriptStoreError, match="erased"):
        transcript_api.get_blob(blob["id"], root=root)
    assert original.exists()
    service.close()
    service = RecordingSourceOwner(**options)
    assert (
        service.request(
            "status",
            {"principal": PRINCIPAL, "requestId": "erase-1"},
            "Bearer independent-owner-token",
        )["receipt"]["phase"]
        == "fenced"
    )
    result = service.request("purge", intent, "Bearer independent-owner-token")
    assert result["receipt"]["privateCleanupVerified"] is True
    assert not original.exists()
    assert not second_original.exists()
    assert not Path(blob["stored_path"]).exists()
    assert (
        service.request(
            "purge",
            dict(reversed(list(intent.items()))),
            "Bearer independent-owner-token",
        )
        == result
    )
    recreated = tmp_path / "recreated.wav"
    recreated.write_bytes(b"RIFF-synthetic-original")
    with pytest.raises(transcript_store.TranscriptStoreError, match="erased"):
        transcript_store.prepare_blob(root, str(recreated))
    service.close()


def test_owner_http_requires_independent_token_and_rejects_changed_physical_copies(
    tmp_path,
):
    from contextlib import closing
    from http import HTTPStatus
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from speech_source_erasure import RecordingSourceOwner

    root, original, blob, source = fixture(tmp_path)
    service = RecordingSourceOwner(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=root,
        state_root=tmp_path / "state",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
        static_dir=None,
    )
    server.speech_source_owner = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(action, value, token="owner-token"):
        return json.load(
            urlopen(
                Request(
                    f"http://127.0.0.1:{server.server_port}/api/speech-harvest/source-erasure/{action}",
                    data=json.dumps(value).encode(),
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    },
                ),
                timeout=3,
            )
        )

    try:
        with pytest.raises(HTTPError) as denied:
            request(
                "inventory", {"principal": PRINCIPAL, "source": source}, "source-token"
            )
        assert denied.value.code == HTTPStatus.FORBIDDEN
        preview = request("inventory", {"principal": PRINCIPAL, "source": source})[
            "inventory"
        ]
        original.write_bytes(b"Changed outside requested immutable source")
        with pytest.raises(HTTPError):
            request(
                "fence",
                {
                    "principal": PRINCIPAL,
                    "requestId": "stale",
                    "source": source,
                    "expectedFingerprint": preview["fingerprint"],
                },
            )
        assert Path(blob["stored_path"]).exists()
        assert (
            request("inventory", {"principal": PRINCIPAL, "source": source})[
                "inventory"
            ]["available"]
            is False
        )
        original.unlink()
        original.symlink_to(Path(blob["stored_path"]))
        assert (
            request("inventory", {"principal": PRINCIPAL, "source": source})[
                "inventory"
            ]["available"]
            is False
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_owner_resumes_its_recorded_partial_purge_after_lost_unlink_ack_without_touching_other_sources(
    tmp_path, monkeypatch
):
    from speech_source_erasure import RecordingSourceOwner

    root, original, blob, source = fixture(tmp_path)
    unrelated = original.parent / "other.wav"
    unrelated.write_bytes(b"Different source")
    options = dict(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    service = RecordingSourceOwner(**options)
    preview = service.request(
        "inventory", {"principal": PRINCIPAL, "source": source}, "Bearer owner-token"
    )["inventory"]
    intent = {
        "principal": PRINCIPAL,
        "requestId": "partial",
        "source": source,
        "expectedFingerprint": preview["fingerprint"],
    }
    service.request("fence", intent, "Bearer owner-token")
    native = Path.unlink
    calls = []

    def lose_ack(path, *args, **kwargs):
        native(path, *args, **kwargs)
        calls.append(path)
        raise OSError("lost acknowledgement after known original unlink")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", lose_ack)
        with pytest.raises(OSError):
            service.request("purge", intent, "Bearer owner-token")
    service.close()
    service = RecordingSourceOwner(**options)
    status = service.request(
        "status", {"principal": PRINCIPAL, "requestId": "partial"}, "Bearer owner-token"
    )["receipt"]
    assert status["phase"] == "purged"
    assert status["privateCleanupVerified"] is True
    assert unrelated.read_bytes() == b"Different source"
    assert len(calls) == 1
    service.close()


def test_owner_fence_waits_for_existing_blob_stream_publication(tmp_path):
    """The source owner cannot acknowledge quiescence while a download still emits bytes."""
    from threading import Event, Thread
    from types import SimpleNamespace
    from speech_source_erasure import RecordingSourceOwner

    root, original, blob, source = fixture(tmp_path)
    owner = RecordingSourceOwner(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    preview = owner.request(
        "inventory", {"principal": PRINCIPAL, "source": source}, "Bearer owner-token"
    )["inventory"]
    intent = {
        "principal": PRINCIPAL,
        "requestId": "stream-fence",
        "source": source,
        "expectedFingerprint": preview["fingerprint"],
    }
    started, release, fenced = Event(), Event(), Event()
    output, failures = [], []

    class Transport:
        def write(self, chunk):
            started.set()
            if not release.wait(3):
                raise RuntimeError("bounded synthetic transport deadline")
            output.append(bytes(chunk))

    handler = object.__new__(transcript_api.TranscriptApiHandler)
    handler.server, handler.headers, handler.wfile = (
        SimpleNamespace(store_root=root),
        {},
        Transport(),
    )
    handler.send_response = lambda *_args: None
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None

    def stream():
        try:
            handler.write_blob(blob["id"])
        except Exception as error:
            failures.append(error)

    def fence():
        try:
            owner.request("fence", intent, "Bearer owner-token")
            fenced.set()
        except Exception as error:
            failures.append(error)

    reader = Thread(target=stream)
    writer = Thread(target=fence)
    reader.start()
    try:
        assert started.wait(2)
        writer.start()
        assert not fenced.wait(0.15), (
            "Owner acknowledged quiescence before existing media publication stopped"
        )
    finally:
        release.set()
        reader.join(timeout=3)
        if writer.ident is not None:
            writer.join(timeout=3)
        owner.close()
    assert not failures
    assert fenced.is_set()
    assert b"".join(output) == b"RIFF-synthetic-original"


def test_owner_fence_waits_for_supported_ingest_copy_in_another_process(tmp_path):
    """A watcher/CLI using the same store cannot create a late media copy after owner quiescence."""
    import subprocess
    import sys
    import time
    from threading import Event, Thread
    from speech_source_erasure import RecordingSourceOwner

    root, original, _blob, source = fixture(tmp_path)
    other = original.with_suffix(".flac")
    other.write_bytes(original.read_bytes())
    owner = RecordingSourceOwner(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    inventory = owner.request(
        "inventory", {"principal": PRINCIPAL, "source": source}, "Bearer owner-token"
    )["inventory"]
    intent = {
        "principal": PRINCIPAL,
        "requestId": "ingest-fence",
        "source": source,
        "expectedFingerprint": inventory["fingerprint"],
    }
    ready, release = tmp_path / "copy-ready", tmp_path / "copy-release"
    script = r"""
import sys,time
from pathlib import Path
import transcript_store
root,source,ready,release=map(Path,sys.argv[1:])
native=transcript_store.shutil.copy2
def delayed(*args,**kwargs):
    ready.write_text("copy reached")
    deadline=time.monotonic()+5
    while not release.exists():
        if time.monotonic()>deadline: raise RuntimeError("bounded copy deadline")
        time.sleep(.01)
    return native(*args,**kwargs)
transcript_store.shutil.copy2=delayed
transcript_store.prepare_blob(root,str(source))
"""
    worker = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(other), str(ready), str(release)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fenced, failures = Event(), []

    def fence():
        try:
            result = owner.request("fence", intent, "Bearer owner-token")
            if result["receipt"]["quiescent"]:
                fenced.set()
        except Exception as error:
            failures.append(error)

    thread = Thread(target=fence)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Supported ingest did not reach the bounded copy seam"
        thread.start()
        assert not fenced.wait(0.15), (
            "Process-local lock allowed owner quiescence before a separate supported ingest copy stopped"
        )
    finally:
        release.write_text("resume copy")
        stdout, stderr = worker.communicate(timeout=5)
        if thread.ident is not None:
            thread.join(timeout=5)
        owner.close()
    assert worker.returncode == 0, (stdout, stderr)
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "preview_stale" in str(failures[0])
    assert not fenced.is_set()
    canonical_copy = transcript_store.blob_store_path(root, _blob["id"], other)
    assert canonical_copy.read_bytes() == original.read_bytes()
    assert Path(_blob["stored_path"]).exists()


def test_speech_original_owner_client_uses_actual_authenticated_ta_contract(tmp_path):
    import os
    import subprocess
    from threading import Thread
    from speech_source_erasure import RecordingSourceOwner

    module = os.environ.get("TRANSCRIBE_SPEECH_INTEGRATION_MODULE")
    if not module:
        pytest.skip(
            "Opt-in actual Speech Node library; default lane needs no sibling checkout"
        )
    root, original, blob, source = fixture(tmp_path)
    service = RecordingSourceOwner(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[original.parent],
    )
    server = transcript_api.TranscriptApiServer(
        ("127.0.0.1", 0),
        transcript_api.TranscriptApiHandler,
        store_root=root,
        state_root=tmp_path / "state",
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
        quiet=True,
        static_dir=None,
    )
    server.speech_source_owner = service
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = tmp_path / "owner-client.mjs"
    script.write_text(f"""import assert from 'node:assert/strict';import {{transcribeSourceOwner}} from {json.dumps(module)};
const owner={json.dumps(OWNER)},source={json.dumps(source)},principal={json.dumps(PRINCIPAL)};
const grant={{...principal,audience:'speech-receipts',expiresAt:4102444800000,scopes:['speech:readiness']}};
const client=transcribeSourceOwner({{baseUrl:'http://127.0.0.1:{server.server_port}',token:'owner-token',owner,principals:[principal]}});
await assert.rejects(client.sourceInventory({{...grant,tenant:'other'}},source));
const preview=await client.sourceInventory(grant,source);assert.equal(preview.available,true);assert.equal(preview.items.length,2);
const intent={{requestId:'actual-client',source,expectedFingerprint:preview.fingerprint}};
assert.equal((await client.sourceFence(grant,intent)).quiescent,true);
assert.equal((await client.sourceStatus(grant,intent.requestId)).phase,'fenced');
assert.equal((await client.purgeSource(grant,intent)).privateCleanupVerified,true);
assert.equal((await client.sourceStatus(grant,intent.requestId)).phase,'purged');
console.log('actual owner client passed');""")
    try:
        result = subprocess.run(
            ["node", str(script)],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        assert result.stdout.strip() == "actual owner client passed"
        assert not original.exists()
        assert not Path(blob["stored_path"]).exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_optional_owner_does_not_break_ordinary_media_prepare_without_posix_identity(
    tmp_path, monkeypatch
):
    import os
    from speech_source_erasure import RecordingSourceOwner

    root, original, _, _ = fixture(tmp_path)
    with monkeypatch.context() as patch:
        patch.delattr(os, "getuid")
        assert transcript_store.prepare_blob(root, str(original))["bytes"] == len(
            b"RIFF-synthetic-original"
        )
        with pytest.raises(ValueError, match="platform_unavailable"):
            RecordingSourceOwner(
                store_root=root,
                directory=tmp_path / "owner",
                owner=OWNER,
                token="owner-token",
                principals=[PRINCIPAL],
                original_roots=[original.parent],
            )


def test_owner_binds_container_original_digest_and_rejects_its_reingestion_before_creating_new_media(
    tmp_path,
):
    from speech_source_erasure import RecordingSourceOwner

    root, working, blob, source = fixture(tmp_path)
    container = working.with_suffix(".m4a")
    container_bytes = b"Different container encoding of the registered recording"
    container.write_bytes(container_bytes)
    artifact = tmp_path / "meeting.transcript.json"
    payload = json.loads(artifact.read_text())
    payload["source_media_path"] = str(container)
    artifact.write_text(json.dumps(payload))
    transcript_store.ingest_artifact(
        artifact,
        root=root,
        embedding_provider="debug-hash",
        embedding_model="debug-hash",
    )
    service = RecordingSourceOwner(
        store_root=root,
        directory=tmp_path / "owner",
        owner=OWNER,
        token="owner-token",
        principals=[PRINCIPAL],
        original_roots=[working.parent],
    )
    preview = service.request(
        "inventory", {"principal": PRINCIPAL, "source": source}, "Bearer owner-token"
    )["inventory"]
    assert preview["available"] is True
    assert len(preview["items"]) == 3
    intent = {
        "principal": PRINCIPAL,
        "requestId": "container",
        "source": source,
        "expectedFingerprint": preview["fingerprint"],
    }
    service.request("fence", intent, "Bearer owner-token")
    service.request("purge", intent, "Bearer owner-token")
    assert not container.exists()
    assert not working.exists()
    service.close()
    backup = tmp_path / "backup.m4a"
    backup.write_bytes(container_bytes)
    processed = tmp_path / "new-working.wav"
    processed.write_bytes(b"A new processed representation")
    attempt = tmp_path / "attempt.transcript.json"
    attempt.write_text(
        json.dumps(
            {
                **payload,
                "source_media_path": str(backup),
                "working_media_path": str(processed),
            }
        )
    )
    expected = transcript_store.blob_store_path(
        root,
        transcript_store.stable_id(
            "source_recording", hashlib.sha256(processed.read_bytes()).hexdigest()
        ),
        processed,
    )
    with pytest.raises(transcript_store.TranscriptStoreError, match="erased"):
        transcript_store.ingest_artifact(
            attempt,
            root=root,
            embedding_provider="debug-hash",
            embedding_model="debug-hash",
        )
    assert not expected.exists()
