from __future__ import annotations

import pytest


def test_consumer_requires_explicit_owner_and_supported_endpoint(tmp_path):
    from speech_harvesting import SpeechHarvestConsumer

    with pytest.raises(ValueError, match="speech_owner_required"):
        SpeechHarvestConsumer(
            directory=tmp_path / "bridge",
            speech_url="http://127.0.0.1:1234",
            token="fixture",
            source_token="source",
            owner={},
            resolve=lambda _: {},
        )


import hashlib
import json
import math
import os
import select
import struct
import subprocess
import threading
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import transcript_api
import transcript_store
from speech_harvesting import SpeechHarvestConsumer, normalize_diarization

OWNER = {"tenant": "fixture", "app": "meeting-app", "account": "fixture-recordings"}
SPEECH_MODULE = (
    Path(os.environ["TRANSCRIBE_SPEECH_INTEGRATION_MODULE"])
    if os.environ.get("TRANSCRIBE_SPEECH_INTEGRATION_MODULE")
    else None
)


def test_exporter_units_are_explicit_and_never_guessed():
    payload = {
        "backend": "faster_whisper",
        "utterances": [
            {
                "speaker": "Mixed A",
                "start": 0,
                "end": 2000,
                "words": [{"start": 0.3, "end": 1.1}],
            }
        ],
    }
    result = normalize_diarization(payload, document_revision="r1")
    assert result["words"] == [{"speaker": "Mixed A", "startMs": 300, "endMs": 1100}]
    assert result["units"] == "ms"
    payload["backend"] = "unknown"
    assert "words" not in normalize_diarization(payload, document_revision="r1")


@contextmanager
def integration(tmp_path):
    if SPEECH_MODULE is None:
        pytest.skip(
            "Opt-in actual cross-repository lane: set TRANSCRIBE_SPEECH_INTEGRATION_MODULE"
        )
    if not SPEECH_MODULE.is_absolute() or not SPEECH_MODULE.is_file():
        raise AssertionError(
            "Configured Speech integration module must be an existing absolute build path"
        )
    media = tmp_path / "meeting.wav"
    with wave.open(str(media), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(
            b"".join(
                struct.pack("<h", round(5000 * math.sin(i * 0.16)))
                for i in range(128000)
            )
        )
    artifact = tmp_path / "meeting.transcript.json"
    artifact.write_text(
        json.dumps(
            {
                "transcript_title": "Synthetic meeting",
                "backend": "assembly",
                "source_media_path": str(media),
                "working_media_path": str(media),
                "transcript_text": "Fixture words only.",
                "utterances": [
                    {
                        "speaker": "A",
                        "start": 0,
                        "end": 8000,
                        "words": [
                            {"start": 300, "end": 1000},
                            {"start": 1200, "end": 3900},
                            {"start": 4200, "end": 7800},
                        ],
                    }
                ],
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
    ta_url = "http://127.0.0.1:" + str(server.server_address[1])
    script = tmp_path / "speech.mjs"
    script.write_text(f"""import * as s from {json.dumps(str(SPEECH_MODULE))};
const owner={json.dumps(OWNER)},grant={{...owner,audience:'speech-receipts',expiresAt:4102444800000,scopes:['speech:readiness','speech:harvest','speech:transcribe']}};
const directory={json.dumps(str(tmp_path / "speech"))};
const saved=s.openSavedArtifacts({{directory:directory+'/saved',operators:[owner],principals:[owner]}});
const harvest=s.openSpeakerHarvesting({{directory:directory+'/harvest',operators:[owner],conversion:{{ffmpeg:false}},readSource:s.transcribeHarvestingSource({{baseUrl:{json.dumps(ta_url)},token:'source-fixture',bindings:[owner]}})}});
const samples=s.openSpeakerSamples({{directory:directory+'/samples',operators:[owner],savedArtifacts:saved,harvesting:harvest}});
const associations=s.openSampleAssociations({{directory:directory+'/associations',operators:[owner],speakers:samples,contacts:{{search:()=>Promise.resolve([]),person:()=>Promise.reject(Error('not needed')),create:()=>Promise.reject(Error('not permitted'))}},quiesce:()=>Promise.resolve({{quiescent:true,evidence:['synthetic no workers']}})}});
const service=await s.startHttp(0,{{authenticate:h=>h==='Bearer speech-fixture'?grant:undefined,core:{{close:()=>Promise.resolve()}},harvesting:harvest,speakerSamples:samples,sampleAssociations:associations}});
console.log(JSON.stringify({{url:service.url}}));process.once('SIGTERM',async()=>{{await service.close();associations.close();samples.close();await harvest.close();saved.close();}});
""")
    child = subprocess.Popen(
        ["node", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    thread = None
    try:
        if not select.select([child.stdout], [], [], 5)[0]:
            raise AssertionError("Speech fixture did not become ready")
        speech_url = json.loads(child.stdout.readline())["url"]
        bridge = SpeechHarvestConsumer(
            directory=tmp_path / "consumer",
            speech_url=speech_url,
            token="speech-fixture",
            source_token="source-fixture",
            owner=OWNER,
            resolve=lambda identifier: transcript_api.speech_source_for_conversation(
                identifier, root=root, state_root=tmp_path / "state", owner=OWNER
            ),
            allowed_origins=(ta_url,),
            observation_token="observation-fixture",
        )
        server.speech_harvesting = bridge
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield {
            "bridge": bridge,
            "server": server,
            "url": ta_url,
            "document": document.id,
            "root": root,
            "media": media,
            "speech": speech_url,
            "directory": tmp_path,
        }
    finally:
        if thread:
            server.shutdown()
        server.server_close()
        if thread:
            thread.join(timeout=5)
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
        if child.returncode not in {0, -15}:
            raise AssertionError(
                "Owned Speech fixture failed: " + child.stderr.read()[-500:]
            )


def request(url, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(
        Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        ),
        timeout=5,
    ) as response:
        return json.load(response)


def wait_for(read, states):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = read()
        if value["state"] in states:
            return value
        time.sleep(0.01)
    raise AssertionError("Public journal never reached a terminal state")


def test_real_cross_service_harvesting_retention_and_proposals_preserve_meeting_authority(
    tmp_path,
):
    with integration(tmp_path) as f:
        base = f["url"] + "/api/conversations/" + f["document"] + "/speech-harvest"
        context = request(base)
        source = context["prepared"]
        intent = {
            "requestId": "meeting-harvest",
            "sourceSha256": source["source"]["sha256"],
            "sourceRevision": source["source"]["revision"],
            "diarizationSha256": source["diarizationSha256"],
        }
        with pytest.raises(HTTPError) as denied:
            request(base, intent, {"Origin": "https://foreign.example"})
        assert denied.value.code == 403
        started = request(base, intent)
        result = wait_for(
            lambda: request(base + "/meeting-harvest"), {"completed", "failed"}
        )
        assert result["state"] == "completed", result
        assert result["speechRequestId"] == started["speechRequestId"]
        assert result["result"]["source"]["service"] == "transcribe-audio"
        candidate = result["result"]["candidates"][0]
        audio = request(base + "/meeting-harvest/candidates/" + candidate["id"])
        assert audio["mediaType"] == "audio/wav"
        retained = request(
            base + "/meeting-harvest/retain",
            {
                "requestId": "keep-one",
                "candidateId": candidate["id"],
                "reviewed": True,
                "name": "Meeting A",
            },
        )
        assert retained["state"] == "applied", retained
        proposed = request(
            base + "/keep-one/propose",
            {
                "requestId": "proposal-one",
                "target": {
                    "kind": "entity",
                    "id": "11111111-1111-4111-8111-111111111111",
                    "revision": "person-r1",
                },
                "evidence": "Operator supplied reference; proposal only",
            },
        )
        assert proposed["state"] == "proposed", proposed
        assert proposed["sampleId"] == retained["sampleId"]
        references = request(
            f["url"] + "/api/speech-harvest/references",
            headers={"Authorization": "Bearer observation-fixture"},
        )
        assert len(references["references"]) == 1
        reference = references["references"][0]
        assert reference["sourceId"] == result["source"]["id"]
        assert reference["diarizationSha256"] == result["result"]["diarizationSha256"]
        observation = {
            key: reference[key]
            for key in (
                "sampleId",
                "sampleRevision",
                "associationRevision",
                "sourceId",
                "sourceRevision",
                "sourceSha256",
                "diarizationSha256",
                "person",
            )
        }
        observation.update(
            requestId="observed-one",
            actor={"tenant": OWNER["tenant"], "app": "speech-operator"},
            evidence="Observed current Speech sample state",
        )
        observed = request(
            base + "/meeting-harvest/observe",
            observation,
            {"Authorization": "Bearer observation-fixture"},
        )
        assert observed["state"] == "proposed"
        assert (
            request(
                base + "/observed-one",
                headers={"Authorization": "Bearer observation-fixture"},
            )
            == observed
        )
        assert (
            request(
                base + "/observed-one/reconcile",
                {},
                {"Authorization": "Bearer observation-fixture"},
            )
            == observed
        )
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "Bearer source-fixture"},
        ):
            with pytest.raises(HTTPError) as denied:
                request(base + "/observed-one", headers=headers)
            assert denied.value.code == 403
        with pytest.raises(HTTPError) as foreign:
            request(
                f["url"] + "/api/conversations/foreign/speech-harvest/observed-one",
                headers={"Authorization": "Bearer observation-fixture"},
            )
        assert foreign.value.code == 403
        conflict = request(
            base + "/meeting-harvest/observe",
            {
                **observation,
                "requestId": "observed-conflict",
                "person": proposed["result"]["target"],
            },
            {"Authorization": "Bearer observation-fixture"},
        )
        assert conflict["state"] == "conflict"
        after = request(
            f["url"] + "/api/speech-harvest/references",
            headers={"Authorization": "Bearer observation-fixture"},
        )
        assert after["revision"] != references["revision"]
        assert after["references"][0]["state"] == "conflict"
        assert len(after["references"][0]["conflictingObservationIds"]) == 2
        snapshot = transcript_api.get_conversation_detail(
            f["document"], root=f["root"], state_root=f["directory"] / "state"
        )
        assert snapshot["identity_review"]["pending_count"] > 0
        assert not list((f["directory"] / "consumer").glob("*.wav"))


def test_outage_unknown_restart_source_drift_and_cancellation_never_resubmit(tmp_path):
    with integration(tmp_path) as f:
        bridge = f["bridge"]
        context = bridge.prepare(f["document"])
        intent = {
            "requestId": "outage",
            "documentId": f["document"],
            "sourceSha256": context["source"]["sha256"],
            "sourceRevision": context["source"]["revision"],
            "diarizationSha256": context["diarizationSha256"],
        }
        real_url = bridge.url
        bridge.url = "http://127.0.0.1:1/"
        unknown = bridge.start(intent)
        assert unknown["state"] == "unknown"
        bridge.url = real_url
        assert bridge.start(intent) == unknown
        assert bridge.status("outage")["error"] == "speech_not_found"
        with pytest.raises(ValueError, match="speech_source_forbidden"):
            bridge.export_source("outage", "Bearer foreign")
        exported = bridge.export_source("outage", "Bearer source-fixture")
        assert exported["owner"] == OWNER
        assert exported["source"]["operationId"] == "outage"
        assert (
            exported["source"]["sha256"]
            == hashlib.sha256(f["media"].read_bytes()).hexdigest()
        )
        current = transcript_api.get_conversation_detail(
            f["document"], root=f["root"], state_root=f["directory"] / "state"
        )
        blob = transcript_api.get_blob(current["media_blob"]["id"], root=f["root"])
        stored = Path(blob["path"])
        original = stored.read_bytes()
        stored.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(ValueError, match="speech_source_changed"):
            bridge.export_source("outage", "Bearer source-fixture")
        stored.write_bytes(original)
        live = bridge.start({**intent, "requestId": "actual"})
        result = wait_for(lambda: bridge.status("actual"), {"completed", "failed"})
        assert result["state"] == "completed", result
        candidate = result["result"]["candidates"][0]
        retained = bridge.retain(
            "actual",
            {"requestId": "retained", "candidateId": candidate["id"], "reviewed": True},
        )
        assert retained["state"] == "applied"
        assert bridge.cancel("actual")["state"] == "cancelled"
        with pytest.raises(ValueError, match="speech_source_unavailable"):
            bridge.export_source("actual", "Bearer source-fixture")
        bridge.close()
        restarted = SpeechHarvestConsumer(
            directory=f["directory"] / "consumer",
            speech_url=f["speech"],
            token="speech-fixture",
            source_token="source-fixture",
            owner=OWNER,
            resolve=bridge.resolve,
        )
        f["server"].speech_harvesting = restarted
        assert restarted.start(intent)["state"] == "unknown"
        assert (
            restarted.reconcile_retention("retained")["sampleId"]
            == retained["sampleId"]
        )
        assert stored.read_bytes() == original


def test_default_consumer_uses_authenticated_http_and_reconciles_without_node_or_peer_checkout(
    tmp_path,
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    captured = []
    remote = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer fixture":
                self.send_error(403)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append(body)
            remote.update(requestId=body["requestId"], state="queued", candidates=[])
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(remote).encode())

        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer fixture":
                self.send_error(403)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({**remote, "state": "completed"}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    media = tmp_path / "owned.audio"
    media.write_bytes(b"synthetic media for transport only")
    sha = hashlib.sha256(media.read_bytes()).hexdigest()
    timeline = {
        "units": "ms",
        "revision": "r1",
        "turns": [{"speaker": "A", "startMs": 0, "endMs": 3000}],
    }
    resolved = {
        "owner": OWNER,
        "source": {
            "id": "recording",
            "service": "transcribe-audio",
            "account": OWNER["account"],
            "revision": sha,
            "sha256": sha,
            "stream": 0,
            "channelPolicy": "mix-to-mono",
        },
        "path": media,
        "diarization": timeline,
    }
    bridge = SpeechHarvestConsumer(
        directory=tmp_path / "consumer",
        speech_url="http://127.0.0.1:" + str(server.server_address[1]),
        token="fixture",
        source_token="source",
        owner=OWNER,
        resolve=lambda _: resolved,
    )
    try:
        prepared = bridge.prepare("meeting")
        intent = {
            "requestId": "one",
            "documentId": "meeting",
            "sourceSha256": sha,
            "sourceRevision": sha,
            "diarizationSha256": prepared["diarizationSha256"],
        }
        assert bridge.start(intent)["state"] == "queued"
        assert bridge.start(intent)["state"] == "queued"
        assert len(captured) == 1
        assert captured[0]["source"]["kind"] == "application"
        assert captured[0]["source"]["operationId"] == "one"
        assert "audioBase64" not in captured[0]
        assert bridge.status("one")["state"] == "completed"
        assert len(captured) == 1
        assert not list((tmp_path / "consumer").glob("*.audio"))
    finally:
        bridge.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def lost_ack_proxy(upstream):
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    writes = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def forward(self):
            data = (
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.command == "POST"
                else None
            )
            with urlopen(
                Request(
                    upstream + self.path,
                    data=data,
                    method=self.command,
                    headers={
                        "Authorization": self.headers.get("Authorization", ""),
                        "Content-Type": "application/json",
                    },
                ),
                timeout=5,
            ) as response:
                payload, status = response.read(), response.status
            if self.command == "POST":
                writes.append(self.path)
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        do_GET = forward
        do_POST = forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_address[1]) + "/", writes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_actual_cross_service_lost_acknowledgements_reconcile_without_repeated_effects(
    tmp_path,
):
    with integration(tmp_path) as f, lost_ack_proxy(f["speech"]) as (proxy, writes):
        bridge = f["bridge"]
        bridge.url = proxy
        prepared = bridge.prepare(f["document"])
        intent = {
            "requestId": "lost-harvest",
            "documentId": f["document"],
            "sourceSha256": prepared["source"]["sha256"],
            "sourceRevision": prepared["source"]["revision"],
            "diarizationSha256": prepared["diarizationSha256"],
        }
        assert bridge.start(intent)["state"] == "unknown"
        result = wait_for(
            lambda: bridge.status("lost-harvest"), {"completed", "failed"}
        )
        assert result["state"] == "completed", result
        assert bridge.start(intent)["state"] == "completed"
        candidate = result["result"]["candidates"][0]
        keep = {
            "requestId": "lost-keep",
            "candidateId": candidate["id"],
            "reviewed": True,
        }
        assert bridge.retain("lost-harvest", keep)["state"] == "unknown"
        retained = bridge.reconcile_retention("lost-keep")
        assert retained["state"] == "applied"
        assert bridge.retain("lost-harvest", keep) == retained
        proposal = {
            "requestId": "lost-proposal",
            "target": {
                "kind": "entity",
                "id": "11111111-1111-4111-8111-111111111111",
                "revision": "r1",
            },
            "evidence": "Explicit attributed proposal only",
        }
        assert bridge.propose("lost-keep", proposal)["state"] == "unknown"
        assert bridge.status("lost-proposal")["state"] == "proposed"
        assert len(writes) == 3
        assert writes == [
            "/v1/harvesting",
            "/v1/speakers/harvest-retain",
            "/v1/sample-associations/proposals",
        ]


def test_missing_meeting_is_a_sanitized_unavailable_source(tmp_path):
    with pytest.raises(ValueError, match="^speech_recording_unavailable$"):
        transcript_api.speech_source_for_conversation(
            "missing",
            root=tmp_path / "store",
            state_root=tmp_path / "state",
            owner=OWNER,
        )
