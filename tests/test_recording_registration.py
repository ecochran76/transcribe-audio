from hashlib import sha256
from http.client import HTTPConnection
import json
from threading import Thread

from transcript_api import TranscriptApiHandler, TranscriptApiServer


def test_authority_admits_exact_bytes_once_and_reads_them_back(tmp_path):
    token = "fixture-token-01234567890123456789"
    token_file = tmp_path / "token"
    token_file.write_text(token)
    token_file.chmod(0o600)
    transcript = b' {"transcript_text":"Verbatim text","source_media_path":"/foreign/raw.wav"}\n'
    recording = b"exact original recording bytes"
    server = TranscriptApiServer(
        ("127.0.0.1", 0), TranscriptApiHandler,
        store_root=tmp_path / "store",
        embedding_provider="debug-hash", embedding_model="debug-hash",
        recording_registration_token_file=token_file,
        quiet=True,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "X-Transcript-Bytes": str(len(transcript)),
        "X-Transcript-Sha256": sha256(transcript).hexdigest(),
        "X-Recording-Sha256": sha256(recording).hexdigest(),
        "X-Recording-Extension": ".wav",
    }

    def request(path, method="GET", body=None, headers=None):
        connection = HTTPConnection(*server.server_address, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        route = "/api/recording-registrations"
        packet = transcript + recording
        unauthorized = {key: value for key, value in headers.items() if key != "Authorization"}
        assert request(route, "POST", packet, unauthorized)[0] == 403
        assert request(route, "POST", packet + b"changed", headers)[0] == 422
        assert json.loads(request("/api/library")[1])["total"] == 0

        status, response = request(route, "POST", packet, headers)
        assert status == 201
        assert request(route, "POST", packet, headers) == (200, response)
        receipt = json.loads(response)
        document_id = receipt["native_document_id"]
        blob_id = receipt["native_recording_id"]
        assert receipt["transcript_sha256"] == headers["X-Transcript-Sha256"]
        assert receipt["recording_sha256"] == headers["X-Recording-Sha256"]

        assert json.loads(request("/api/library")[1])["total"] == 1
        status, result = request(f"/api/documents/{document_id}")
        assert status == 200
        document = json.loads(result)
        assert document["artifact_sha256"] == receipt["transcript_sha256"]
        assert any(blob["id"] == blob_id and blob["role"] == "source_recording"
                   and blob["sha256"] == receipt["recording_sha256"] for blob in document["blobs"])
        assert request(f"/api/blobs/{blob_id}") == (200, recording)
        download = f"{route}/{document_id}/transcript"
        assert request(download)[0] == 403
        assert request(download, headers={"Authorization": headers["Authorization"]}) == (200, transcript)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
