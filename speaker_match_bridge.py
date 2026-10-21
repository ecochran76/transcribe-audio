"""Read-side comparison against exact scoped reviewed enrollment heads.

Scores are uncalibrated evidence, never identity assignments or enrollment.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
from typing import Any, Mapping

from acoustic_verification import VerificationAdapter, adapter_registry, cosine_score
from speaker_enrollment_bridge import (
    EnrollmentBridge,
    EnrollmentError,
    MAX_INPUT,
    _hash,
    _id,
    _json,
    _normalized,
    _private,
)

SCHEMA = "transcribe-audio.speaker-comparison-process.v1"


def _fail(code: str) -> None:
    raise EnrollmentError("comparison_" + code)


class ComparisonBridge:
    """Injectable test seam; production CLI uses only the genuine pinned registry."""

    def __init__(self, config: Mapping[str, Any], adapter: VerificationAdapter):
        self.config = json.loads(_json(config))
        root = Path(str(self.config.get("directory", "")))
        _private(root)
        # This read-side boundary cannot provision a different enrollment store.
        for name in ("bridge-owner.json", "bridge.sqlite"):
            item = root / name
            if item.is_symlink() or not item.is_file():
                _fail("existing_enrollment_store_required")
        self.enrollment = EnrollmentBridge(self.config, adapter)
        self.adapter = adapter

    def close(self) -> None:
        self.enrollment.close()

    def _profile(
        self, app: str, pin: Mapping[str, Any]
    ) -> tuple[dict[str, Any], tuple[float, ...]]:
        if not isinstance(pin, dict) or not isinstance(pin.get("person"), dict):
            _fail("invalid_profile")
        person = pin["person"]
        _id(person.get("id"))
        _id(pin.get("profile_version_id"))
        if not isinstance(person.get("revision"), str) or not person["revision"]:
            _fail("invalid_profile")
        sha = pin.get("profile_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
            _fail("invalid_profile")
        if (
            pin.get("adapter_id") != self.adapter.candidate_id
            or pin.get("model_revision") != self.adapter.revision_sha
        ):
            _fail("model_drift")
        reviewed = self.enrollment.config["authorities"].get(person["id"], {})
        if reviewed.get("person_revision") != person["revision"]:
            _fail("authority_drift")
        row = self.enrollment.db.execute(
            "SELECT jobs.payload FROM heads JOIN jobs ON jobs.id=heads.job WHERE heads.person=? AND heads.profile=? AND jobs.app=?",
            (person["id"], pin["profile_version_id"], app),
        ).fetchone()
        if row is None:
            _fail("profile_not_current")
        job = json.loads(row[0])
        if job.get("state") != "completed" or any(
            job.get(key) != pin.get(key)
            for key in (
                "person",
                "profile_version_id",
                "profile_sha256",
                "adapter_id",
                "model_revision",
            )
        ):
            _fail("profile_drift")
        state = self.enrollment.custody.profile_state(pin["profile_version_id"])
        if state["status"] != "active" or state["person_id"] != person["id"]:
            _fail("profile_not_active")
        sample_states = []
        for sample_id in state["sample_ids"]:
            sample = self.enrollment.custody.load_sample(sample_id)
            if self.enrollment._reference_blocked(
                sample["preparation_lineage"].get("source_sample_id", ""),
                sample["sample_sha256"],
            ):
                _fail("profile_sample_unavailable")
            sample_state = self.enrollment.custody.sample_state(sample_id)
            if sample_state["exclusion_state"] != "included":
                _fail("profile_sample_unavailable")
            if (
                sample["review_authority_id"] != reviewed.get("review_id")
                or sample["consent_authority"] != reviewed.get("enrollment_id")
                or sample["person_id"] != person["id"]
            ):
                _fail("authority_drift")
            sample_states.append(sample_state)
        obj = self.enrollment.private_root / "objects" / ("profile-" + sha)
        metadata = obj.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 65_536
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("invalid_profile_object")
        raw = obj.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            _fail("profile_hash_mismatch")
        payload = json.loads(raw)
        if (
            payload.get("adapter_id") != pin["adapter_id"]
            or payload.get("model_revision") != pin["model_revision"]
            or payload.get("person") != person
            or payload.get("sample_manifest_sha256") != job["sample_manifest_sha256"]
        ):
            _fail("profile_object_mismatch")
        return {
            "job": job,
            "state": state,
            "sample_states": sample_states,
            "authority": dict(reviewed),
        }, _normalized(payload.get("centroid", []), self.adapter.embedding_dimension)

    def request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            _fail("invalid_request")
        app = self.enrollment._authorize(request)
        action = request.get("action")
        if action in {"readiness", "profiles"}:
            return self.enrollment.request(request)
        if action != "compare":
            _fail("invalid_request")
        job_id = _id(request.get("job_id"))
        if not isinstance(request.get("samples"), list) or len(request["samples"]) != 1:
            _fail("one_probe_required")
        manifest, decoded = self.enrollment._samples(request)
        profiles = request.get("profiles")
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= 32:
            _fail("profile_limits")
        if any(not isinstance(pin, dict) for pin in profiles):
            _fail("invalid_profile")
        if len({pin.get("profile_version_id") for pin in profiles}) != len(profiles):
            _fail("duplicate_profile")
        checked = [self._profile(app, pin) for pin in profiles]
        with self.enrollment._worker_lock():
            # Revalidate after acquiring ownership; an enrollment may have promoted.
            if [self._profile(app, pin)[0] for pin in profiles] != [
                value[0] for value in checked
            ]:
                _fail("profile_changed")
            vector = _normalized(
                self.adapter.embed(decoded[0][2], sample_rate=16_000),
                self.adapter.embedding_dimension,
            )
            candidates = []
            for pin, (snapshot, centroid) in zip(profiles, checked):
                if self._profile(app, pin)[0] != snapshot:
                    _fail("profile_changed")
                candidates.append(
                    {
                        **{
                            key: pin[key]
                            for key in (
                                "person",
                                "profile_version_id",
                                "profile_sha256",
                                "adapter_id",
                                "model_revision",
                            )
                        },
                        "score": cosine_score(centroid, vector),
                        "score_kind": "uncalibrated_cosine",
                        "qualified": False,
                    }
                )
            return {
                "id": job_id,
                "schema_version": SCHEMA,
                "sample_manifest": manifest,
                "sample_manifest_sha256": _hash(manifest),
                "candidates": candidates,
                "qualified": False,
                "identity_assigned": False,
                "enrollment_performed": False,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    bridge = None
    try:
        if sys.platform != "linux" or args.parent_pid < 1:
            _fail("parent_guard_unavailable")
        if (
            ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0
            or os.getppid() != args.parent_pid
        ):
            _fail("parent_guard_unavailable")
        if (
            args.config.is_symlink()
            or not args.config.is_file()
            or stat.S_IMODE(args.config.stat().st_mode) & 0o077
        ):
            _fail("invalid_configuration")
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        raw = sys.stdin.buffer.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            _fail("input_limit")
        request = json.loads(raw)
        if (
            not isinstance(request, dict)
            or request.get("tenant") != config.get("tenant")
            or request.get("app") not in config.get("apps", [])
        ):
            _fail("forbidden")
        adapter = adapter_registry(snapshot_root=Path(config["model_snapshot_root"]))[
            config["adapter_id"]
        ]
        bridge = ComparisonBridge(config, adapter)
        with contextlib.redirect_stdout(sys.stderr):
            result = bridge.request(request)
        if args.config.read_bytes() != config_bytes:
            _fail("configuration_changed")
        print(_json({"ok": True, "result": result}))
        return 0
    except Exception as exc:
        code = (
            str(exc)
            if isinstance(exc, EnrollmentError)
            else "comparison_request_failed"
        )
        print(_json({"ok": False, "error": code}))
        return 1
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
