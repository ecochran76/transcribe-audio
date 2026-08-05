from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import subprocess
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import acoustic_generation3_recalibration as recalibration
import acoustic_generation5_e2 as generation5_e2
import acoustic_generation4_acoustic_contract as acoustic_contract
import acoustic_plan0056_execution as authority
import acoustic_plan0056_pilot as pilot
import acoustic_verification as verification
from acoustic_audio_derivatives import (
    ensure_private_tree,
    read_private_object,
    require_private_file,
    write_immutable_private_json,
)


EXECUTION_SCHEMA = "transcribe-audio.plan0056-local-pilot-execution.v1"
EXECUTION_REPLAY_SCHEMA = "transcribe-audio.plan0056-local-pilot-execution-replay.v1"
MATRIX_SCHEMA = "transcribe-audio.plan0056-acoustic-matrix.v1"
DEFAULT_RUNTIME_ROOT = Path("~/.local/state/transcribe-audio/plan-0056/p1")

class Plan0056RunnerError(ValueError):
    """Raised when the local acoustic pilot execution cannot remain complete."""


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_review_segments(
    timeline: Sequence[Mapping[str, Any]],
    *,
    minimum_turn_seconds: float,
    maximum_turn_seconds: float,
    maximum_turns_per_speaker: int,
    target_seconds_per_speaker: float,
    minimum_usable_seconds_per_speaker: float,
) -> dict[str, list[dict[str, float]]]:
    """Select bounded review audio in first-appearance speaker order."""

    normalized = []
    for raw in timeline:
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Plan0056RunnerError("A diarization turn is invalid.") from exc
        speaker = str(raw.get("speaker") or "")
        if not speaker or start < 0 or end <= start:
            raise Plan0056RunnerError("A diarization turn is invalid.")
        normalized.append((start, end, speaker))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    if not normalized:
        raise Plan0056RunnerError("Local diarization returned no speaker turns.")

    speaker_map: dict[str, str] = {}
    turns: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for start, end, raw_speaker in normalized:
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = f"SPEAKER_{len(speaker_map) + 1}"
        if end - start >= minimum_turn_seconds:
            turns[speaker_map[raw_speaker]].append(
                (start, min(end, start + maximum_turn_seconds))
            )

    selected: dict[str, list[dict[str, float]]] = {}
    for speaker in speaker_map.values():
        total = 0.0
        chosen = []
        for start, end in turns.get(speaker, []):
            if len(chosen) == maximum_turns_per_speaker or total >= target_seconds_per_speaker:
                break
            chosen.append({"start": start, "end": end})
            total += end - start
        if total < minimum_usable_seconds_per_speaker:
            raise Plan0056RunnerError(
                f"{speaker} has insufficient usable diarized speech for review."
            )
        selected[speaker] = chosen
    return selected


def _execution_paths(runtime_root: Path, authority_sha256: str) -> dict[str, Path]:
    root = runtime_root.expanduser().absolute()
    authority_paths = authority._authority_paths(root, authority_sha256)
    run = authority_paths["run"] / "local-pilot-execution"
    return {
        "root": root,
        "authority_run": authority_paths["run"],
        "run": run,
        "manifest": run / "private-manifest.json",
        "receipt": run / "receipt.json",
        "pcm": run / "source-pcm.wav",
        "diarization": run / "diarization.json",
        "clips": run / "clips",
        "transcripts": run / "transcripts",
        "p1": run / "preparation-p1",
        "p2": run / "preparation-p2",
        "matrices": run / "matrices",
    }


def _read_frozen_preview(paths: Mapping[str, Path]) -> dict[str, Any]:
    require_private_file(paths["authority_run"] / "private-manifest.json", paths["root"])
    manifest = read_private_object(paths["authority_run"] / "private-manifest.json")
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise Plan0056RunnerError("Frozen local execution authority is invalid.")
    return preview


def _read_p0_preview() -> dict[str, Any]:
    paths = pilot._authority_paths(pilot.DEFAULT_RUNTIME_ROOT, authority.P0_CONTENT_SHA256)
    require_private_file(paths["manifest"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    preview = manifest.get("preview")
    if not isinstance(preview, dict):
        raise Plan0056RunnerError("Frozen P0 authority is invalid.")
    return preview


def _decode_private_pcm(source: Path, output: Path) -> None:
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(output),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode or not output.is_file():
        raise Plan0056RunnerError("The frozen source could not be decoded locally.")
    output.chmod(0o600)


def _read_pcm16_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise Plan0056RunnerError("Pilot audio must be mono PCM16.")
        sample_rate = audio.getframerate()
        samples = array.array("h", audio.readframes(audio.getnframes()))
    return [value / 32768.0 for value in samples], sample_rate


def _run_local_diarization(
    pcm_path: Path,
    *,
    model_root: Path,
    minimum_speakers: int,
    maximum_speakers: int,
) -> list[dict[str, Any]]:
    import torch
    from pyannote.audio import Pipeline

    samples, sample_rate = _read_pcm16_mono(pcm_path)
    configured_root = model_root.expanduser().absolute()
    pipeline_root = configured_root / "community-1"
    if not (pipeline_root / "config.yaml").is_file():
        raise Plan0056RunnerError("The frozen local diarization model is incomplete.")
    previous = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        pipeline = Pipeline.from_pretrained(str(pipeline_root))
        result = pipeline(
            {"waveform": torch.tensor(samples).unsqueeze(0), "sample_rate": sample_rate},
            min_speakers=minimum_speakers,
            max_speakers=maximum_speakers,
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
    return [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]


def _write_speaker_clip(
    source_pcm: Path,
    output: Path,
    segments: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    with wave.open(str(source_pcm), "rb") as source:
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        chunks = []
        for segment in segments:
            start_frame = max(0, min(total_frames, round(float(segment["start"]) * sample_rate)))
            end_frame = max(start_frame, min(total_frames, round(float(segment["end"]) * sample_rate)))
            source.setpos(start_frame)
            chunks.append(source.readframes(end_frame - start_frame))
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"".join(chunks))
    output.chmod(0o600)
    return {
        "clip_path": str(output),
        "clip_sha256": _sha256_file(output),
        "duration_seconds": sum(float(item["end"]) - float(item["start"]) for item in segments),
        "segments": [dict(item) for item in segments],
    }


def _transcribe_clips(
    bindings: Sequence[Mapping[str, Any]], *, model_snapshot: Path
) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_snapshot), device="cpu", compute_type="int8")
    rows = []
    for binding in bindings:
        segments, _info = model.transcribe(
            str(binding["clip_path"]), language="en", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        text = " ".join((item.text or "").strip() for item in segments if (item.text or "").strip())
        rows.append({"speaker_ref": binding["speaker_ref"], "transcript": text})
    return rows


def _score_matrices(
    bindings: Sequence[Mapping[str, Any]],
    *,
    preview: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    profiles, _profile_summary = recalibration._active_profiles(
        calibration_root=recalibration.DEFAULT_CALIBRATION_ROOT,
        p3_runtime_root=recalibration.DEFAULT_P3_RUNTIME_ROOT,
    )
    adapters = verification.adapter_registry()
    allowlist = set(preview["p0_authority"]["allowlisted_subject_ids"])
    if {item["person_ref_id"] for item in profiles} != allowlist:
        raise Plan0056RunnerError("The enrolled acoustic profile allowlist drifted.")
    thresholds = {
        (item["candidate_id"], item["method_id"]): item
        for item in preview["threshold_units"]
    }
    rows_by_unit = {
        (candidate, method): []
        for candidate in acoustic_contract.CANDIDATE_IDS
        for method in acoustic_contract.METHOD_IDS
    }
    for binding in bindings:
        methods = generation5_e2._prepare_clip_methods(
            binding, authority_sha256=preview["content_sha256"], paths=paths
        )
        for method_id, pcm_path in sorted(methods.items()):
            samples, sample_rate = _read_pcm16_mono(pcm_path)
            for candidate_id in acoustic_contract.CANDIDATE_IDS:
                scores = []
                for profile in profiles:
                    if profile["candidate_id"] != candidate_id:
                        continue
                    scored = verification.score_profile(
                        profile["profile_id"], adapter=adapters[candidate_id],
                        probe_samples=samples, sample_rate=sample_rate,
                        runtime_root=recalibration.DEFAULT_CALIBRATION_ROOT,
                        p3_runtime_root=recalibration.DEFAULT_P3_RUNTIME_ROOT,
                    )
                    value = float(scored["score"])
                    if not math.isfinite(value):
                        raise Plan0056RunnerError("An acoustic score is nonfinite.")
                    scores.append({
                        "subject_id": profile["person_ref_id"], "score": value,
                        "trial_id": scored["trial_id"], "probe_sha256": scored["probe_sha256"],
                    })
                scores.sort(key=lambda item: item["subject_id"])
                if len(scores) != 2:
                    raise Plan0056RunnerError("Each matrix row requires two enrolled scores.")
                rows_by_unit[(candidate_id, method_id)].append(
                    {"speaker_ref": binding["speaker_ref"], "scores": scores}
                )
    ensure_private_tree(paths["root"], paths["run"], paths["matrices"])
    matrices = []
    for candidate_id, method_id in sorted(rows_by_unit):
        threshold = thresholds[(candidate_id, method_id)]
        core = {
            "schema_version": MATRIX_SCHEMA, "candidate_id": candidate_id,
            "method_id": method_id, "threshold": threshold["threshold"],
            "temperature": threshold["temperature"], "rows": rows_by_unit[(candidate_id, method_id)],
            "contains_gold": False, "contains_display_names": False,
        }
        matrix = {**core, "content_sha256": _canonical_hash(core)}
        write_immutable_private_json(paths["matrices"] / f"{candidate_id}--{method_id}.json", matrix)
        matrices.append(matrix)
    return matrices


def _current_identity_state() -> dict[str, Any]:
    return pilot.snapshot_identity_state(
        primary_store=pilot.DEFAULT_PRIMARY_STORE,
        knowledge_store=pilot.DEFAULT_KNOWLEDGE_STORE,
        profile_store=pilot.DEFAULT_PROFILE_STORE,
        reference_store=pilot.DEFAULT_REFERENCE_STORE,
    )


def execute_local_pilot(
    expected_authority_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    authority.replay_plan0056_execution_authority(
        expected_authority_sha256, runtime_root=runtime_root
    )
    pilot.replay_plan0056_authority(
        authority.P0_CONTENT_SHA256, runtime_root=pilot.DEFAULT_RUNTIME_ROOT
    )
    paths = _execution_paths(runtime_root, expected_authority_sha256)
    if paths["receipt"].exists():
        return replay_local_pilot(expected_authority_sha256, runtime_root=runtime_root)
    preview = _read_frozen_preview(paths)
    p0_preview = _read_p0_preview()
    before = _current_identity_state()
    if before != p0_preview["identity_state_before"]:
        raise Plan0056RunnerError("Identity state drifted before local pilot execution.")
    source_binding = p0_preview["private_evidence"]["sources"][0]
    source = Path(source_binding["path"])
    ensure_private_tree(paths["root"], paths["authority_run"], paths["run"])
    _decode_private_pcm(source, paths["pcm"])
    timeline = _run_local_diarization(
        paths["pcm"], model_root=Path(preview["local_runtime"]["diarization_model"]["root"]),
        minimum_speakers=preview["diarization_policy"]["minimum_speakers"],
        maximum_speakers=preview["diarization_policy"]["maximum_speakers"],
    )
    policy = preview["review_clip_policy"]
    selected = select_review_segments(
        timeline, minimum_turn_seconds=policy["minimum_turn_seconds"],
        maximum_turn_seconds=policy["maximum_turn_seconds"],
        maximum_turns_per_speaker=policy["maximum_turns_per_speaker"],
        target_seconds_per_speaker=policy["target_seconds_per_speaker"],
        minimum_usable_seconds_per_speaker=policy["minimum_usable_seconds_per_speaker"],
    )
    write_immutable_private_json(paths["diarization"], {"timeline": timeline, "selected": selected})
    bindings = []
    for speaker_ref, segments in selected.items():
        binding = _write_speaker_clip(paths["pcm"], paths["clips"] / f"{speaker_ref}.wav", segments)
        bindings.append({"speaker_ref": speaker_ref, **binding})
    snapshots = Path(authority.DEFAULT_WHISPER_CACHE_ROOT.expanduser().absolute() / "snapshots")
    model_snapshot = next(path for path in sorted(snapshots.iterdir()) if path.is_dir())
    transcripts = _transcribe_clips(bindings, model_snapshot=model_snapshot)
    ensure_private_tree(paths["root"], paths["run"], paths["transcripts"])
    write_immutable_private_json(paths["transcripts"] / "speaker-transcripts.json", {"rows": transcripts})
    matrices = _score_matrices(bindings, preview=preview, paths=paths)
    proposals = authority.proposals_from_matrices(
        matrices, expected_speaker_refs=list(selected),
        allowlisted_subject_ids=preview["p0_authority"]["allowlisted_subject_ids"],
    )
    after = _current_identity_state()
    if after != before:
        raise Plan0056RunnerError("Identity or acoustic profile state changed during the pilot.")
    artifacts = {
        "source_pcm": {"path": str(paths["pcm"]), "sha256": _sha256_file(paths["pcm"])},
        "diarization": {"path": str(paths["diarization"]), "sha256": _sha256_file(paths["diarization"])},
        "clips": bindings,
        "transcripts": transcripts,
        "matrices": matrices,
        "proposals": proposals,
    }
    manifest = {
        "schema_version": EXECUTION_SCHEMA, "status": "complete_pending_human_review",
        "authority_content_sha256": expected_authority_sha256,
        "p0_content_sha256": authority.P0_CONTENT_SHA256,
        "source_sha256": source_binding["source_sha256"],
        "identity_state_before": before, "identity_state_after": after,
        "identity_state_unchanged": True, "artifacts": artifacts,
        "read_pilot_outcome_gold": False, "applied_assignments": False,
        "requires_human_review": True,
    }
    write_immutable_private_json(paths["manifest"], manifest)
    receipt = {
        "schema_version": EXECUTION_SCHEMA, "status": manifest["status"],
        "authority_content_sha256": expected_authority_sha256,
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "speaker_count": len(bindings), "matrix_count": len(matrices),
        "proposal_count": len(proposals["proposals"]),
        "proposal_content_sha256": proposals["content_sha256"],
        "identity_state_unchanged": True, "requires_human_review": True,
        "applied_assignments": False,
    }
    write_immutable_private_json(paths["receipt"], receipt)
    return {**receipt, "idempotent_replay": False}


def replay_local_pilot(
    expected_authority_sha256: str,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    authority.replay_plan0056_execution_authority(expected_authority_sha256, runtime_root=runtime_root)
    paths = _execution_paths(runtime_root, expected_authority_sha256)
    require_private_file(paths["manifest"], paths["root"])
    require_private_file(paths["receipt"], paths["root"])
    manifest = read_private_object(paths["manifest"])
    receipt = read_private_object(paths["receipt"])
    if (
        receipt.get("manifest_sha256") != _sha256_file(paths["manifest"])
        or manifest.get("authority_content_sha256") != expected_authority_sha256
        or manifest.get("identity_state_unchanged") is not True
        or _current_identity_state() != manifest.get("identity_state_after")
    ):
        raise Plan0056RunnerError("Local pilot execution evidence drifted.")
    return {**receipt, "replay_schema_version": EXECUTION_REPLAY_SCHEMA, "idempotent_replay": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute or replay the frozen Plan 0056 local pilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("execute", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--authority-content-sha256", required=True)
        child.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operation = execute_local_pilot if args.command == "execute" else replay_local_pilot
    result = operation(args.authority_content_sha256, runtime_root=args.runtime_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
