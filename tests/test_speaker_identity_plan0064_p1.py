from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import speaker_identity_plan0064_p1 as p1


@dataclass
class Adapter:
    candidate_id: str


def _profiles() -> list[dict[str, object]]:
    result = []
    for model in ("m1", "m2", "m3"):
        for index in range(7):
            bound = index < 6
            result.append(
                {
                    "profile_id": f"profile-{model}-{index}",
                    "person_ref_id": f"subject-{index}",
                    "canonical_person_id": f"person-{index}" if bound else None,
                    "identity_candidate_eligible": bound,
                    "candidate_id": model,
                    "model_revision": f"revision-{model}",
                    "artifact": {"sha256": f"artifact-{model}-{index}"},
                }
            )
    return result


def _score(values: dict[tuple[str, str], float]):
    def score(profile_id, *, adapter, **_kwargs):
        subject = profile_id.rsplit("-", 1)[-1]
        return {
            "score": values.get((adapter.candidate_id, subject), 0.1),
            "trial_id": f"trial-{adapter.candidate_id}-{subject}",
        }

    return score


def _slot(values: dict[tuple[str, str], float]):
    return p1._score_slot(
        document_id="doc",
        speaker="A",
        probe=[0.25] * (p1.SAMPLE_RATE * 3),
        profiles=_profiles(),
        thresholds={"m1": 0.5, "m2": 0.5, "m3": 0.5},
        adapters={name: Adapter(name) for name in ("m1", "m2", "m3")},
        score_fn=_score(values),
        profile_root=Path("/profiles"),
        reference_root=Path("/references"),
    )


def test_multi_model_support_emits_bound_person_candidate():
    result = _slot({("m1", "1"): 0.9, ("m2", "1"): 0.8})
    assert result["status"] == "candidate"
    assert result["candidate_person_id"] == "person-1"
    assert result["candidate_acoustic_subject_id"] == "subject-1"
    assert result["supporting_model_count"] == 2
    assert len(result["model_rows"]) == 3
    assert all(len(row["scores"]) == 7 for row in result["model_rows"])


def test_unbound_top_scores_never_emit_person_candidate():
    result = _slot({("m1", "6"): 0.9, ("m2", "6"): 0.9, ("m3", "6"): 0.9})
    assert result["status"] == "abstain"
    assert result["reason_code"] == "no_bound_profile_threshold_pass"
    assert result["candidate_person_id"] is None
    assert all(row["binding_eligible"] is False for row in result["model_rows"])


def test_conflicting_model_support_routes_review_without_identity():
    result = _slot({("m1", "1"): 0.9, ("m2", "2"): 0.9})
    assert result["status"] == "review"
    assert result["reason_code"] == "conflicting_acoustic_support"
    assert result["candidate_person_id"] is None


def test_short_probe_is_reason_coded_without_scoring():
    called = False

    def score(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    result = p1._score_slot(
        document_id="doc", speaker="B", probe=[0.0] * 100,
        profiles=_profiles(), thresholds={"m1": 0.5}, adapters={"m1": Adapter("m1")},
        score_fn=score, profile_root=Path("/profiles"), reference_root=Path("/references"),
    )
    assert result["reason_code"] == "insufficient_speaker_audio"
    assert called is False


def test_threshold_loader_requires_complete_frozen_no_enhancement_matrix(tmp_path):
    path = tmp_path / "authority" / "threshold-application.json"
    path.parent.mkdir(mode=0o700)
    payload = {
        "status": "success",
        "contains_frozen_thresholds": True,
        "did_enable_default_integration": False,
        "did_mutate_profiles_or_references": False,
        "execution_authority_sha256": "a" * 64,
        "score_matrix_sha256": "b" * 64,
        "thresholds": [
            {"candidate_id": "m1", "method_id": p1.METHOD_ID, "threshold": 0.5, "temperature": 0.1}
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    loaded = p1._thresholds(path, ["m1"])
    assert loaded["units"][0]["threshold"] == 0.5
    with pytest.raises(p1.Plan0064P1Error, match="incomplete"):
        p1._thresholds(path, ["m1", "m2"])


def test_slot_probe_is_chronological_and_bounded():
    transcript = {"utterances": [
        {"speaker": "A", "start": 0, "end": 1000},
        {"speaker": "B", "start": 1000, "end": 2000},
        {"speaker": "A", "start": 2000, "end": 40000},
    ]}
    samples = p1.array("f", [float(index) for index in range(p1.SAMPLE_RATE * 40)])
    probe = p1._slot_probe(transcript, "A", samples)
    assert len(probe) == int(p1.MAX_PROBE_SECONDS * p1.SAMPLE_RATE)
    assert probe[0] == 0.0
    assert probe[p1.SAMPLE_RATE] == float(2 * p1.SAMPLE_RATE)


def test_caching_adapter_embeds_same_probe_once():
    class Wrapped:
        candidate_id = "m1"
        revision_sha = "revision"
        embedding_dimension = 2
        model_loaded = False

        def __init__(self):
            self.calls = 0

        def embed(self, samples, *, sample_rate):
            self.calls += 1
            return (float(len(samples)), float(sample_rate))

    wrapped = Wrapped()
    adapter = p1._CachingAdapter(wrapped)
    probe = [0.0, 1.0]
    assert adapter.embed(probe, sample_rate=16_000) == (2.0, 16_000.0)
    assert adapter.embed(probe, sample_rate=16_000) == (2.0, 16_000.0)
    assert wrapped.calls == 1
    assert adapter.embed(list(probe), sample_rate=16_000) == (2.0, 16_000.0)
    assert wrapped.calls == 2


def test_decode_normalizes_signed_pcm(monkeypatch):
    pcm = p1.array("h", [-32768, -1, 0, 1, 32767])
    monkeypatch.setattr(
        p1.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=pcm.tobytes(), stderr=b""
        ),
    )
    decoded = p1._decode(Path("fixture.wav"))
    assert min(decoded) == -1.0
    assert max(decoded) < 1.0
    assert all(-1.0 <= value <= 1.0 for value in decoded)
