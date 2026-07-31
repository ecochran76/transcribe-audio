# Acoustic processing and speaker verification research

Last updated: 2026-07-31

This explanation records the research and architecture direction for adding
speech cleanup and biometric speaker evidence to the transcription pipeline.
The corresponding execution authority is
[Plan 0037](../plans/0037-2026-07-31-audio-enhancement-biometric-speaker-identity.md).

> **Note:** These features are experimental. A voice match supplies identity
> evidence; it does not authenticate a person or prove that a recording is
> genuine.

## Decision summary

Use a local, host-owned acoustic processing module before App Intelligence.
The module preserves original audio, produces versioned derived audio, and
returns bounded acoustic evidence. App Intelligence combines that evidence
with transcript, calendar, contact, relationship, email, Drive, and Odollo
provenance.

The first model bake-off will compare SpeechBrain ECAPA-TDNN with WeSpeaker
CAM++ and one WeSpeaker ResNet or ECAPA checkpoint. WeSpeaker is the leading
production candidate because it supports pretrained speaker-verification
models, ONNX inference, PLDA, score normalization, and quality-aware
calibration. SpeechBrain provides a compact baseline. NVIDIA TitaNet remains a
later GPU-oriented candidate.

Use pyannote.audio for speaker segmentation, overlap detection, and
diarization repair. Use Silero VAD for inexpensive speech-region detection.
Compare DeepFilterNet and RNNoise for noise suppression. Enhancement must earn
promotion by improving transcription, diarization, or verification on local
recordings without increasing identity errors.

## Why the prototype is not enough

The review-only prototype normalized audio to 16 kHz mono, extracted WavLM
Base Plus representations in 2.5-second windows, pooled the final hidden
layers, and compared centroids with cosine similarity. It supplied useful
supporting evidence for one difficult conversation, but it has four limits:

- WavLM Base Plus is a general speech representation, not a calibrated
  speaker-verification model.
- Raw cosine similarity is not a probability and cannot support a stable
  confidence policy across recording conditions.
- A single centroid can hide disagreement among windows, channel conditions,
  and diarization errors.
- The prototype has no durable enrollment, model-version, privacy, or
  abstention contract.

These limits require a product module and evaluation campaign rather than a
direct promotion of the prototype.

## Candidate libraries

The following projects cover different parts of the local acoustic pipeline.

| Project | Proposed role | Relevant capabilities | Initial posture |
| --- | --- | --- | --- |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) | Primary speaker-verification candidate | CAM++, ECAPA, ResNet, ReDimNet, ONNX, PLDA, score normalization, and quality-aware calibration | Evaluate first |
| [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Verification baseline | Compact pretrained English speaker embedding and verification model | Evaluate first |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Segmentation and diarization repair | Speech segmentation, speaker changes, overlap, diarization, and embeddings | Evaluate for turn preparation, not named identity authority |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice activity detection | Lightweight local speech timestamps and silence exclusion | Preferred first VAD |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Full-band enhancement candidate | Local neural noise suppression with Python and native implementations | Compare against no enhancement and RNNoise |
| [RNNoise](https://github.com/xiph/rnnoise) | Lightweight enhancement baseline | Real-time recurrent noise suppression with a small native runtime | Compare as low-cost baseline |
| [NVIDIA TitaNet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/models.html) | Later verification candidate | Scalable speaker embeddings, pretrained checkpoints, and fine-tuning through NeMo | Defer unless GPU evidence justifies its dependency cost |

WeSpeaker code uses Apache-2.0, while its pretrained checkpoint terms follow
their training datasets. SpeechBrain and NVIDIA NeMo use Apache-2.0.
pyannote.audio code uses MIT, but its hosted checkpoints can have separate
access terms. Silero VAD uses MIT, RNNoise uses BSD-3-Clause, and DeepFilterNet
offers MIT or Apache-2.0. Record both code and checkpoint terms before pinning
any model.

## Processing architecture

The external seam should be one deep module with a small interface:

```python
analyze_recording(
    source_audio,
    diarized_turns,
    enrolled_people,
    policy,
) -> AcousticEvidenceBundle
```

Its implementation owns decoding, channel policy, voice activity detection,
enhancement, quality measurement, clean-window selection, embedding
extraction, enrollment aggregation, score normalization, calibration,
same-person clustering, and abstention. Model adapters remain internal so the
caller does not depend on WeSpeaker, SpeechBrain, or pyannote types.

The module returns derived evidence rather than a final identity decision:

- source and derived-audio hashes;
- preprocessing recipe and model revisions;
- diarized label and candidate person identifier;
- calibrated score, confidence band, top-candidate margin, and threshold
  revision;
- usable speech duration, window count, quality measures, and per-window
  agreement;
- reference-profile identifier and enrollment provenance;
- same-person evidence across diarized labels;
- abstention or degradation reasons;
- warnings for overlap, short speech, clipping, enhancement instability,
  channel mismatch, and insufficient reference diversity.

`build_speaker_clue_packet(...)` can include this bundle as another prepared
evidence family. The existing host validator must reject model output that
references an unprepared acoustic candidate or evidence identifier.

## Audio preservation and enhancement

Never replace the original recording. Store each cleaned track as a derived,
content-addressed artifact linked to the original source hash. Record the
decoder, resampling, channel-selection, VAD, enhancement model, parameters,
and output hash.

Silence removal means excluding non-speech windows from model inputs. It does
not mean destructively cutting the archived recording. Keep timestamps mapped
back to the original timeline so transcripts, diarization, playback, and
evidence citations remain aligned.

Noise suppression can change speaker-discriminating traits. Compare every
candidate model on original and enhanced audio. Allow the policy to use the
original, enhanced, or both score streams. Abstain when enhancement changes
the candidate ordering or lowers window agreement beyond a calibrated limit.

## Biometric library

The biometric library is a private speaker-reference store, not a contact
database. A person record can reference several enrollment sessions across
devices and acoustic conditions. Enrollment requires operator-confirmed
identity and source provenance.

Store raw audio only through existing private blob references. Store voice
embeddings and calibration data in private user-scoped storage with restrictive
permissions. Transcript sidecars may contain derived evidence identifiers,
scores, model revisions, and provenance references, but not raw embeddings.

An enrollment profile should contain:

- stable person identifier;
- source recording and confirmed speaker-label references;
- selected segment timestamps and quality measures;
- embedding model and preprocessing revisions;
- session and device diversity summaries;
- aggregate representation plus within-person dispersion;
- creation, supersession, withdrawal, and deletion audit fields.

Do not enroll a person from a calendar invitation, transcript inference, or
unreviewed model proposal. Do not use this feature for login, authorization,
liveness, fraud proof, or synthetic-audio detection.

## Evaluation and calibration

Public VoxCeleb scores do not determine our operating thresholds. Build local
same-person and different-person trials from operator-confirmed recordings.
Keep conversations, not windows, as the split unit so segments from one
recording cannot leak across train and evaluation sets.

Measure at least:

- false acceptance rate and false rejection rate;
- equal error rate as a diagnostic, not the operating objective;
- top-1 and top-k candidate recall;
- open-set rejection and abstention yield;
- expected calibration error or Brier score;
- performance by usable duration, device, telephone bandwidth, overlap,
  noise, and enhancement path;
- same-person diarization-label merge precision and recall;
- downstream transcript word error rate and diarization error rate changes.

Prefer a conservative operating point that produces useful supporting
evidence while keeping false identity support low. Confidence must combine the
calibrated verification score, candidate margin, signal quality, reference
diversity, window agreement, and corroborating non-acoustic evidence.

## Sources

These primary project sources were reviewed on 2026-07-31:

- [WeSpeaker project and recipes](https://github.com/wenet-e2e/wespeaker)
- [WeSpeaker pretrained models and checkpoint terms](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- [SpeechBrain ECAPA-TDNN checkpoint](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [pyannote.audio Community-1 pipeline](https://github.com/pyannote/pyannote-audio)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [RNNoise](https://github.com/xiph/rnnoise)
- [NVIDIA NeMo speaker models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/models.html)

## Next step

Execute Plan 0037 from its evaluation and artifact-contract gate. Do not start
historical reprocessing or biometric enrollment until the private storage,
provenance, deletion, and benchmark contracts pass review.
