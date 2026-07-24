# Conversation Intelligence

This context defines the language used to identify anonymous speakers from
transcript clues and source-attributed information before interpreting the
conversation as a whole.

## Conversation Association

**Recording**:
A single captured media file or segment. A recording has its own identity even
when it belongs to a conversation with other recordings.
_Avoid_: Conversation, transcript

**Conversation**:
The interaction represented by one or more recordings and their derived
artifacts. One recording per conversation is the ordinary case, not an
identity invariant.
_Avoid_: Recording, source file

**Calendar Association**:
A confidence-bearing hypothesis that a recording corresponds to a calendar
event, based on time proximity together with title, attendee, contact, and
topic fit. An association may be weak or spurious even when time ranges
overlap.
_Avoid_: Calendar match, matched appointment

**Calendar Association Confidence**:
The strength of the calendar association independently of any particular
speaker identity proposal.
_Avoid_: Speaker confidence, match confidence

**Conversation Clue**:
A bounded part of the conversation that bears on event or speaker identity,
such as a self-introduction, form of address, affiliation, relationship, or
topic. It is evidence for preprocessing, not an interpretation of the whole
conversation.
_Avoid_: Full-context readout, conversation summary

**Clue Discovery**:
The first preprocessing phase, in which App Intelligence identifies cited
identity and retrieval clues without accessing provenance providers or
interpreting the conversation as a whole.
_Avoid_: Identity evaluation, contextual readout

**Identity Evaluation**:
The second preprocessing phase, in which App Intelligence assesses calendar
association, person links, and speaker identities against host-prepared
evidence.
_Avoid_: Clue discovery, speaker assignment

**Confidence Assessment**:
A confidence judgment expressed as both a numeric value for filtering and
future agent use and a plain-English band for human review. The judgment is
supported by named positive and negative evidence factors.
_Avoid_: Confidence score, confidence label

**Evidence Strength Score**:
The numeric part of a confidence assessment, measuring support under a
versioned evidence rubric rather than asserting a probability of correctness.
_Avoid_: Probability, certainty

**Evidence Rubric**:
The versioned, task-specific meaning of an evidence strength score. Calendar
association, person linking, and speaker identity use separate rubrics while
sharing evidence-factor language.
_Avoid_: Threshold, model intuition

**Evidence Factor Assessment**:
An App Intelligence judgment with a named factor type, direction, strength,
cited clues and provenance, bounded rationale, and credible alternatives. The
host validates it and applies the relevant evidence rubric.
_Avoid_: Model confidence score, uncited rationale

**Evidence Independence Group**:
A set of evidence factors that may derive from the same underlying fact or
interaction and therefore must not be counted as independent corroboration.
_Avoid_: Source type, database count

**Evidence Assessment Record**:
The preserved, validated evidence factor assessments for one inference task,
including the task and rubric version. Scores and bands can be recalculated
from this record without rerunning App Intelligence.
_Avoid_: Final score, model transcript

**Evidence Snapshot**:
The immutable, bounded, source-attributed excerpts and metadata actually
presented during an evaluation. It preserves auditability without retaining
complete source bodies.
_Avoid_: Live provenance, full source export

**Conversation Processing Record**:
The conversation-owned, history-preserving collection of preprocessing
evaluations, evidence and confidence assessments, review state, and
source-artifact references. One evaluation is explicitly designated current.
_Avoid_: Transcript, intelligence result

**Conversation ID**:
A durable opaque identity assigned to a conversation and preserved across
artifact renames, regeneration, and storage migration.
_Avoid_: Recording ID, file path, conversation key, content hash

**Re-scoring**:
Recalculating a confidence assessment from an existing evidence assessment
record under a selected evidence rubric, without collecting or interpreting
new evidence.
_Avoid_: Re-evaluation, provenance refresh

**Re-evaluation**:
Creating a new evidence assessment from newly collected provenance and
conversation clues. It does not alter the historical assessment it follows.
_Avoid_: Re-scoring, historical correction

**Confidence Band**:
A deterministic human-readable view of a numeric confidence assessment under
a defined scoring rubric. Its boundaries are meaningful only within the
versioned scale that defines the numeric value.
_Avoid_: Model-selected band, unresolved, conflicting

## Speaker Identity

**Diarized Speaker**:
An anonymous label assigned to a set of utterances by a diarization process;
it is an observation about the audio and is not necessarily one person.
_Avoid_: Person, participant, identified speaker

**Split Speaker**:
A diarization error in which one person is represented by multiple diarized
speakers in the same conversation.
_Avoid_: Multiple participants

**Merged Speaker**:
A diarization error in which one diarized speaker contains utterances from
multiple people.
_Avoid_: Shared identity

**Mixed Speaker Finding**:
A review-required finding that one diarized speaker may contain multiple
people. It preserves the original label and identifies the utterances that
support the finding.
_Avoid_: Merged identity, corrected diarization

**Utterance Identity Proposal**:
A review-required identity claim for particular utterances within a mixed
speaker finding, used when the evidence does not support one identity for the
entire diarized speaker.
_Avoid_: Speaker assignment, automatic re-diarization

**Prepared Candidate**:
A person gathered from calendar-associated or other reviewed provenance who
may be one of the conversation's anonymous speakers.
_Avoid_: Known speaker, calendar speaker

**Person Candidate**:
A prepared candidate representing one apparent person whose clearly linked
source records have been grouped while their individual provenance remains
visible. Ambiguous same-name records remain separate person candidates.
_Avoid_: Contact record, merged contact

**Person Link Assessment**:
A confidence assessment that two or more source records represent the same
person, supported by exact identifiers or contextual inference. It groups the
records for preprocessing without merging or changing their source data.
_Avoid_: Contact merge, deterministic duplicate rule

**Source Record**:
A source-attributed representation of a person in one account, tenant, or
database. Several source records may contribute to one person candidate.
_Avoid_: Person, duplicate person

**Relationship Context**:
Information implied by the account, tenant, or database in which a source
record appears, describing whose relationship with the person that source is
likely to represent.
_Avoid_: Identity proof, source priority

**Source Affinity**:
The relationship between a person candidate and a provenance source in which
that person has a source record. A person candidate may have multiple source
affinities, each retaining its own relationship context.
_Avoid_: Preferred source, duplicate identity

**Source Context**:
The declared owner, relationship scope, account or tenant label, evidence
capabilities, and identifier authority of a configured provenance source.
_Avoid_: Inferred tenant meaning, source credentials

**Independent Corroboration**:
Support from distinct evidence types or interactions rather than repeated
copies of the same underlying fact. Duplicate source records do not create
independent corroboration by themselves.
_Avoid_: Record count, source count

**Speaker Identity Proposal**:
A review-required claim connecting an anonymous speaker either to a prepared
candidate or to a person suggested by the conversation itself.
_Avoid_: Speaker assignment, resolved speaker

**Ready-to-Confirm Proposal**:
A high-strength speaker identity proposal whose evidence is complete enough
for lightweight human confirmation but which has not changed any speaker
assignment.
_Avoid_: Confirmed identity, automatic assignment

**Confirmed Speaker Identity**:
A speaker identity proposal accepted by a human reviewer and designated as the
current identity for its speaker or speaker group.
_Avoid_: Model proposal, high-confidence match

**Reviewer-Asserted Identity**:
A confirmed speaker identity based on explicit reviewer knowledge when no
supporting provenance is available. It records its human origin and does not
claim an evidence strength score.
_Avoid_: High-confidence proposal, inferred identity

**Calibration Outcome**:
The preserved comparison between an evidence-backed proposal and its human
review decision, including any structured correction and rejection reason.
_Avoid_: Training example, contact update, memory write

**Review Decision**:
An attributable, timestamped human decision on a specific proposal and
evaluation. A later decision may supersede it without erasing its history.
_Avoid_: Anonymous approval, overwritten status

**Speaker Group Proposal**:
A review-required claim that two or more diarized speakers represent the same
person. It preserves the original diarized labels and carries one speaker
identity proposal for the group.
_Avoid_: Merged speaker, relabeled transcript

**Speaker Identity Confidence**:
The strength of a speaker identity proposal independently of confidence in the
underlying calendar association.
_Avoid_: Calendar confidence, combined confidence

**Candidate Match**:
A speaker identity proposal that identifies exactly one prepared candidate.
_Avoid_: Confirmed identity, automatic match

**Unlisted Person Suggestion**:
A speaker identity proposal for a person absent from the prepared candidates,
supported by conversation clues and requiring further lookup and human review.
_Avoid_: New contact, invented candidate

**Unresolved Proposal**:
A speaker identity proposal state used when the evidence does not support a
specific candidate match or unlisted person suggestion.
_Avoid_: Zero confidence, low-confidence match

**Conflicting Proposal**:
A speaker identity proposal state used when credible evidence supports
incompatible identities or mappings that require human resolution.
_Avoid_: Low confidence, unresolved proposal
