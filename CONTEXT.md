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

**Calibrated Likelihood**:
An empirical estimate of correctness for a score band, shown only with enough
reviewed source-disjoint outcomes and accompanied by sample size, interval,
and evaluation version.
_Avoid_: Evidence strength, model confidence

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

**Evidence Retrieval Request**:
The immutable host-owned specification of which conversation clues, source
profiles, candidates, temporal policy, and packet budgets may be used to
collect identity evidence.
_Avoid_: Model search request, unrestricted provenance search

**Evidence Bundle**:
The immutable, content-hashed collection of prepared people, clues, evidence
snapshots, relationship and concept summaries, warnings, and exact output
allowlists supplied to one evaluation.
_Avoid_: Live provider results, mutable context window

**Observation**:
An attributable record of what a source, transcript, or reviewer supplied at
a particular time. Observations remain immutable and support rebuildable
profiles and claims.
_Avoid_: Current truth, overwritten profile

**Claim**:
A reviewable assertion derived from one or more observations, with supporting
and contradicting evidence, temporal applicability, alternatives, confidence,
and status.
_Avoid_: Observation, established fact

**As-Of Time**:
The time boundary used to decide which evidence and accumulated knowledge an
evaluation may treat as available.
_Avoid_: Retrieval time, review time

**Hindsight Evidence**:
Evidence learned only after the conversation or evaluation time, including
later reviewer corrections. It must be labeled and excluded from blind
historical evaluation unless policy explicitly allows it.
_Avoid_: Contemporaneous evidence, refreshed provenance

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

**External Identity**:
A source-scoped identifier such as an email, Google Workspace contact ID,
Odollo contact or lead ID, or calendar attendee identity that may link a
source record to a person.
_Avoid_: Person ID, display name

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

**Derived Person Profile**:
A rebuildable summary of reviewed identities, aliases, source affinities,
relationships, interactions, topics, and terminology supported by explicit
observation IDs and a projection version.
_Avoid_: Contact record, model memory, independent evidence

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

## Contact and correction lifecycle

**Person Registry**:
The normalized set of canonical people and their reviewed or provisional links
to still-independent source records, aliases, roles, and relationships. It is
a view of the conversation knowledge authority, not a second contact database.
_Avoid_: Contact database, provider contact list

**Provisional Person**:
A durable internal person whose source records appear to represent one person
but whose resolution remains reviewable. It may later be confirmed, merged,
split, redirected, or retired without changing its historical identifiers.
_Avoid_: Confirmed person, temporary contact

**Participant Hypothesis**:
A reviewable claim that a person or source record likely participated in a
conversation. Calendar attendance or organizer status may support the claim
but does not establish that the person attended or spoke.
_Avoid_: Participant, calendar speaker

**Current Speaker Assignment**:
The rebuildable present view of which person, if any, is accepted for a
diarized speaker, speaker group, or utterance set. It is projected from review
decisions and correction history rather than overwritten directly.
_Avoid_: Speaker identity proposal, diarized label

**Voice Sample**:
An immutable reference to an exact source-audio interval with recording,
speaker or utterance, preparation, quality, identity-review, and biometric-use
lineage. A sample is not a person and is not automatically eligible to train a
profile.
_Avoid_: Voice profile, speaker identity

**Anonymous Acoustic Subject**:
A still-person-unbound recurring voice representation used to organize samples
without asserting a canonical identity.
_Avoid_: Person, confirmed speaker

**Voice Cluster Membership**:
A soft, reversible claim that a voice sample belongs to an anonymous acoustic
subject, retaining alternatives, pairwise evidence, and review state.
_Avoid_: Speaker assignment, profile enrollment

**Voice Profile Version**:
A derived biometric profile built from an exact allowlist of reviewed eligible
voice samples under one model and recipe version, with evaluation, activation,
predecessor, rollback, retention, and deletion state.
_Avoid_: Voice sample, permanent voiceprint

**Correction Event**:
An append-only, attributable change that supersedes a person link, speaker
assignment, role, relationship, source record, or biometric derivative while
preserving both the earlier record and the current accepted view.
_Avoid_: Edit, overwrite, deletion

**Identity Review Queue Item**:
A rebuildable work item that gathers the current conversation-association,
participant, person-link, speaker, relationship, and acoustic claims requiring
human action. It points to the authoritative records rather than copying them.
_Avoid_: Review decision, static review page

**Original Recording Filename**:
The basename recorded by the source recording before transcript enrichment,
blob storage, artifact naming, or migration. It remains visible alongside the
durable recording ID and hashes but is not itself an identity key.
_Avoid_: Transcript filename, stored blob name, recording ID

**Relationship Family**:
A versioned parent category such as family, professional, medical, legal,
commercial, educational, project, or social that organizes more specific
relationship types.
_Avoid_: Relationship assertion, role

**Relationship Type**:
A directional or symmetric ontology leaf such as parent-of, spouse-of,
physician-for, or employee-of, with inverse rules and optional role detail.
_Avoid_: Relationship family, freeform label

**Contextual Role**:
A time- and context-bound role a person holds in a conversation, event,
organization, project, or matter, distinct from a durable entity relationship.
_Avoid_: Person attribute, relationship type

**Conversation Purpose Claim**:
A reviewable primary or secondary explanation of why a conversation occurred,
linked to projects, matters, organizations, evidence, alternatives, and score
history.
_Avoid_: Calendar event title, final summary

**Transcript Correction Proposal**:
A versioned span-level proposal that preserves raw ASR while identifying exact
original and replacement text, utterance/time, scoped context, evidence, and
review state.
_Avoid_: Transcript overwrite, global replacement

**Normalized Transcript Version**:
A derived transcript generation that applies accepted corrections while
retaining complete lineage to immutable raw ASR and diarization.
_Avoid_: Raw transcript, silent edit

**Terminology Entry**:
A versioned, scope-aware record of canonical spelling, expansion, definition,
aliases, pronunciation hints, and ASR confusions.
_Avoid_: Search-and-replace rule, unscoped synonym

**ASR Confusion**:
A scoped record that an ASR system may emit one form when another term is
intended. It is not a semantic synonym and is never globally applied by
default.
_Avoid_: Alias, synonym

**Semantic Correction Comment**:
An immutable freeform reviewer observation that may produce a separately
reviewed structured correction but is not itself a learning label.
_Avoid_: Accepted correction, model truth

**Speaker Identification**:
An evidence-bearing estimate of which person spoke in a conversation. It is
not identity authentication and must not authorize access, transactions, or
legal attestation.
_Avoid_: Speaker authentication, voice authentication
