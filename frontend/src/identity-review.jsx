import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./icons.jsx";
import {
  createInFlightGate,
  findUniqueAcceptedPersonTarget,
  flattenDirectoryReviewRows,
  hasDirectoryReviewDecision,
  isCompletePersonName,
  personTargetDisplayLabel
} from "./directory-review-utils.js";

const SPEAKER_ACTIONS = [
  "confirm",
  "choose_existing_person",
  "create_reviewed_provisional_person",
  "not_listed",
  "unresolved",
  "mixed_speaker",
  "group_labels",
  "split_label",
  "defer"
];
const SPEAKER_ACTION_LABELS = {
  confirm: "Confirm automated ID",
  choose_existing_person: "Choose existing person",
  create_reviewed_provisional_person: "Create provisional person",
  not_listed: "Person not listed",
  unresolved: "Mark unresolved",
  mixed_speaker: "Mixed speaker",
  group_labels: "Group labels",
  split_label: "Split label",
  defer: "Defer"
};

const REVIEW_COLUMNS = [
  { key: "title", label: "Recording", defaultDirection: "asc" },
  { key: "date", label: "Date", defaultDirection: "desc" },
  { key: "duration", label: "Duration", defaultDirection: "desc" },
  { key: "speakers", label: "Speakers", defaultDirection: "desc" },
  { key: "turns", label: "Turns", defaultDirection: "desc" },
  { key: "status", label: "Status", defaultDirection: "desc" }
];
const DEFAULT_REVIEW_COLUMN_WIDTHS = [42, 16, 9, 9, 8, 16];
const MIN_REVIEW_COLUMN_WIDTHS = [24, 10, 7, 7, 6, 10];
const REVIEW_COLUMN_STORAGE_KEY = "transcribe-review-column-widths-v1";

const FALLBACK_QUEUE = {
  items: [
    {
      queue_item_id: "fixture-queue-1",
      conversation_id: "fixture-conversation-1",
      recording_id: "fixture-recording-1",
      original_recording_filename: "Monday planning.m4a",
      speakers: [
        {
          speaker_ref: "A",
          proposal_id: "proposal-1",
          best_guess: { person_id: "", label: "Unresolved" },
          alternatives: [],
          audio: { media_url: "", start_ms: 1250, end_ms: 6900 }
        }
      ],
      review_state: "unreviewed",
      projection_version: "1",
      created_at: "2026-08-16T18:00:00Z",
      display: {
        title: "Monday planning",
        recorded_at: "2026-08-16T18:00:00Z",
        duration_ms: 1890000,
        utterance_count: 42,
        media_url: "",
        diarization: [
          {
            speaker_ref: "A",
            utterance_count: 20,
            talk_time_ms: 850000,
            sample_segments: [{ start_ms: 1250, end_ms: 6900, text: "Representative sample" }]
          }
        ]
      }
    }
  ],
  total: 1
};

const FALLBACK_PEOPLE = {
  items: [
    {
      person_id: "contact:contact-redacted-alex",
      source_identity_id: "contact-redacted-alex",
      identity_kind: "local_contact",
      status: "provisional",
      primary_name: "Alex Example",
      aliases: [],
      contact_class: "person_candidate",
      contact_methods: [{ kind: "email", value: "alex@example.test" }],
      source_records: [{ source_record_id: "contact-redacted-alex", provider_kind: "fixture", record_type: "calendar_attendee_contact", label: "Alex Example", resolution_status: "provisional" }],
      roles: [],
      relationships: [],
      role_hypotheses: [{
        hypothesis_id: "mail-hypothesis-role-redacted-1",
        hypothesis_kind: "contextual_role",
        relationship_type: "HAS_CONTEXTUAL_ROLE",
        display_value: "Program Director",
        counterpart_label: "Program Director",
        organization: "Example Organization",
        department: "Programs",
        basis: "A structured mail signature declares this title.",
        why_not_accepted: "A signature title may be stale or contextual and has not been reviewed.",
        evidence_source: "mail_metadata",
        observation_count: 2,
        independent_thread_count: 2,
        first_observed_at: "2025-12-15T12:00:00Z",
        last_observed_at: "2026-01-06T12:00:00Z",
        directionality: "directional",
        conflicts: [{ reason: "conflicting_structured_role", title: "Acting Director", observed_at: "2026-01-06T12:00:00Z" }],
        evidence_observation_ids: ["mail-observation-redacted-1", "mail-observation-redacted-2"],
        evidence_independence_group_ids: ["mail-interaction-redacted-1", "mail-interaction-redacted-2"],
        status: "proposed"
      }],
      relationship_hypotheses: [{
        hypothesis_id: "mail-hypothesis-relationship-redacted-1",
        hypothesis_kind: "correspondence",
        relationship_type: "CORRESPONDED_WITH",
        counterpart_label: "Account Contact",
        basis: "Mail occurred in both directions across 2 independent threads.",
        why_not_accepted: "Correspondence does not establish a named personal or professional relationship.",
        evidence_source: "mail_metadata",
        observation_count: 2,
        independent_thread_count: 2,
        first_observed_at: "2025-12-15T12:00:00Z",
        last_observed_at: "2026-01-06T12:00:00Z",
        directionality: "symmetric",
        mail_direction: "symmetric",
        conflicts: [],
        evidence_observation_ids: ["mail-observation-redacted-1", "mail-observation-redacted-2"],
        evidence_independence_group_ids: ["mail-interaction-redacted-1", "mail-interaction-redacted-2"],
        status: "proposed"
      }],
      review_occurrences: [],
      speaker_review_count: 0,
      recording_count: 2,
      possible_related_records: [],
      input_watermark: "fixture-watermark-1",
      built_at: "2026-08-16T18:00:00Z"
    }
  ],
  total: 1,
  counts: { canonical_person: 0, local_contact: 1, reviewed_speaker: 0 },
  graph_discovery: { mail_hypothesis_count: 2, accepted_effect_count: 0, provider_write_count: 0 }
};

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `${response.status} ${response.statusText}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function label(value) {
  return String(value || "").replaceAll("_", " ");
}

const HYPOTHESIS_LABELS = {
  affiliated_with: "Affiliated with",
  calendar_co_invitation: "Calendar co-invitation",
  calendar_co_invited_with: "Calendar co-invited with",
  corresponded_with: "Corresponded with",
  has_contextual_role: "Contextual role",
  mail_thread_coparticipant_with: "Shared mail thread",
  sent_mail_to: "Sent mail to"
};

function hypothesisLabel(value) {
  const key = String(value || "").toLowerCase();
  if (HYPOTHESIS_LABELS[key]) return HYPOTHESIS_LABELS[key];
  const text = label(key).trim();
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : "—";
}

function compactHash(value) {
  const text = String(value || "");
  return text ? `${text.slice(0, 10)}…${text.slice(-8)}` : "Unavailable";
}

function strength(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "";
}

function operationId(prefix) {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`}`;
}

function formatDate(value) {
  const date = new Date(value || "");
  if (Number.isNaN(date.getTime())) return "Date unavailable";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function formatDuration(value) {
  const totalSeconds = Math.max(0, Math.round(Number(value || 0) / 1000));
  if (!totalSeconds) return "—";
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatTime(value) {
  const totalSeconds = Math.max(0, Math.round(Number(value || 0) / 1000));
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, "0")}`;
}

function recordingTitle(item) {
  return item.display?.title || item.original_recording_filename || "Untitled recording";
}

function recordingDate(item) {
  const parsed = Date.parse(item.display?.recorded_at || item.created_at || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function operatorReview(item) {
  return item.display?.operator_review || null;
}

function reviewedOutcome(item, speakerRef) {
  return operatorReview(item)?.speaker_outcomes?.find((outcome) => outcome.speaker_ref === speakerRef) || null;
}

function reviewCounts(item) {
  const review = operatorReview(item);
  const outcomes = review?.speaker_outcomes || [];
  return {
    reviewed: outcomes.length,
    matched: outcomes.filter((outcome) => outcome.outcome === "person").length,
    mixed: outcomes.filter((outcome) => outcome.outcome === "mixed").length,
    unknown: outcomes.filter((outcome) => outcome.outcome === "unknown_to_reviewer").length,
    insufficient: outcomes.filter((outcome) => outcome.outcome === "insufficient_transcript").length
  };
}

function reviewStatus(item) {
  const review = operatorReview(item);
  if (review?.disposition === "duplicate_member") return "Prior · duplicate";
  if (review) {
    const counts = reviewCounts(item);
    const details = [
      counts.matched ? `${counts.matched} matched` : "",
      counts.mixed ? `${counts.mixed} mixed` : "",
      counts.unknown ? `${counts.unknown} unknown` : "",
      counts.insufficient ? `${counts.insufficient} insufficient` : ""
    ].filter(Boolean);
    return `Prior · ${details.join(" · ") || "complete"}`;
  }
  const unresolved = (item.speakers || []).filter((speaker) => !speaker.best_guess?.person_id).length;
  return unresolved ? `${unresolved} unresolved` : label(item.review_state);
}

function loadReviewColumnWidths() {
  try {
    const value = JSON.parse(globalThis.localStorage?.getItem(REVIEW_COLUMN_STORAGE_KEY) || "null");
    if (
      Array.isArray(value)
      && value.length === DEFAULT_REVIEW_COLUMN_WIDTHS.length
      && value.every((width, index) => Number.isFinite(width) && width >= MIN_REVIEW_COLUMN_WIDTHS[index])
    ) return value;
  } catch {
    // Fall through to the stable default when storage is unavailable or stale.
  }
  return DEFAULT_REVIEW_COLUMN_WIDTHS;
}

export function IdentityReviewView({ mode }) {
  return mode === "people" ? <PeopleView /> : <RecordingReviewQueue />;
}

function RecordingReviewQueue() {
  const [payload, setPayload] = useState(FALLBACK_QUEUE);
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [sort, setSort] = useState({ key: "date", direction: "desc" });
  const [columnWidths, setColumnWidths] = useState(loadReviewColumnWidths);
  const [expandedId, setExpandedId] = useState("");
  const [loadState, setLoadState] = useState({ status: "loading", message: "Loading recordings…" });
  const [preview, setPreview] = useState(null);
  const [pendingSubmission, setPendingSubmission] = useState(null);
  const [decisionState, setDecisionState] = useState({ status: "idle", message: "" });
  const audioRef = useRef(null);
  const headerColumnsRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      const params = new URLSearchParams({ limit: "100" });
      if (query.trim()) params.set("q", query.trim());
      if (stateFilter) params.set("state", stateFilter);
      try {
        const next = await fetchJson(`/api/identity-review?${params}`);
        if (cancelled) return;
        setPayload(next);
        setExpandedId((current) => next.items?.some((item) => item.queue_item_id === current) ? current : "");
        setLoadState({ status: "live", message: "recordings ready" });
      } catch (error) {
        if (cancelled) return;
        setPayload(FALLBACK_QUEUE);
        setLoadState({ status: "preview", message: `Redacted preview data: ${error.message}` });
      }
    }, query.trim() ? 180 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, stateFilter]);

  const items = useMemo(() => {
    const next = [...(payload.items || [])];
    next.sort((left, right) => {
      const direction = sort.direction === "asc" ? 1 : -1;
      let comparison = 0;
      if (sort.key === "title") comparison = recordingTitle(left).localeCompare(recordingTitle(right));
      if (sort.key === "date") comparison = recordingDate(left) - recordingDate(right);
      if (sort.key === "duration") comparison = Number(left.display?.duration_ms || 0) - Number(right.display?.duration_ms || 0);
      if (sort.key === "speakers") comparison = (left.speakers?.length || 0) - (right.speakers?.length || 0);
      if (sort.key === "turns") comparison = Number(left.display?.utterance_count || 0) - Number(right.display?.utterance_count || 0);
      if (sort.key === "status") comparison = reviewCounts(left).matched - reviewCounts(right).matched;
      return comparison * direction || recordingDate(right) - recordingDate(left) || recordingTitle(left).localeCompare(recordingTitle(right));
    });
    return next;
  }, [payload.items, sort]);

  const columnTemplate = columnWidths.map((width) => `${width}fr`).join(" ");

  useEffect(() => {
    try {
      globalThis.localStorage?.setItem(REVIEW_COLUMN_STORAGE_KEY, JSON.stringify(columnWidths));
    } catch {
      // Resizing remains available for the current session if storage is blocked.
    }
  }, [columnWidths]);

  useEffect(() => {
    setPreview(null);
    setPendingSubmission(null);
    setDecisionState({ status: "idle", message: "" });
  }, [expandedId]);

  function seekTo(startMs) {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Number(startMs || 0) / 1000);
    audioRef.current.play().catch(() => {});
  }

  function sortBy(key) {
    const column = REVIEW_COLUMNS.find((value) => value.key === key);
    setSort((current) => current.key === key
      ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key, direction: column?.defaultDirection || "asc" });
  }

  function resizeColumns(index, startWidths, deltaPercent) {
    const pairTotal = startWidths[index] + startWidths[index + 1];
    const left = Math.min(
      pairTotal - MIN_REVIEW_COLUMN_WIDTHS[index + 1],
      Math.max(MIN_REVIEW_COLUMN_WIDTHS[index], startWidths[index] + deltaPercent)
    );
    const next = [...startWidths];
    next[index] = left;
    next[index + 1] = pairTotal - left;
    setColumnWidths(next);
  }

  function beginColumnResize(event, index) {
    event.preventDefault();
    event.stopPropagation();
    const width = headerColumnsRef.current?.getBoundingClientRect().width || 1;
    const startX = event.clientX;
    const startWidths = [...columnWidths];
    const move = (nextEvent) => resizeColumns(index, startWidths, ((nextEvent.clientX - startX) / width) * 100);
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  function resizeColumnWithKeyboard(event, index) {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    resizeColumns(index, columnWidths, event.key === 'ArrowLeft' ? -1.5 : 1.5);
  }

  function resetColumnWidths() {
    setColumnWidths([...DEFAULT_REVIEW_COLUMN_WIDTHS]);
  }

  async function previewDecision(item, speaker, draft) {
    const submission = {
      schema_version: "transcribe-audio.identity-review-submission.v1",
      submission_id: operationId("submission"),
      queue_item_id: item.queue_item_id,
      conversation_id: item.conversation_id,
      proposal_id: speaker.proposal_id,
      action: draft.action,
      expected_projection_version: item.projection_version,
      decision_payload: {
        speaker_ref: speaker.speaker_ref,
        ...(draft.action === "choose_existing_person" ? { person_id: draft.personId } : {})
      },
      comment: draft.comment,
      idempotency_key: operationId("identity-review"),
      reviewer: "operator",
      decided_at: new Date().toISOString()
    };
    setDecisionState({ status: "loading", message: "Preparing preview…" });
    try {
      const next = await fetchJson(`/api/identity-review/items/${encodeURIComponent(item.queue_item_id)}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submission)
      });
      setPendingSubmission(submission);
      setPreview(next);
      setDecisionState({ status: "previewed", message: `Preview ready for Speaker ${speaker.speaker_ref}. Nothing has been saved.` });
    } catch (error) {
      setDecisionState({ status: "error", message: `Preview failed: ${error.message}` });
    }
  }

  async function recordDecision(item) {
    if (!pendingSubmission) return;
    setDecisionState({ status: "loading", message: "Saving correction…" });
    try {
      const receipt = await fetchJson(`/api/identity-review/items/${encodeURIComponent(item.queue_item_id)}/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pendingSubmission)
      });
      const params = new URLSearchParams({ limit: "100" });
      if (query.trim()) params.set("q", query.trim());
      if (stateFilter) params.set("state", stateFilter);
      setPayload(await fetchJson(`/api/identity-review?${params}`));
      setPreview(receipt.effect_preview);
      setPendingSubmission(null);
      setDecisionState({ status: "recorded", message: "Correction saved to the local review history." });
    } catch (error) {
      setDecisionState({ status: "error", message: `Correction rejected: ${error.message}` });
    }
  }

  return (
    <section className="recording-review" aria-label="Recording review queue">
      <div className="recording-toolbar">
        <label>
          <span>Search</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Title, filename, person, or event" />
        </label>
        <label>
          <span>Current queue</span>
          <select value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}>
            <option value="">All recordings</option>
            <option value="unreviewed">Needs current review</option>
            <option value="unresolved">Current unresolved</option>
            <option value="reviewed">Current review saved</option>
          </select>
        </label>
        <label>
          <span>Sort</span>
          <select
            value={`${sort.key}:${sort.direction}`}
            onChange={(event) => {
              const [key, direction] = event.target.value.split(":");
              setSort({ key, direction });
            }}
          >
            <option value="date:desc">Newest first</option>
            <option value="date:asc">Oldest first</option>
            <option value="title:asc">Title A–Z</option>
            <option value="title:desc">Title Z–A</option>
            <option value="duration:desc">Longest first</option>
            <option value="duration:asc">Shortest first</option>
            <option value="speakers:desc">Most speakers</option>
            <option value="turns:desc">Most turns</option>
            <option value="status:desc">Most matched</option>
          </select>
        </label>
      </div>

      <div className="recording-queue-status" role="status">
        <span><strong>{payload.total || 0} recordings</strong> · {loadState.message}</span>
        <button aria-label="Reset recording column widths" onClick={resetColumnWidths} title="Reset column widths" type="button"><Icon name="columnsReset" size={16} /></button>
      </div>
      <div className="recording-list-head" role="row">
        <span aria-hidden="true" className="recording-head-gutter" />
        <div className="recording-head-columns" ref={headerColumnsRef} style={{ "--recording-columns": columnTemplate }}>
          {REVIEW_COLUMNS.map((column, index) => {
            const active = sort.key === column.key;
            return (
              <div aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"} className={`recording-head-cell${active ? " active" : ""}`} key={column.key} role="columnheader">
                <button aria-label={`Sort by ${column.label}`} onClick={() => sortBy(column.key)} type="button">
                  <span>{column.label}</span>
                  <Icon name={active ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={14} />
                </button>
                {index < REVIEW_COLUMNS.length - 1 && (
                  <span
                    aria-label={`Resize ${column.label} column`}
                    aria-orientation="vertical"
                    aria-valuemax={Math.round(columnWidths[index] + columnWidths[index + 1] - MIN_REVIEW_COLUMN_WIDTHS[index + 1])}
                    aria-valuemin={MIN_REVIEW_COLUMN_WIDTHS[index]}
                    aria-valuenow={Math.round(columnWidths[index])}
                    aria-valuetext={`${Math.round(columnWidths[index])}% relative width`}
                    className="column-resizer"
                    onDoubleClick={resetColumnWidths}
                    onKeyDown={(event) => resizeColumnWithKeyboard(event, index)}
                    onPointerDown={(event) => beginColumnResize(event, index)}
                    role="separator"
                    tabIndex={0}
                    title={`Drag to resize ${column.label}; double-click to reset`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
      <div className="recording-list">
        {items.map((item) => {
          const expanded = expandedId === item.queue_item_id;
          const display = item.display || {};
          const mediaUrl = display.media_url || item.speakers?.find((speaker) => speaker.audio?.media_url)?.audio?.media_url || "";
          const playbackUrl = mediaUrl
            ? `${mediaUrl}${mediaUrl.includes("?") ? "&" : "?"}playback=mp3&end_ms=${Math.max(1000, Number(display.duration_ms || 0))}`
            : "";
          return (
            <article className={`recording-row${expanded ? " expanded" : ""}`} key={item.queue_item_id}>
              <button aria-expanded={expanded} className="recording-summary" onClick={() => setExpandedId(expanded ? "" : item.queue_item_id)} type="button">
                <Icon name={expanded ? "chevronDown" : "chevronRight"} size={17} />
                <span className="recording-summary-columns" style={{ "--recording-columns": columnTemplate }}>
                  <span className="recording-title"><strong>{recordingTitle(item)}</strong><small>{item.original_recording_filename}</small></span>
                  <time>{formatDate(display.recorded_at || item.created_at)}</time>
                  <span className="recording-duration">{formatDuration(display.duration_ms)}</span>
                  <span className="recording-speakers">{item.speakers?.length || 0}</span>
                  <span className="recording-turns">{display.utterance_count || 0}</span>
                  <span className={`recording-state${operatorReview(item) ? " reviewed" : ` ${item.review_state}`}`} title={reviewStatus(item)}>{reviewStatus(item)}</span>
                </span>
              </button>
              {expanded && (
                <div className="recording-expanded">
                  <div className="recording-player">
                    <div><strong>Recording</strong><span>Select any excerpt below to seek.</span></div>
                    {playbackUrl
                      ? <audio aria-label={`${recordingTitle(item)} audio`} controls preload="metadata" ref={audioRef} src={playbackUrl} />
                      : <p className="muted">Audio is unavailable for this recording.</p>}
                  </div>
                  <div className="speaker-table-head" aria-hidden="true">
                    <span>Speaker / automated + reviewed ID</span><span>Diarization</span><span>Correction</span>
                  </div>
                  {(item.speakers || []).map((speaker) => {
                    const diarization = (display.diarization || []).find((value) => value.speaker_ref === speaker.speaker_ref) || {};
                    const fallbackAudio = speaker.audio || {};
                    const segments = diarization.sample_segments?.length
                      ? diarization.sample_segments
                      : [{ start_ms: fallbackAudio.start_ms, end_ms: fallbackAudio.end_ms, text: "Representative sample" }];
                    const active = pendingSubmission?.proposal_id === speaker.proposal_id;
                    return (
                      <SpeakerCorrectionRow
                        decisionState={active ? decisionState : { status: "idle", message: "" }}
                        diarization={diarization}
                        item={item}
                        key={speaker.proposal_id}
                        onPreview={previewDecision}
                        onRecord={() => recordDecision(item)}
                        onSeek={seekTo}
                        pending={active}
                        preview={active ? preview : null}
                        reviewed={reviewedOutcome(item, speaker.speaker_ref)}
                        segments={segments}
                        speaker={speaker}
                      />
                    );
                  })}
                  <p className="mixed-speaker-note">A diarization label can contain more than one person. Use “Mixed speaker” or “Split label” when the excerpts disagree.</p>
                </div>
              )}
            </article>
          );
        })}
        {!items.length && <p className="recording-empty">No recordings match these filters.</p>}
      </div>
    </section>
  );
}

function SpeakerCorrectionRow({ item, speaker, reviewed, diarization, segments, onSeek, onPreview, onRecord, pending, preview, decisionState }) {
  const [action, setAction] = useState("");
  const [personId, setPersonId] = useState(speaker.best_guess?.person_id || "");
  const [comment, setComment] = useState("");
  const draft = { action, personId, comment };
  return (
    <section className="speaker-correction-row">
      <div className="speaker-identity">
        <strong>Speaker {speaker.speaker_ref}</strong>
        <span>Automated: {speaker.best_guess?.label || "Unresolved"}{speaker.best_guess?.strength != null ? ` · ${strength(speaker.best_guess.strength)}` : ""}</span>
        {reviewed && <span className={`speaker-reviewed ${reviewed.outcome}`}><Icon name="reviewed" size={14} />Prior review: {reviewed.label}</span>}
        {!!speaker.alternatives?.length && <small>Also: {speaker.alternatives.slice(0, 2).map((candidate) => candidate.label).join(", ")}</small>}
      </div>
      <div className="speaker-diarization">
        <span>{diarization.utterance_count || 0} turns · {formatDuration(diarization.talk_time_ms)}</span>
        <div className="speaker-excerpts">
          {segments.filter((segment) => segment.start_ms != null).map((segment, index) => (
            <button aria-label={`Play Speaker ${speaker.speaker_ref} at ${formatTime(segment.start_ms)}`} key={`${speaker.proposal_id}-${segment.start_ms}-${index}`} onClick={() => onSeek(segment.start_ms)} title={segment.text || "Play excerpt"} type="button">
              <Icon name="play" size={15} /><time>{formatTime(segment.start_ms)}</time><span>{segment.text || "Play excerpt"}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="speaker-correction">
        <label><span>Correction</span><select value={action} onChange={(event) => setAction(event.target.value)}><option value="">No correction selected</option>{SPEAKER_ACTIONS.map((value) => <option key={value} value={value}>{SPEAKER_ACTION_LABELS[value]}</option>)}</select></label>
        {action === "choose_existing_person" && <label><span>Person ID</span><input required value={personId} onChange={(event) => setPersonId(event.target.value)} /></label>}
        <label className="speaker-comment"><span>Note</span><input onChange={(event) => setComment(event.target.value)} placeholder="Optional review note" value={comment} /></label>
        <div className="speaker-actions">
          <button aria-label={`Preview correction for Speaker ${speaker.speaker_ref}`} disabled={!action || decisionState.status === "loading" || (action === "choose_existing_person" && !personId.trim())} onClick={() => onPreview(item, speaker, draft)} title="Preview correction" type="button"><Icon name="preview" /></button>
          <button aria-label={`Save correction for Speaker ${speaker.speaker_ref}`} className="save" disabled={!pending || !preview || decisionState.status === "loading"} onClick={onRecord} title="Save correction" type="button"><Icon name="record" /></button>
        </div>
        {decisionState.message && <p className={`speaker-decision-message ${decisionState.status}`} role="status">{decisionState.message}</p>}
        {preview && <EffectPreview preview={preview} />}
      </div>
    </section>
  );
}

function EffectPreview({ preview }) {
  return (
    <div className="effect-preview" aria-label="Exact effect preview">
      <header><strong>Preview only</strong><span>{preview.provider_write_count} provider writes · {preview.raw_deletion_count} deletions</span></header>
      {(preview.proposed_effects || []).map((effect, index) => <p key={`${effect.effect_type}-${index}`}>{label(effect.effect_type)}</p>)}
      {(preview.warnings || []).map((warning) => <p key={warning}>{warning}</p>)}
    </div>
  );
}

function identityKindLabel(value) {
  if (value === "canonical_person") return "Person";
  if (value === "local_contact") return "Contact";
  if (value === "reviewed_speaker") return "Review name";
  return "Record";
}

function counted(value, singular, plural = `${singular}s`) {
  const count = Number(value) || 0;
  return `${count} ${count === 1 ? singular : plural}`;
}

function identitySummary(item) {
  if (item.identity_kind === "reviewed_speaker") {
    return `${counted(item.speaker_review_count, "speaker review")} · ${counted(item.recording_count, "recording")}`;
  }
  if (item.identity_kind === "local_contact") {
    const email = item.contact_methods?.find((method) => method.kind === "email")?.value || "Unlinked local contact";
    const graphCount = (item.role_hypotheses?.length || 0) + (item.relationship_hypotheses?.length || 0);
    const recording = item.recording_count ? ` · ${counted(item.recording_count, "recording")}` : "";
    const graph = graphCount ? ` · ${counted(graphCount, "graph lead")}` : "";
    return `${email}${recording}${graph}`;
  }
  return `${item.source_records?.length || 0} sources${item.roles?.length ? ` · ${item.roles.length} roles` : ""}`;
}

const DIRECTORY_COLUMNS = [
  { key: "name", label: "Person", width: 20 },
  { key: "organization", label: "Organization and role", width: 20 },
  { key: "transcript", label: "Transcripts", width: 12 },
  { key: "calendar", label: "Calendar", width: 12 },
  { key: "email", label: "Email", width: 12 },
  { key: "last", label: "Last interaction", width: 13 },
  { key: "health", label: "Identity health", width: 11 }
];

function directoryId(item) {
  return item.entity_id || item.organization_id || item.person_id;
}

function channelSummary(item, channel) {
  const summary = item.activity_summary?.[channel] || {};
  const counts = [
    summary.confirmed_count ? `${summary.confirmed_count} confirmed` : "",
    summary.proposed_count ? `${summary.proposed_count} proposed` : ""
  ].filter(Boolean).join(" · ") || "No observations";
  return <span className="directory-channel"><strong>{counts}</strong><small>{summary.last_at ? formatDate(summary.last_at) : label(summary.coverage_state || "not queried")}</small></span>;
}

function directorySortValue(item, key) {
  if (key === "name") return item.sort_name || item.display_name || item.primary_name || "";
  if (key === "organization") {
    if (item.entity_kind === "organization") return item.affiliated_person_ids?.length || 0;
    return item.primary_affiliation?.primary_name || item.organizations?.[0]?.primary_name || item.organization_type || "";
  }
  if (["transcript", "calendar", "email"].includes(key)) {
    const summary = item.activity_summary?.[key] || {};
    return Number(summary.confirmed_count || 0) + Number(summary.proposed_count || 0);
  }
  if (key === "last") return Date.parse(item.last_interaction_at || "") || 0;
  if (key === "health") return Number(item.identity_health?.requires_review || 0) * 1000 + Number(item.identity_health?.source_record_count || 0);
  return "";
}

function affiliationRoleSummary(affiliation) {
  const roleTypes = affiliation?.role_types?.length
    ? affiliation.role_types
    : (affiliation?.roles || []).map((role) => role.role_type).filter(Boolean);
  if (!roleTypes.length) return affiliation?.status === "proposed"
    ? "Organization observed · proposed"
    : "No role asserted";
  const visible = roleTypes.slice(0, 2).map(label).join(" / ");
  return roleTypes.length > 2 ? `${visible} · +${roleTypes.length - 2} roles` : visible;
}

function roleValidity(role) {
  if (!role?.starts_at && !role?.ends_at) return "Not established";
  const start = role.starts_at ? formatDate(role.starts_at) : "Earlier";
  const end = role.ends_at ? formatDate(role.ends_at) : "Current";
  return `${start} – ${end}`;
}

function PeopleView() {
  const [payload, setPayload] = useState({ items: [], counts: {} });
  const [repairPayload, setRepairPayload] = useState({ items: [], counts: {} });
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("approvals");
  const [view, setView] = useState("people");
  const [sort, setSort] = useState({ key: "last", direction: "desc" });
  const [widths, setWidths] = useState(DIRECTORY_COLUMNS.map((column) => column.width));
  const [expandedId, setExpandedId] = useState("");
  const [loadState, setLoadState] = useState({ status: "loading", message: "Loading directory…" });
  const [refreshToken, setRefreshToken] = useState(0);
  const tableRef = useRef(null);
  const requestedView = mode === "directory" ? view : "people";

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      const params = new URLSearchParams({ limit: "500", view: requestedView });
      if (query.trim()) params.set("q", query.trim());
      try {
        const [next, repairs] = await Promise.all([
          fetchJson(`/api/people?${params}`),
          mode === "repairs"
            ? fetchJson(`/api/person-repairs?${new URLSearchParams({ limit: "500", ...(query.trim() ? { q: query.trim() } : {}) })}`)
            : Promise.resolve({ items: [], counts: {} })
        ]);
        if (cancelled) return;
        setPayload(next);
        if (mode === "repairs") setRepairPayload({ ...repairs, loaded: true });
        setExpandedId((current) => next.items?.some((item) => directoryId(item) === current) ? current : "");
        setLoadState({ status: "live", message: "Canonical local index" });
      } catch (error) {
        if (cancelled) return;
        setPayload({ items: [], counts: {} });
        setLoadState({ status: "preview", message: `Directory unavailable: ${error.message}` });
      }
    }, query.trim() ? 180 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [mode, query, requestedView, refreshToken]);

  const approvalRows = useMemo(() => flattenDirectoryReviewRows(payload.items), [payload.items]);
  const acceptedRows = useMemo(() => flattenDirectoryReviewRows(payload.items, "accepted"), [payload.items]);
  const items = useMemo(() => [...payload.items].sort((left, right) => {
    const leftValue = directorySortValue(left, sort.key);
    const rightValue = directorySortValue(right, sort.key);
    const comparison = typeof leftValue === "number"
      ? leftValue - rightValue
      : String(leftValue).localeCompare(String(rightValue));
    return (sort.direction === "asc" ? comparison : -comparison)
      || String(left.sort_name || left.display_name || left.primary_name || "").localeCompare(String(right.sort_name || right.display_name || right.primary_name || ""));
  }), [payload.items, sort]);
  const directoryCount = (payload.counts?.people || 0) + (payload.counts?.unresolved_groups || 0);

  function sortBy(key) {
    setSort((current) => current.key === key
      ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key, direction: key === "name" || key === "organization" ? "asc" : "desc" });
  }

  function resize(index, clientX, startX, startWidths) {
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    const delta = ((clientX - startX) / tableWidth) * 100;
    const pair = startWidths[index] + startWidths[index + 1];
    const nextLeft = Math.max(7, Math.min(pair - 7, startWidths[index] + delta));
    const next = [...startWidths];
    next[index] = nextLeft;
    next[index + 1] = pair - nextLeft;
    setWidths(next);
  }

  function beginResize(event, index) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidths = [...widths];
    const move = (moveEvent) => resize(index, moveEvent.clientX, startX, startWidths);
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function resizeWithKeyboard(event, index) {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    resize(index, tableWidth * (event.key === "ArrowLeft" ? -0.015 : 0.015), 0, widths);
  }

  return (
    <section className="identity-surface directory-surface" aria-label="People and organizations">
      <div className="directory-mode-bar">
        <nav aria-label="Contacts work modes">
          <button aria-current={mode === "approvals" ? "page" : undefined} onClick={() => setMode("approvals")} type="button">Approval rows <span>{approvalRows.length}</span></button>
          <button aria-current={mode === "repairs" ? "page" : undefined} onClick={() => setMode("repairs")} type="button">Repairs {repairPayload.loaded && <span>{(repairPayload.counts?.all || 0) + acceptedRows.length}</span>}</button>
          <button aria-current={mode === "directory" ? "page" : undefined} onClick={() => setMode("directory")} type="button">Directory <span>{directoryCount}</span></button>
        </nav>
      </div>
      {mode === "approvals" ? <>
        <div className="directory-approval-toolbar">
          <label><span>Search approvals</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Contact, organization, role, or source" /></label>
          <div className={`identity-load-state ${loadState.status}`} role="status"><strong>{approvalRows.length}</strong><span>{loadState.message}</span></div>
        </div>
        <DirectoryApprovalTable onReviewed={() => setRefreshToken((value) => value + 1)} reviewRows={approvalRows} reviewTargets={payload.review_targets || { people: [], organizations: [] }} />
      </> : mode === "repairs" ? <>
        <div className="directory-approval-toolbar">
          <label><span>Search repairs</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Person, organization, issue, or source" /></label>
          <div className={`identity-load-state ${loadState.status}`} role="status"><strong>{(repairPayload.counts?.all || 0) + acceptedRows.length}</strong><span>{loadState.message}</span></div>
        </div>
        <PersonRepairTable onRepaired={() => setRefreshToken((value) => value + 1)} repairs={repairPayload.items || []} />
        <section className="accepted-decision-repairs">
          <h3>Accepted organization &amp; role decisions <span>{acceptedRows.length}</span></h3>
          <DirectoryApprovalTable onReviewed={() => setRefreshToken((value) => value + 1)} reviewRows={acceptedRows} reviewTargets={payload.review_targets || { people: [], organizations: [] }} />
        </section>
      </> : <>
        <div className="directory-toolbar">
          <label><span>Search directory</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name, organization, source, or recording" /></label>
          <nav aria-label="Directory views">
            {[["people", `All contacts ${directoryCount}`], ["organizations", `Organizations ${payload.counts?.organizations || 0}`], ["unresolved", `Unresolved ${payload.counts?.unresolved_groups || 0}`]].map(([key, text]) => <button aria-current={view === key ? "page" : undefined} key={key} onClick={() => setView(key)} type="button">{text}</button>)}
          </nav>
          <div className={`identity-load-state ${loadState.status}`} role="status"><strong>{items.length}</strong><span>{loadState.message}</span></div>
        </div>
        <div className="directory-table-wrap">
        <table className="directory-table" ref={tableRef}>
          <colgroup>{widths.map((width, index) => <col key={DIRECTORY_COLUMNS[index].key} style={{ width: `${width}%` }} />)}</colgroup>
          <thead><tr>{DIRECTORY_COLUMNS.map((column, index) => {
            const active = sort.key === column.key;
            const columnLabel = view === "organizations" && column.key === "name"
              ? "Organization"
              : view === "organizations" && column.key === "organization"
                ? "People and status"
                : column.label;
            return <th aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"} key={column.key}>
              <button className="directory-sort" aria-label={`Sort by ${columnLabel}`} onClick={() => sortBy(column.key)} type="button"><span>{columnLabel}</span><Icon name={active ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={13} /></button>
              {index < DIRECTORY_COLUMNS.length - 1 && <span aria-label={`Resize ${columnLabel} column`} aria-orientation="vertical" aria-valuenow={Math.round(widths[index])} className="directory-resizer" onDoubleClick={() => setWidths(DIRECTORY_COLUMNS.map((value) => value.width))} onKeyDown={(event) => resizeWithKeyboard(event, index)} onPointerDown={(event) => beginResize(event, index)} role="separator" tabIndex={0} />}
            </th>;
          })}</tr></thead>
          <tbody>{items.map((item) => {
            const id = directoryId(item);
            const expanded = expandedId === id;
            const affiliation = item.primary_affiliation || item.organizations?.[0];
            const health = item.identity_health || {};
            return <Fragment key={id}>
              <tr className={expanded ? "directory-row expanded" : "directory-row"}>
                <td><button className="directory-expand" aria-expanded={expanded} aria-label={`${expanded ? "Collapse" : "Expand"} ${personTargetDisplayLabel(item)}`} onClick={() => setExpandedId(expanded ? "" : id)} type="button"><Icon name={expanded ? "chevronDown" : "chevronRight"} size={14} /><span><strong>{personTargetDisplayLabel(item)}</strong><small>{item.entity_kind === "unresolved_group" ? `${health.member_count || 0} separate records · unresolved` : label(item.resolution_state)}</small></span></button></td>
                <td>{view === "organizations"
                  ? <span className="directory-cell-stack"><strong>{counted(item.affiliated_person_ids?.length, "linked person")}</strong><small>{label(item.resolution_state)} organization</small></span>
                  : <span className="directory-cell-stack"><strong>{affiliation?.primary_name || "—"}</strong><small>{affiliation ? `${affiliationRoleSummary(affiliation)}${item.additional_organization_count ? ` · +${item.additional_organization_count} ${item.additional_organization_count === 1 ? "organization" : "organizations"}` : ""}` : "No accepted affiliation"}</small></span>}</td>
                <td>{channelSummary(item, "transcript")}</td>
                <td>{channelSummary(item, "calendar")}</td>
                <td>{channelSummary(item, "email")}</td>
                <td>{item.last_interaction_at ? formatDate(item.last_interaction_at) : "No dated activity"}</td>
                <td><span className="directory-cell-stack"><strong>{health.requires_review ? "Review needed" : "Resolved"}</strong><small>{health.source_record_count ?? health.source_name_count ?? 0} sources · {health.conflict_count || 0} conflicts</small></span></td>
              </tr>
              {expanded && <tr className="directory-expanded-row"><td colSpan={DIRECTORY_COLUMNS.length}><DirectoryDetail item={item} onReviewed={() => setRefreshToken((value) => value + 1)} reviewTargets={payload.review_targets || { people: [], organizations: [] }} /></td></tr>}
            </Fragment>;
          })}</tbody>
        </table>
        {!items.length && <p className="muted contacts-empty">No directory rows match this view.</p>}
        </div>
      </>}
    </section>
  );
}

const DIRECTORY_APPROVAL_COLUMNS = [
  { key: "contact", label: "Contact", width: 13, sortable: true },
  { key: "proposal", label: "Proposed affiliation / role", width: 16, sortable: true },
  { key: "evidence", label: "Source basis", width: 14, sortable: true },
  { key: "history", label: "History", width: 10, sortable: true },
  { key: "person", label: "Person target", width: 15, sortable: false },
  { key: "organizationTarget", label: "Organization target", width: 17, sortable: false },
  { key: "role", label: "Role title", width: 8, sortable: true },
  { key: "actions", label: "Actions", width: 7, sortable: false }
];

function approvalOrganization(lead) {
  return lead.organization || lead.counterpart_label || "";
}

function activityCount(item, channel) {
  const summary = item.activity_summary?.[channel] || {};
  return Number(summary.confirmed_count || 0) + Number(summary.proposed_count || 0);
}

function approvalSortValue(row, key) {
  const { item, lead } = row;
  if (key === "contact") return item.sort_name || item.display_name || item.primary_name || "";
  if (key === "proposal") return `${lead.hypothesis_kind || ""} ${approvalOrganization(lead)} ${lead.display_value || ""}`;
  if (key === "evidence") {
    const source = lead.source_records?.[0] || {};
    return `${source.provider || ""} ${source.profile || ""} ${source.match_basis || ""} ${lead.observation_count || 0}`;
  }
  if (key === "history") return ["transcript", "calendar", "email"].reduce((sum, channel) => sum + activityCount(item, channel), 0);
  if (key === "role") return lead.display_value || "";
  return "";
}

function reviewSourceSummary(lead) {
  const sources = lead.source_records || [];
  const source = sources[0] || {};
  const provider = source.provider ? label(source.provider) : label(lead.evidence_source || "source");
  const match = source.match_basis ? label(source.match_basis) : "Provider evidence";
  const observations = Number(lead.observation_count || sources.length || 0);
  return `${provider}${source.profile ? `/${source.profile}` : ""} · ${match} · ${observations}`;
}

function reviewSourceTitle(lead) {
  const sourceDetails = (lead.source_records || []).map((source) => [source.provider, source.profile, source.record_type, source.match_basis].filter(Boolean).join(" / ")).join("; ");
  return [lead.basis, lead.why_not_accepted, sourceDetails].filter(Boolean).join(" — ");
}

function ActivityHistory({ item }) {
  return <span className="directory-approval-history" aria-label="Transcript, calendar, and email history">
    {["transcript", "calendar", "email"].map((channel) => {
      const summary = item.activity_summary?.[channel] || {};
      const count = activityCount(item, channel);
      const last = summary.last_at ? `; last ${formatDate(summary.last_at)}` : "";
      return <span key={channel} title={`${label(channel)}: ${count} observations${last}`}><Icon name={channel} size={13} /><b>{count}</b></span>;
    })}
  </span>;
}

function DirectoryApprovalTable({ onReviewed, reviewRows, reviewTargets }) {
  const [sort, setSort] = useState({ key: "history", direction: "desc" });
  const [widths, setWidths] = useState(DIRECTORY_APPROVAL_COLUMNS.map((column) => column.width));
  const tableRef = useRef(null);
  const ordered = useMemo(() => [...reviewRows].sort((left, right) => {
    const leftValue = approvalSortValue(left, sort.key);
    const rightValue = approvalSortValue(right, sort.key);
    const comparison = typeof leftValue === "number"
      ? leftValue - rightValue
      : String(leftValue).localeCompare(String(rightValue));
    return (sort.direction === "asc" ? comparison : -comparison)
      || String(left.item.primary_name || "").localeCompare(String(right.item.primary_name || ""))
      || String(left.lead.hypothesis_id).localeCompare(String(right.lead.hypothesis_id));
  }), [reviewRows, sort]);

  function sortBy(column) {
    if (!column.sortable) return;
    setSort((current) => current.key === column.key
      ? { key: column.key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key: column.key, direction: column.key === "history" ? "desc" : "asc" });
  }

  function resize(index, clientX, startX, startWidths) {
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    const pair = startWidths[index] + startWidths[index + 1];
    const nextLeft = Math.max(5, Math.min(pair - 5, startWidths[index] + ((clientX - startX) / tableWidth) * 100));
    const next = [...startWidths];
    next[index] = nextLeft;
    next[index + 1] = pair - nextLeft;
    setWidths(next);
  }

  function beginResize(event, index) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidths = [...widths];
    const move = (moveEvent) => resize(index, moveEvent.clientX, startX, startWidths);
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }

  function resizeWithKeyboard(event, index) {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    resize(index, tableWidth * (event.key === "ArrowLeft" ? -0.012 : 0.012), 0, widths);
  }

  return <div className="directory-table-wrap directory-approval-wrap">
    <table className="directory-approval-table" ref={tableRef}>
      <colgroup>{widths.map((width, index) => <col key={DIRECTORY_APPROVAL_COLUMNS[index].key} style={{ width: `${width}%` }} />)}</colgroup>
      <thead><tr>{DIRECTORY_APPROVAL_COLUMNS.map((column, index) => {
        const active = sort.key === column.key;
        return <th aria-sort={column.sortable ? (active ? (sort.direction === "asc" ? "ascending" : "descending") : "none") : undefined} key={column.key}>
          <button className="directory-approval-sort" disabled={!column.sortable} onClick={() => sortBy(column)} type="button"><span>{column.label}</span>{column.sortable && <Icon name={active ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={12} />}</button>
          {index < DIRECTORY_APPROVAL_COLUMNS.length - 1 && <span aria-label={`Resize ${column.label} column`} aria-orientation="vertical" aria-valuenow={Math.round(widths[index])} className="directory-resizer" onDoubleClick={() => setWidths(DIRECTORY_APPROVAL_COLUMNS.map((value) => value.width))} onKeyDown={(event) => resizeWithKeyboard(event, index)} onPointerDown={(event) => beginResize(event, index)} role="separator" tabIndex={0} />}
        </th>;
      })}</tr></thead>
      <tbody>{ordered.map(({ item, lead, row_id: rowId }) => <DirectoryReviewLeadRow approvalMode item={item} key={rowId} lead={lead} onReviewed={onReviewed} reviewTargets={reviewTargets} />)}</tbody>
    </table>
    {!ordered.length && <p className="muted contacts-empty">No approval rows match this search.</p>}
  </div>;
}

const PERSON_REPAIR_COLUMNS = [
  { key: "kind", label: "Issue", width: 15 },
  { key: "person", label: "Current person", width: 22 },
  { key: "repair", label: "Proposed repair", width: 25 },
  { key: "evidence", label: "Evidence", width: 29 },
  { key: "actions", label: "Actions", width: 9, sortable: false }
];

function personRepairValue(repair, key) {
  if (key === "kind") return repair.repair_kind || "";
  if (key === "person") return repair.display_name || repair.current_primary_name || "";
  if (key === "repair") return repair.suggested_primary_name || repair.display_name || "";
  if (key === "evidence") return `${repair.evidence?.providers?.join(" ") || ""} ${repair.evidence?.organization_names?.join(" ") || ""}`;
  return "";
}

function personRepairLabel(kind) {
  if (kind === "canonical_name") return "Canonical name";
  if (kind === "possible_duplicate") return "Possible duplicate";
  return "Identity ambiguity";
}

function participantRepairSummary(participant) {
  const providers = (participant.evidence?.providers || []).map(label).join("/");
  const organizations = (participant.evidence?.organization_names || []).join("; ");
  return [
    participant.display_name,
    providers,
    organizations,
    `${participant.evidence?.source_count || 0} sources`
  ].filter(Boolean).join(" · ");
}

function PersonRepairRow({ repair, onRepaired }) {
  const candidates = repair.candidate_names || [];
  const participants = repair.participants || [];
  const [replacement, setReplacement] = useState(repair.suggested_primary_name || candidates[0] || "");
  const [targetPersonId, setTargetPersonId] = useState(repair.person_ids?.[0] || "");
  const [decision, setDecision] = useState({ status: "idle", message: "" });
  const inFlight = useRef(createInFlightGate());
  const action = repair.repair_kind === "possible_duplicate" ? "merge_people" : "correct_name";
  const actionable = (repair.allowed_actions || []).includes(action);
  const evidence = repair.repair_kind === "possible_duplicate"
    ? participants.map((participant) => [(participant.evidence?.providers || []).map(label).join("/"), (participant.evidence?.organization_names || []).join("; "), `${participant.evidence?.source_count || 0} sources`].filter(Boolean).join(" · ")).join(" / ")
    : `${repair.evidence?.source_count || 0} sources${repair.evidence?.providers?.length ? ` · ${repair.evidence.providers.map(label).join("/")}` : ""}${repair.evidence?.organization_names?.length ? ` · ${repair.evidence.organization_names.join("; ")}` : ""}`;

  async function applyRepair() {
    if (!inFlight.current.begin()) return;
    setDecision({ status: "loading", message: "Saving…" });
    const submission = {
      schema_version: "transcribe-audio.person-identity-repair-submission.v1",
      repair_id: repair.repair_id,
      repair_kind: repair.repair_kind,
      action,
      expected_content_sha256: repair.content_sha256,
      reviewer: "operator",
      decided_at: new Date().toISOString(),
      idempotency_key: operationId(`person-repair-${repair.repair_id}-${action}`),
      ...(action === "correct_name"
        ? { person_id: repair.person_id, replacement_primary_name: replacement }
        : { target_person_id: targetPersonId })
    };
    try {
      await fetchJson(`/api/person-repairs/${encodeURIComponent(repair.repair_id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submission)
      });
      setDecision({ status: "saved", message: action === "merge_people" ? "Merged" : "Corrected" });
      onRepaired();
    } catch (error) {
      if (error.status === 409) {
        setDecision({ status: "loading", message: "Refreshing current state…" });
        onRepaired();
      } else {
        setDecision({ status: "error", message: error.message });
      }
    } finally {
      inFlight.current.end();
    }
  }

  const current = repair.repair_kind === "possible_duplicate"
    ? participants.map(participantRepairSummary).join(" / ")
    : repair.current_primary_name;
  return <tr className="person-repair-row" data-repair-status={decision.status}>
    <td><span className="directory-approval-text" title={repair.reason}>{personRepairLabel(repair.repair_kind)}</span></td>
    <td><span className="directory-approval-text" title={current}>{current}</span></td>
    <td>{repair.repair_kind === "canonical_name"
      ? <select aria-label={`Corrected name for ${repair.current_primary_name}`} disabled={decision.status === "loading"} onChange={(event) => setReplacement(event.target.value)} value={replacement}>{candidates.map((candidate) => <option key={candidate} value={candidate}>{candidate}</option>)}</select>
      : repair.repair_kind === "possible_duplicate"
        ? <select aria-label={`Person to keep for ${repair.display_name}`} disabled={decision.status === "loading"} onChange={(event) => setTargetPersonId(event.target.value)} value={targetPersonId}>{participants.map((participant) => <option key={participant.person_id} value={participant.person_id}>Keep {participantRepairSummary(participant)}</option>)}</select>
        : <span className="directory-approval-text" title={repair.reason}>Needs person-versus-mailbox review</span>}</td>
    <td><span className="directory-approval-text" title={(repair.evidence?.labels || []).join("; ")}>{evidence}</span></td>
    <td><div className="directory-review-actions" title={decision.message || repair.reason}>
      {decision.status === "error" && <Icon name="warning" size={14} />}
      {actionable
        ? <button aria-label={`${action === "merge_people" ? "Merge" : "Correct"} ${repair.display_name}`} className="accept" disabled={decision.status === "loading" || (action === "correct_name" && !replacement) || (action === "merge_people" && !targetPersonId)} onClick={applyRepair} title={action === "merge_people" ? "Merge into selected person" : "Apply canonical name"} type="button"><Icon name={action === "merge_people" ? "identity" : "reviewed"} size={14} /></button>
        : <span className="directory-approval-none" aria-label="No safe automatic repair">—</span>}
    </div></td>
  </tr>;
}

function PersonRepairTable({ repairs, onRepaired }) {
  const [sort, setSort] = useState({ key: "kind", direction: "asc" });
  const [widths, setWidths] = useState(PERSON_REPAIR_COLUMNS.map((column) => column.width));
  const tableRef = useRef(null);
  const ordered = useMemo(() => [...repairs].sort((left, right) => {
    const comparison = String(personRepairValue(left, sort.key)).localeCompare(String(personRepairValue(right, sort.key)));
    return (sort.direction === "asc" ? comparison : -comparison) || String(left.repair_id).localeCompare(String(right.repair_id));
  }), [repairs, sort]);

  function sortBy(column) {
    if (column.sortable === false) return;
    setSort((current) => current.key === column.key
      ? { key: column.key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key: column.key, direction: "asc" });
  }

  function beginResize(event, index) {
    event.preventDefault();
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    const startX = event.clientX;
    const start = [...widths];
    const move = (nextEvent) => {
      const pair = start[index] + start[index + 1];
      const left = Math.max(6, Math.min(pair - 6, start[index] + ((nextEvent.clientX - startX) / tableWidth) * 100));
      const next = [...start];
      next[index] = left;
      next[index + 1] = pair - left;
      setWidths(next);
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  return <section className="person-repairs">
    <h3>Identity repairs <span>{repairs.length}</span></h3>
    <div className="directory-table-wrap directory-approval-wrap">
      <table className="directory-approval-table" ref={tableRef}>
        <colgroup>{widths.map((width, index) => <col key={PERSON_REPAIR_COLUMNS[index].key} style={{ width: `${width}%` }} />)}</colgroup>
        <thead><tr>{PERSON_REPAIR_COLUMNS.map((column, index) => {
          const active = sort.key === column.key;
          const sortable = column.sortable !== false;
          return <th aria-sort={sortable ? (active ? (sort.direction === "asc" ? "ascending" : "descending") : "none") : undefined} key={column.key}>
            <button className="directory-approval-sort" disabled={!sortable} onClick={() => sortBy(column)} type="button"><span>{column.label}</span>{sortable && <Icon name={active ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={12} />}</button>
            {index < PERSON_REPAIR_COLUMNS.length - 1 && <span aria-label={`Resize ${column.label} column`} aria-orientation="vertical" aria-valuenow={Math.round(widths[index])} className="directory-resizer" onDoubleClick={() => setWidths(PERSON_REPAIR_COLUMNS.map((value) => value.width))} onPointerDown={(event) => beginResize(event, index)} role="separator" tabIndex={0} />}
          </th>;
        })}</tr></thead>
        <tbody>{ordered.map((repair) => <PersonRepairRow key={repair.repair_id} onRepaired={onRepaired} repair={repair} />)}</tbody>
      </table>
      {!ordered.length && <p className="muted contacts-empty">No identity repairs match this search.</p>}
    </div>
  </section>;
}

const DIRECTORY_REVIEW_COLUMNS = [
  { key: "lead", label: "Lead", width: 18 },
  { key: "organization", label: "Organization", width: 17 },
  { key: "person", label: "Person target", width: 16 },
  { key: "organizationTarget", label: "Organization target", width: 17 },
  { key: "title", label: "Role title", width: 15 },
  { key: "state", label: "State", width: 9 },
  { key: "actions", label: "Actions", width: 8 }
];

function reviewLeadValue(lead, key) {
  if (key === "lead") return hypothesisLabel(lead.hypothesis_kind);
  if (key === "organization" || key === "organizationTarget") return lead.organization || lead.counterpart_label || "";
  if (key === "title") return lead.display_value || "";
  if (key === "state") return lead.review_state || "unreviewed";
  return "";
}

function DirectoryReviewLeadRow({ approvalMode = false, item, lead, onReviewed, reviewTargets }) {
  const matchingPerson = findUniqueAcceptedPersonTarget(item, reviewTargets.people);
  const organizationName = lead.organization || lead.counterpart_label || "";
  const matchingOrganization = reviewTargets.organizations?.find((target) => target.label.trim().toLowerCase() === organizationName.trim().toLowerCase());
  const [personTarget, setPersonTarget] = useState(matchingPerson ? `existing:${matchingPerson.id}` : "create");
  const [personName, setPersonName] = useState(item.display_name || "");
  const [organizationTarget, setOrganizationTarget] = useState(matchingOrganization ? `existing:${matchingOrganization.id}` : "create");
  const [roleTitle, setRoleTitle] = useState(lead.display_value || "");
  const [decision, setDecision] = useState({ status: "idle", message: "" });
  const inFlight = useRef(createInFlightGate());

  async function review(action) {
    if (!inFlight.current.begin()) return;
    setDecision({ status: "loading", message: "Saving…" });
    const personMode = personTarget.startsWith("existing:") ? "existing" : "create";
    const personId = personMode === "existing" ? personTarget.slice("existing:".length) : "";
    const organizationMode = organizationTarget.startsWith("existing:") ? "existing" : "create";
    const organizationId = organizationMode === "existing" ? organizationTarget.slice("existing:".length) : "";
    const submission = {
      schema_version: "transcribe-audio.directory-hypothesis-review-submission.v1",
      hypothesis_id: lead.hypothesis_id,
      action,
      expected_projection_version: lead.projection_version,
      source_content_sha256: lead.source_content_sha256,
      reviewer: "operator",
      decided_at: new Date().toISOString(),
      idempotency_key: operationId(`directory-${lead.hypothesis_id}-${action}`),
      person_target: { mode: personMode, ...(personId ? { id: personId } : {}) },
      ...(personMode === "create" ? { person_name: personName.trim() } : {}),
      organization_target: { mode: organizationMode, ...(organizationId ? { id: organizationId } : {}) },
      ...(lead.hypothesis_kind === "contextual_role" ? { role_title: roleTitle.trim() } : {})
    };
    try {
      await fetchJson(`/api/directory-hypotheses/${encodeURIComponent(lead.hypothesis_id)}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submission)
      });
      setDecision({ status: "saved", message: action === "accept" ? "Accepted" : action === "reject" ? "Rejected" : "Deferred" });
      onReviewed();
    } catch (error) {
      let committed = false;
      try {
        const current = await fetchJson("/api/people?limit=500&view=people");
        committed = hasDirectoryReviewDecision(current.items, lead.hypothesis_id, submission.idempotency_key);
      } catch {
        committed = false;
      }
      if (committed) {
        setDecision({ status: "saved", message: action === "accept" ? "Accepted" : action === "reject" ? "Rejected" : "Deferred" });
        onReviewed();
      } else if (error.status === 409) {
        setDecision({ status: "loading", message: "Refreshing current state…" });
        onReviewed();
      } else {
        setDecision({ status: "error", message: error.message });
      }
    } finally {
      inFlight.current.end();
    }
  }

  const disabled = decision.status === "loading";
  const personMode = personTarget.startsWith("existing:") ? "existing" : "create";
  const invalidCreateName = personMode === "create" && !isCompletePersonName(personName, organizationName);
  if (approvalMode) {
    const proposal = lead.hypothesis_kind === "contextual_role"
      ? `Role · ${organizationName}${lead.display_value ? ` · ${lead.display_value}` : ""}`
      : `Affiliation · ${organizationName}`;
    return <tr className="directory-review-row directory-approval-row" data-review-status={decision.status}>
      <td><span className="directory-approval-text" title={item.primary_name || "Unnamed contact"}>{personTargetDisplayLabel(item)}</span></td>
      <td><span className="directory-approval-text" title={proposal}>{proposal}</span></td>
      <td><span className="directory-approval-text" title={reviewSourceTitle(lead)}>{reviewSourceSummary(lead)}</span></td>
      <td><ActivityHistory item={item} /></td>
      <td><span className="directory-person-target"><select aria-label={`Person target for ${item.primary_name || organizationName}`} disabled={disabled} onChange={(event) => setPersonTarget(event.target.value)} title="Choose the canonical person that this approval should update" value={personTarget}><option value="create">Create person</option>{!!reviewTargets.people?.length && <optgroup label={`Accepted people (${reviewTargets.people.length})`}>{reviewTargets.people.map((target) => <option key={target.id} value={`existing:${target.id}`}>{personTargetDisplayLabel(target)}</option>)}</optgroup>}</select>{personMode === "create" && <input aria-invalid={invalidCreateName || undefined} aria-label={`New person name for ${item.primary_name || organizationName}`} disabled={disabled} onChange={(event) => setPersonName(event.target.value)} placeholder="Complete person name" title={invalidCreateName ? "Enter a complete human name, not an email or organization" : "Canonical person name"} value={personName} />}</span></td>
      <td><select aria-label={`Organization target for ${organizationName}`} disabled={disabled} onChange={(event) => setOrganizationTarget(event.target.value)} title="Choose the canonical organization that this approval should update" value={organizationTarget}><option value="create">Create {organizationName}</option>{!!reviewTargets.organizations?.length && <optgroup label={`Accepted organizations (${reviewTargets.organizations.length})`}>{reviewTargets.organizations.map((target) => <option key={target.id} value={`existing:${target.id}`}>{target.label}</option>)}</optgroup>}</select></td>
      <td>{lead.hypothesis_kind === "contextual_role" ? <input aria-label={`Role title for ${organizationName}`} disabled={disabled} onChange={(event) => setRoleTitle(event.target.value)} title="Reviewed role title" value={roleTitle} /> : <span className="directory-approval-none" aria-label="No role title required">—</span>}</td>
      <td><div className="directory-review-actions" title={decision.message || "Accept, reject, or defer this hypothesis"}>
        {decision.status === "error" && <Icon name="warning" size={14} />}
        <button aria-label={`Accept ${proposal} for ${item.primary_name || "contact"}`} className="accept" disabled={disabled || invalidCreateName || lead.review_state === "accepted" || (lead.hypothesis_kind === "contextual_role" && !roleTitle.trim())} onClick={() => review("accept")} title={invalidCreateName ? "Enter a complete person name" : "Accept"} type="button"><Icon name="reviewed" size={14} /></button>
        <button aria-label={`Reject ${proposal} for ${item.primary_name || "contact"}`} className="reject" disabled={disabled || lead.review_state === "rejected"} onClick={() => review("reject")} title="Reject" type="button"><Icon name="reject" size={14} /></button>
        <button aria-label={`Defer ${proposal} for ${item.primary_name || "contact"}`} className="defer" disabled={disabled || lead.review_state === "deferred"} onClick={() => review("defer")} title="Defer" type="button"><Icon name="defer" size={14} /></button>
      </div></td>
    </tr>;
  }
  return <tr className="directory-review-row">
    <td><strong>{lead.hypothesis_kind === "contextual_role" ? "Role" : "Affiliation"}</strong><small>{lead.basis || "Provider observation"}</small></td>
    <td>{organizationName || "—"}</td>
    <td><span className="directory-person-target"><select aria-label={`Person target for ${organizationName}`} disabled={disabled} onChange={(event) => setPersonTarget(event.target.value)} value={personTarget}><option value="create">Create from reviewed contact</option>{!!reviewTargets.people?.length && <optgroup label={`Use accepted person (${reviewTargets.people.length})`}>{reviewTargets.people.map((target) => <option key={target.id} value={`existing:${target.id}`}>{personTargetDisplayLabel(target)}</option>)}</optgroup>}</select>{personMode === "create" && <input aria-invalid={invalidCreateName || undefined} aria-label={`New person name for ${organizationName}`} disabled={disabled} onChange={(event) => setPersonName(event.target.value)} placeholder="Complete person name" value={personName} />}</span></td>
    <td><select aria-label={`Organization target for ${organizationName}`} disabled={disabled} onChange={(event) => setOrganizationTarget(event.target.value)} value={organizationTarget}><option value="create">Create from reviewed proposal: {organizationName}</option>{!!reviewTargets.organizations?.length && <optgroup label={`Use accepted organization (${reviewTargets.organizations.length})`}>{reviewTargets.organizations.map((target) => <option key={target.id} value={`existing:${target.id}`}>{target.label}</option>)}</optgroup>}</select></td>
    <td>{lead.hypothesis_kind === "contextual_role" ? <input aria-label={`Role title for ${organizationName}`} disabled={disabled} onChange={(event) => setRoleTitle(event.target.value)} value={roleTitle} /> : <span className="muted">No role asserted</span>}</td>
    <td><span className={`directory-review-state ${decision.status === "error" ? "error" : lead.review_state || "unreviewed"}`} title={decision.message}>{decision.message || (lead.review_state === "accepted" ? "Accepted" : lead.review_state === "rejected" ? "Rejected" : lead.review_state === "deferred" ? "Deferred" : "Needs review")}</span></td>
    <td><div className="directory-review-actions">
      <button aria-label="Accept organization or role lead" className="accept" disabled={disabled || invalidCreateName || lead.review_state === "accepted" || (lead.hypothesis_kind === "contextual_role" && !roleTitle.trim())} onClick={() => review("accept")} title={invalidCreateName ? "Enter a complete person name" : "Accept"} type="button"><Icon name="reviewed" size={14} /></button>
      <button aria-label="Reject organization or role lead" className="reject" disabled={disabled || lead.review_state === "rejected"} onClick={() => review("reject")} title="Reject" type="button"><Icon name="reject" size={14} /></button>
      <button aria-label="Defer organization or role lead" className="defer" disabled={disabled || lead.review_state === "deferred"} onClick={() => review("defer")} title="Defer" type="button"><Icon name="defer" size={14} /></button>
    </div></td>
  </tr>;
}

function DirectoryReviewLeads({ item, onReviewed, reviewTargets }) {
  const leads = item.review_leads || [];
  const [sort, setSort] = useState({ key: "organization", direction: "asc" });
  const [widths, setWidths] = useState(DIRECTORY_REVIEW_COLUMNS.map((column) => column.width));
  const tableRef = useRef(null);
  const ordered = useMemo(() => [...leads].sort((left, right) => {
    const comparison = String(reviewLeadValue(left, sort.key)).localeCompare(String(reviewLeadValue(right, sort.key)));
    return (sort.direction === "asc" ? comparison : -comparison) || String(left.hypothesis_id).localeCompare(String(right.hypothesis_id));
  }), [leads, sort]);

  function sortBy(key) {
    if (key === "actions" || key === "person") return;
    setSort((current) => current.key === key ? { key, direction: current.direction === "asc" ? "desc" : "asc" } : { key, direction: "asc" });
  }

  function beginResize(event, index) {
    event.preventDefault();
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    const startX = event.clientX;
    const start = [...widths];
    const move = (nextEvent) => {
      const pair = start[index] + start[index + 1];
      const left = Math.max(7, Math.min(pair - 7, start[index] + ((nextEvent.clientX - startX) / tableWidth) * 100));
      const next = [...start];
      next[index] = left;
      next[index + 1] = pair - left;
      setWidths(next);
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  if (!leads.length) return null;
  return <section className="directory-review-leads"><h3>Organization &amp; role leads <span>{leads.length}</span></h3><div className="directory-detail-scroll"><table ref={tableRef}><colgroup>{widths.map((width, index) => <col key={DIRECTORY_REVIEW_COLUMNS[index].key} style={{ width: `${width}%` }} />)}</colgroup><thead><tr>{DIRECTORY_REVIEW_COLUMNS.map((column, index) => <th aria-sort={sort.key === column.key ? (sort.direction === "asc" ? "ascending" : "descending") : "none"} key={column.key}><button className="directory-review-sort" disabled={column.key === "actions" || column.key === "person"} onClick={() => sortBy(column.key)} type="button"><span>{column.label}</span>{column.key !== "actions" && column.key !== "person" && <Icon name={sort.key === column.key ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={12} />}</button>{index < DIRECTORY_REVIEW_COLUMNS.length - 1 && <span aria-label={`Resize ${column.label} column`} aria-orientation="vertical" className="directory-resizer" onDoubleClick={() => setWidths(DIRECTORY_REVIEW_COLUMNS.map((value) => value.width))} onPointerDown={(event) => beginResize(event, index)} role="separator" tabIndex={0} />}</th>)}</tr></thead><tbody>{ordered.map((lead) => <DirectoryReviewLeadRow item={item} key={`${lead.hypothesis_id}:${lead.projection_version}`} lead={lead} onReviewed={onReviewed} reviewTargets={reviewTargets} />)}</tbody></table></div></section>;
}

function DirectoryDetail({ item, onReviewed, reviewTargets }) {
  const members = item.members || [];
  const sources = item.source_records || members.flatMap((member) => member.source_records || []);
  const affiliations = item.organizations || [];
  const affiliationRows = affiliations.flatMap((affiliation) => affiliation.roles?.length
    ? affiliation.roles.map((role) => ({ affiliation, role }))
    : [{ affiliation, role: null }]);
  const roleCount = affiliationRows.filter(({ role }) => role).length;
  return <div className="directory-detail">
    <DirectoryReviewLeads item={item} onReviewed={onReviewed} reviewTargets={reviewTargets} />
    <details className="directory-evidence"><summary><span>Evidence &amp; history</span><small>{counted(item.activities?.length || 0, "activity")} · {counted(sources.length, "source")}{affiliations.length ? ` · ${counted(affiliations.length, "organization")}` : ""}</small></summary><div className="directory-evidence-body">
    <section><h3>Activity timeline <span>{item.activities?.length || 0}</span></h3>
      <div className="directory-detail-scroll"><table><thead><tr><th>Date</th><th>Channel</th><th>Conversation or evidence</th><th>Participation</th><th>Source</th></tr></thead><tbody>{(item.activities || []).map((activity) => <tr key={`${activity.channel}-${activity.observation_id}`}><td>{activity.occurred_at ? formatDate(activity.occurred_at) : "Undated"}</td><td>{label(activity.channel)}</td><td>{activity.title || "Bounded source evidence"}</td><td>{label(activity.participation_status)} · {label(activity.evidence_status)}</td><td><code>{compactHash(activity.source_record_id)}</code></td></tr>)}</tbody></table></div>
    </section>
    <section><h3>Source identities <span>{sources.length}</span></h3>
      <div className="directory-detail-scroll"><table><thead><tr><th>Member</th><th>Provider</th><th>Type</th><th>Label</th><th>Status</th></tr></thead><tbody>{sources.map((source, index) => <tr key={source.source_record_id || index}><td>{source.person_id || source.organization_id || "—"}</td><td>{label(source.provider_kind)}</td><td>{label(source.record_type)}</td><td>{source.label || source.external_ref || "—"}</td><td>{label(source.resolution_status)}</td></tr>)}</tbody></table></div>
    </section>
    {!!affiliations.length && <section><h3>Affiliations <span>{counted(affiliations.length, "organization")} · {counted(roleCount, "role")}</span></h3><div className="directory-detail-scroll"><table><thead><tr><th>Organization</th><th>Role</th><th>State</th><th>Valid dates</th><th>Evidence</th><th>Basis</th></tr></thead><tbody>{affiliationRows.map(({ affiliation, role }) => <tr key={`${affiliation.affiliation_id || affiliation.organization_id}:${role?.role_id || "unassigned"}`}><td>{affiliation.primary_name}</td><td>{role ? label(role.role_type) : "No role asserted"}</td><td>{label(role?.status || affiliation.status)}</td><td>{roleValidity(role || affiliation)}</td><td>{role ? counted(role.evidence_ids?.length, "item") : "—"}</td><td>{label(role?.basis || affiliation.basis)}</td></tr>)}</tbody></table></div></section>}
    {item.entity_kind === "unresolved_group" && <p className="people-editing-boundary">These source records share a display name. They remain separate identities until an explicit reviewed reconciliation decision.</p>}
    </div></details>
  </div>;
}

function PeopleTable({ title, columns, rows }) {
  return (
    <section className="identity-section people-table-section">
      <div className="identity-section-heading"><h3>{title}</h3><span>{rows.length}</span></div>
      {rows.length
        ? <div className="people-table-wrap"><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={`${title}-${rowIndex}`}>{row.map((cell, cellIndex) => <td key={`${title}-${rowIndex}-${cellIndex}`}>{cell ?? "—"}</td>)}</tr>)}</tbody></table></div>
        : <p className="muted">No {title.toLowerCase()} in this projection.</p>}
    </section>
  );
}

const HYPOTHESIS_COLUMNS = [
  { key: "proposal", label: "Proposal" },
  { key: "counterpart", label: "Related" },
  { key: "source", label: "Source" },
  { key: "threads", label: "Threads" },
  { key: "last", label: "Last observed" },
  { key: "status", label: "Status" }
];
const DEFAULT_HYPOTHESIS_WIDTHS = [24, 20, 13, 10, 20, 13];
const MIN_HYPOTHESIS_WIDTHS = [14, 14, 9, 8, 12, 10];

function hypothesisValue(row, key, role) {
  if (key === "proposal") return role ? row.display_value || row.counterpart_label : hypothesisLabel(row.relationship_type);
  if (key === "counterpart") return role ? row.organization || row.department || "—" : row.counterpart_label || "—";
  if (key === "source") return row.evidence_source === "mail_metadata" ? "Mail metadata" : hypothesisLabel(row.hypothesis_kind);
  if (key === "threads") return Number(row.independent_thread_count || row.observation_count || 0);
  if (key === "last") return Date.parse(row.last_observed_at || "") || 0;
  if (key === "status") return row.status || "proposed";
  return "";
}

function HypothesisTable({ title, rows, role = false, onReview, hypothesisDecision }) {
  const [sort, setSort] = useState({ key: "last", direction: "desc" });
  const [expandedId, setExpandedId] = useState("");
  const [widths, setWidths] = useState(DEFAULT_HYPOTHESIS_WIDTHS);
  const tableRef = useRef(null);
  const ordered = useMemo(() => [...rows].sort((left, right) => {
    const leftValue = hypothesisValue(left, sort.key, role);
    const rightValue = hypothesisValue(right, sort.key, role);
    const comparison = typeof leftValue === "number"
      ? leftValue - rightValue
      : String(leftValue).localeCompare(String(rightValue));
    return comparison * (sort.direction === "asc" ? 1 : -1)
      || String(left.hypothesis_id).localeCompare(String(right.hypothesis_id));
  }), [rows, role, sort]);

  function sortBy(key) {
    setSort((current) => current.key === key
      ? { key, direction: current.direction === "asc" ? "desc" : "asc" }
      : { key, direction: key === "last" || key === "threads" ? "desc" : "asc" });
  }

  function resize(index, startWidths, deltaPercent) {
    const pairTotal = startWidths[index] + startWidths[index + 1];
    const left = Math.min(pairTotal - MIN_HYPOTHESIS_WIDTHS[index + 1], Math.max(MIN_HYPOTHESIS_WIDTHS[index], startWidths[index] + deltaPercent));
    const next = [...startWidths];
    next[index] = left;
    next[index + 1] = pairTotal - left;
    setWidths(next);
  }

  function beginResize(event, index) {
    event.preventDefault();
    event.stopPropagation();
    const tableWidth = tableRef.current?.getBoundingClientRect().width || 1;
    const startX = event.clientX;
    const startWidths = [...widths];
    const move = (nextEvent) => resize(index, startWidths, ((nextEvent.clientX - startX) / tableWidth) * 100);
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  function resizeWithKeyboard(event, index) {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    resize(index, widths, event.key === "ArrowLeft" ? -1.5 : 1.5);
  }

  return (
    <section className="identity-section people-table-section hypothesis-table-section">
      <div className="identity-section-heading"><h3>{title}</h3><span>{rows.length}</span></div>
      <div className="people-table-wrap">
        <table ref={tableRef}>
          <colgroup>{widths.map((width, index) => <col key={`${title}-width-${index}`} style={{ width: `${width}%` }} />)}</colgroup>
          <thead><tr>{HYPOTHESIS_COLUMNS.map((column, index) => {
            const active = sort.key === column.key;
            return <th aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"} key={column.key}>
              <button className="hypothesis-sort" aria-label={`Sort ${title} by ${column.label}`} onClick={() => sortBy(column.key)} type="button"><span>{column.label}</span><Icon name={active ? (sort.direction === "asc" ? "sortAscending" : "sortDescending") : "sortNone"} size={13} /></button>
              {index < HYPOTHESIS_COLUMNS.length - 1 && <span aria-label={`Resize ${column.label} column`} aria-orientation="vertical" aria-valuenow={Math.round(widths[index])} className="hypothesis-resizer" onDoubleClick={() => setWidths([...DEFAULT_HYPOTHESIS_WIDTHS])} onKeyDown={(event) => resizeWithKeyboard(event, index)} onPointerDown={(event) => beginResize(event, index)} role="separator" tabIndex={0} />}
            </th>;
          })}</tr></thead>
          <tbody>{ordered.map((row) => {
            const expanded = expandedId === row.hypothesis_id;
            return <Fragment key={row.hypothesis_id}>
              <tr className={expanded ? "hypothesis-row expanded" : "hypothesis-row"}>
                <td><button className="hypothesis-expand" aria-expanded={expanded} aria-label={`${expanded ? "Collapse" : "Expand"} ${role ? row.display_value || row.counterpart_label : hypothesisLabel(row.relationship_type)} evidence`} onClick={() => setExpandedId(expanded ? "" : row.hypothesis_id)} type="button"><Icon name={expanded ? "chevronDown" : "chevronRight"} size={14} /><strong>{role ? row.display_value || row.counterpart_label : hypothesisLabel(row.relationship_type)}</strong></button></td>
                <td>{role ? row.organization || row.department || "—" : row.counterpart_label || "—"}</td>
                <td>{row.evidence_source === "mail_metadata" ? "Mail metadata" : hypothesisLabel(row.hypothesis_kind)}</td>
                <td>{row.independent_thread_count || row.observation_count || 0}</td>
                <td>{row.last_observed_at ? formatDate(row.last_observed_at) : "—"}</td>
                <td><span className={`hypothesis-status ${row.review_state || "unreviewed"}`}><Icon name={row.review_state === "accepted" ? "reviewed" : row.review_state === "rejected" ? "reject" : row.review_state === "deferred" ? "defer" : "preview"} size={13} />{row.review_state === "accepted" ? "Accepted" : row.review_state === "rejected" ? "Rejected" : row.review_state === "deferred" ? "Deferred" : "Needs review"}</span></td>
              </tr>
              {expanded && <tr className="hypothesis-evidence-row"><td colSpan={HYPOTHESIS_COLUMNS.length}><dl>
                <div><dt>Basis</dt><dd>{row.basis || "—"}</dd></div>
                <div><dt>Why unaccepted</dt><dd>{row.why_not_accepted || "Review is required."}</dd></div>
                <div><dt>Direction</dt><dd>{label(row.mail_direction || row.directionality)}</dd></div>
                <div><dt>Time range</dt><dd>{row.first_observed_at ? `${formatDate(row.first_observed_at)} – ${formatDate(row.last_observed_at)}` : "—"}</dd></div>
                <div><dt>Evidence</dt><dd>{counted(row.observation_count, "observation")} · {counted(row.independent_thread_count, "independent thread")}</dd></div>
                <div><dt>Conflicts</dt><dd>{row.conflicts?.length ? row.conflicts.map((conflict) => conflict.title || conflict.reason).join(" · ") : "None recorded"}</dd></div>
              </dl>{row.evidence_source === "mail_metadata" && onReview ? <div className="hypothesis-review-strip" aria-label="Relationship review actions">
                <span>{hypothesisDecision?.id === row.hypothesis_id && hypothesisDecision.message ? hypothesisDecision.message : "Review this mail-derived lead"}</span>
                <div>
                  <button aria-label="Accept relationship lead" className="accept" disabled={hypothesisDecision?.status === "loading" || row.review_state === "accepted"} onClick={() => onReview(row, "accept")} title="Accept" type="button"><Icon name="reviewed" size={15} /></button>
                  <button aria-label="Reject relationship lead" className="reject" disabled={hypothesisDecision?.status === "loading" || row.review_state === "rejected"} onClick={() => onReview(row, "reject")} title="Reject" type="button"><Icon name="reject" size={15} /></button>
                  <button aria-label="Defer relationship lead" className="defer" disabled={hypothesisDecision?.status === "loading" || row.review_state === "deferred"} onClick={() => onReview(row, "defer")} title="Defer" type="button"><Icon name="defer" size={15} /></button>
                </div>
              </div> : null}</td></tr>}
            </Fragment>;
          })}</tbody>
        </table>
      </div>
    </section>
  );
}
