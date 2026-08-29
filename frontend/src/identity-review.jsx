import { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "./icons.jsx";

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
      person_id: "person-1",
      identity_kind: "canonical_person",
      status: "reviewed",
      primary_name: "Alex Example",
      aliases: ["Alex E."],
      source_records: [{ source_record_id: "source-1", provider_kind: "fixture", record_type: "contact", label: "Alex Example", resolution_status: "reviewed" }],
      roles: [{ role_id: "role-1", role_type: "project_lead", organization_id: "org-1", status: "reviewed", evidence_ids: ["evidence-1"] }],
      relationships: [{ relationship_id: "relationship-1", relationship_type: "works_with", subject_id: "person-1", object_id: "person-2", status: "reviewed", evidence_ids: ["evidence-2"] }],
      review_occurrences: [],
      speaker_review_count: 0,
      recording_count: 0,
      possible_related_records: [],
      input_watermark: "fixture-watermark-1",
      built_at: "2026-08-16T18:00:00Z"
    }
  ],
  total: 1,
  counts: { canonical_person: 1, local_contact: 0, reviewed_speaker: 0 }
};

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function label(value) {
  return String(value || "").replaceAll("_", " ");
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

function identitySummary(item) {
  if (item.identity_kind === "reviewed_speaker") {
    return `${item.speaker_review_count || 0} speaker reviews · ${item.recording_count || 0} recordings`;
  }
  if (item.identity_kind === "local_contact") {
    return item.contact_methods?.[0]?.value || "Unlinked local contact";
  }
  return `${item.source_records?.length || 0} sources${item.roles?.length ? ` · ${item.roles.length} roles` : ""}`;
}

function PeopleView() {
  const [payload, setPayload] = useState(FALLBACK_PEOPLE);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState("");
  const [sort, setSort] = useState("name");
  const [selectedId, setSelectedId] = useState("");
  const [loadState, setLoadState] = useState({ status: "loading", message: "Loading contacts…" });

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      const params = new URLSearchParams({ limit: "100" });
      if (query.trim()) params.set("q", query.trim());
      if (kindFilter) params.set("kind", kindFilter);
      try {
        const next = await fetchJson(`/api/people?${params}`);
        if (cancelled) return;
        setPayload(next);
        setSelectedId((current) => next.items?.some((item) => item.person_id === current) ? current : next.items?.[0]?.person_id || "");
        setLoadState({ status: "live", message: "Local directory loaded" });
      } catch (error) {
        if (cancelled) return;
        setPayload(FALLBACK_PEOPLE);
        setSelectedId((current) => current || FALLBACK_PEOPLE.items[0]?.person_id || "");
        setLoadState({ status: "preview", message: `Redacted preview data: ${error.message}` });
      }
    }, query.trim() ? 180 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, kindFilter]);

  const items = useMemo(() => {
    const next = [...(payload.items || [])];
    if (sort === "reviews") {
      next.sort((left, right) => (right.speaker_review_count || 0) - (left.speaker_review_count || 0) || left.primary_name.localeCompare(right.primary_name));
    } else if (sort === "type") {
      next.sort((left, right) => identityKindLabel(left.identity_kind).localeCompare(identityKindLabel(right.identity_kind)) || left.primary_name.localeCompare(right.primary_name));
    } else {
      next.sort((left, right) => left.primary_name.localeCompare(right.primary_name));
    }
    return next;
  }, [payload.items, sort]);
  const selected = items.find((item) => item.person_id === selectedId) || items[0] || null;
  const counts = payload.counts || {};
  return (
    <section className="identity-surface" aria-label="Contacts">
      <div className="identity-toolbar contacts-toolbar">
        <label><span>Search</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name, email, recording, or source" /></label>
        <label><span>Record type</span><select value={kindFilter} onChange={(event) => setKindFilter(event.target.value)}><option value="">All records</option><option value="canonical_person">People ({counts.canonical_person || 0})</option><option value="local_contact">Local contacts ({counts.local_contact || 0})</option><option value="reviewed_speaker">Review names ({counts.reviewed_speaker || 0})</option></select></label>
        <label><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="name">Name A–Z</option><option value="reviews">Most speaker reviews</option><option value="type">Record type</option></select></label>
        <div className={`identity-load-state ${loadState.status}`} role="status"><strong>{payload.total || 0}</strong><span>{loadState.message}</span></div>
      </div>
      <div className="identity-master-detail">
        <label className="identity-mobile-picker"><span>Selected contact</span><select value={selected?.person_id || ""} onChange={(event) => setSelectedId(event.target.value)}>{items.map((item) => <option key={item.person_id} value={item.person_id}>{item.primary_name} · {identityKindLabel(item.identity_kind)}</option>)}</select></label>
        <div className="identity-list" aria-label="Contacts list">
          {items.map((item) => (
            <button className={item.person_id === selected?.person_id ? "active" : ""} key={item.person_id} onClick={() => setSelectedId(item.person_id)} type="button">
              <span className="identity-list-copy"><strong>{item.primary_name || "Unnamed record"}</strong><small>{identitySummary(item)}</small></span>
              <span className={`identity-list-kicker ${item.identity_kind}`}>{identityKindLabel(item.identity_kind)}</span>
            </button>
          ))}
        </div>
        <div className="identity-detail">{selected ? <PeopleDetail person={selected} /> : <p className="muted contacts-empty">No contacts match these filters.</p>}</div>
      </div>
    </section>
  );
}

function PeopleDetail({ person }) {
  const kind = identityKindLabel(person.identity_kind);
  const boundary = person.identity_kind === "reviewed_speaker"
    ? "Operator-reviewed speaker name. It is evidence for reconciliation, not yet a canonical contact link."
    : person.identity_kind === "local_contact"
      ? "Local contact record. It remains separate until an explicit reviewed link or merge."
      : "Canonical conversation person assembled from reviewed local knowledge sources.";
  return (
    <>
      <header className="identity-detail-heading"><div><p className="eyebrow">{kind}</p><h2>{person.primary_name || "Unnamed record"}</h2>{!!person.aliases?.length && <p>{person.aliases.join(" · ")}</p>}</div><span className={`identity-state-pill ${person.status}`}>{label(person.status)}</span></header>
      <p className="people-editing-boundary">{boundary}</p>
      {!!person.possible_related_records?.length && <PeopleTable title="Possible related records" columns={["Type", "Name", "Why not linked"]} rows={person.possible_related_records.map((row) => [identityKindLabel(row.identity_kind), row.primary_name, "Exact display name only — review required"])} />}
      {!!person.contact_methods?.length && <PeopleTable title="Contact methods" columns={["Type", "Value"]} rows={person.contact_methods.map((row) => [label(row.kind), row.value])} />}
      {!!person.review_occurrences?.length && <PeopleTable title="Speaker review appearances" columns={["Recording", "File", "Speaker", "Reviewed"]} rows={person.review_occurrences.map((row) => [row.recording_title || "Untitled recording", row.recording_filename, row.speaker_ref, formatDate(row.reviewed_at)])} />}
      {person.identity_kind !== "reviewed_speaker" && <PeopleTable title="Source records" columns={["Provider", "Type", "Label", "Status"]} rows={(person.source_records || []).map((row) => [row.provider_kind, row.record_type, row.label || row.external_ref, row.resolution_status])} />}
      {!!person.roles?.length && <PeopleTable title="Roles" columns={["Role", "Organization", "Status", "Evidence"]} rows={person.roles.map((row) => [label(row.role_type), row.organization_id || "—", row.status, (row.evidence_ids || []).length])} />}
      {!!person.relationships?.length && <PeopleTable title="Relationships" columns={["Relationship", "Subject", "Object", "Status"]} rows={person.relationships.map((row) => [label(row.relationship_type), row.subject_id, row.object_id, row.status])} />}
      <details className="identity-provenance"><summary>Record provenance</summary><dl className="identity-lineage"><div><dt>Record ID</dt><dd><code>{person.source_identity_id || person.person_id}</code></dd></div><div><dt>Projection watermark</dt><dd><code>{compactHash(person.input_watermark)}</code></dd></div><div><dt>Built</dt><dd>{person.built_at ? formatDate(person.built_at) : "Unavailable"}</dd></div></dl></details>
    </>
  );
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
