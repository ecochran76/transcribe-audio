import { useEffect, useMemo, useState } from "react";

const REVIEW_ACTIONS = [
  "confirm",
  "choose_existing_person",
  "create_reviewed_provisional_person",
  "not_listed",
  "unresolved",
  "reject_event",
  "choose_event",
  "no_matching_event",
  "mixed_speaker",
  "group_labels",
  "split_label",
  "correct_source_record",
  "correct_role",
  "correct_relationship",
  "merge_people",
  "split_person",
  "supersede",
  "defer"
];

const FALLBACK_QUEUE = {
  items: [
    {
      queue_item_id: "fixture-queue-1",
      conversation_id: "fixture-conversation-1",
      recording_id: "fixture-recording-1",
      original_recording_filename: "Monday planning.m4a",
      source_artifact_sha256: "a".repeat(64),
      source_media_sha256: "b".repeat(64),
      processing_run_id: "fixture-run-1",
      model_versions: ["identity-model-v1"],
      rubric_versions: ["identity-rubric-v1"],
      profile_versions: [],
      calendar_candidates: [
        { candidate_id: "event-1", label: "Monday planning", association_strength: 0.83, attendees: ["Alex Example", "Morgan Example"] },
        { candidate_id: "event-2", label: "Project checkpoint", association_strength: 0.51, attendees: ["Alex Example"] }
      ],
      participant_hypotheses: [
        { hypothesis_id: "participant-1", label: "Alex Example", kind: "participant" },
        { hypothesis_id: "mentioned-1", label: "Taylor Example", kind: "mentioned_person" }
      ],
      speakers: [
        {
          speaker_ref: "SPEAKER_01",
          proposal_id: "proposal-1",
          best_guess: { person_id: "person-1", label: "Alex Example", strength: 0.76 },
          alternatives: [{ person_id: "person-2", label: "Morgan Example", strength: 0.42 }],
          evidence: [
            { pillar: "calendar", direction: "supporting", summary: "Candidate-event attendee snapshot" },
            { pillar: "acoustic", direction: "contradicting", summary: "Insufficient reviewed acoustic coverage" }
          ],
          audio: { media_url: "", start_ms: 1250, end_ms: 6900 }
        }
      ],
      review_state: "unreviewed",
      decision_history: [],
      effect_preview_ref: "",
      projection_version: "1",
      created_at: "2026-08-16T18:00:00Z",
      priority: 90,
      impact_score: 0.8
    }
  ],
  total: 1
};

const FALLBACK_PEOPLE = {
  items: [
    {
      person_id: "person-1",
      status: "reviewed",
      primary_name: "Alex Example",
      aliases: ["Alex E."],
      source_records: [{ source_record_id: "source-1", provider_kind: "fixture", record_type: "contact", label: "Alex Example", resolution_status: "reviewed" }],
      roles: [{ role_id: "role-1", role_type: "project_lead", organization_id: "org-1", status: "reviewed", evidence_ids: ["evidence-1"] }],
      relationships: [{ relationship_id: "relationship-1", relationship_type: "works_with", subject_id: "person-1", object_id: "person-2", status: "reviewed", evidence_ids: ["evidence-2"] }],
      input_watermark: "fixture-watermark-1",
      built_at: "2026-08-16T18:00:00Z"
    }
  ],
  total: 1,
  relationship_hop_limit: 2
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
  return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : "Unscored";
}

function operationId(prefix) {
  return `${prefix}-${globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`}`;
}

/*
  Low-fi layout:
  [query + state] [status]
  [priority queue, 34%] | [original filename + candidates + speakers/audio, 66%]
                         [decision form -> exact effect preview -> record]
*/
export function IdentityReviewView({ mode }) {
  const peopleMode = mode === "people";
  const [payload, setPayload] = useState(peopleMode ? FALLBACK_PEOPLE : FALLBACK_QUEUE);
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState(peopleMode ? "" : "unreviewed");
  const [selectedId, setSelectedId] = useState("");
  const [loadState, setLoadState] = useState({ status: "loading", message: "Loading local projection…" });
  const [action, setAction] = useState("confirm");
  const [personId, setPersonId] = useState("");
  const [comment, setComment] = useState("");
  const [preview, setPreview] = useState(null);
  const [pendingSubmission, setPendingSubmission] = useState(null);
  const [decisionState, setDecisionState] = useState({ status: "idle", message: "" });

  useEffect(() => {
    setQuery("");
    setStateFilter(peopleMode ? "" : "unreviewed");
    setSelectedId("");
  }, [peopleMode]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      const params = new URLSearchParams({ limit: "100" });
      if (query.trim()) params.set("q", query.trim());
      if (stateFilter) params.set(peopleMode ? "status" : "state", stateFilter);
      try {
        const next = await fetchJson(`${peopleMode ? "/api/people" : "/api/identity-review"}?${params}`);
        if (cancelled) return;
        setPayload(next);
        setSelectedId((current) => next.items?.some((item) => (item.person_id || item.queue_item_id) === current)
          ? current
          : next.items?.[0]?.person_id || next.items?.[0]?.queue_item_id || "");
        setLoadState({ status: "live", message: "Local projection loaded" });
      } catch (error) {
        if (cancelled) return;
        const fallback = peopleMode ? FALLBACK_PEOPLE : FALLBACK_QUEUE;
        setPayload(fallback);
        setSelectedId((current) => current || fallback.items[0]?.person_id || fallback.items[0]?.queue_item_id || "");
        setLoadState({ status: "preview", message: `Redacted preview data: ${error.message}` });
      }
    }, query.trim() ? 180 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [peopleMode, query, stateFilter]);

  const selected = useMemo(
    () => payload.items?.find((item) => (item.person_id || item.queue_item_id) === selectedId) || payload.items?.[0] || null,
    [payload.items, selectedId]
  );
  const firstSpeaker = !peopleMode ? selected?.speakers?.[0] : null;

  useEffect(() => {
    setPreview(null);
    setPendingSubmission(null);
    setDecisionState({ status: "idle", message: "" });
    setPersonId(firstSpeaker?.best_guess?.person_id || "");
  }, [selectedId, firstSpeaker?.proposal_id, peopleMode]);

  async function previewDecision() {
    if (!selected || !firstSpeaker) return;
    const submissionId = operationId("submission");
    const submission = {
      schema_version: "transcribe-audio.identity-review-submission.v1",
      submission_id: submissionId,
      queue_item_id: selected.queue_item_id,
      conversation_id: selected.conversation_id,
      proposal_id: firstSpeaker.proposal_id,
      action,
      expected_projection_version: selected.projection_version,
      decision_payload: {
        speaker_ref: firstSpeaker.speaker_ref,
        ...(action === "choose_existing_person" ? { person_id: personId } : {})
      },
      comment,
      idempotency_key: operationId("identity-review"),
      reviewer: "operator",
      decided_at: new Date().toISOString()
    };
    setDecisionState({ status: "loading", message: "Computing exact local effect preview…" });
    try {
      const next = await fetchJson(`/api/identity-review/items/${encodeURIComponent(selected.queue_item_id)}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submission)
      });
      setPendingSubmission(submission);
      setPreview(next);
      setDecisionState({ status: "previewed", message: "Preview ready. No identity, profile, provider, or deletion effect has run." });
    } catch (error) {
      setDecisionState({ status: "error", message: `Preview failed: ${error.message}` });
    }
  }

  async function recordDecision() {
    if (!pendingSubmission || !selected) return;
    setDecisionState({ status: "loading", message: "Recording the local review decision…" });
    try {
      const receipt = await fetchJson(`/api/identity-review/items/${encodeURIComponent(selected.queue_item_id)}/decisions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pendingSubmission)
      });
      const refreshed = await fetchJson(`/api/identity-review?limit=100`);
      setPayload(refreshed);
      setPreview(receipt.effect_preview);
      setPendingSubmission(null);
      setDecisionState({ status: "recorded", message: `Decision recorded at projection v${receipt.projection_version}; accepted identity effects remain zero.` });
    } catch (error) {
      setDecisionState({ status: "error", message: `Decision rejected: ${error.message}` });
    }
  }

  return (
    <section className="identity-surface" aria-label={peopleMode ? "People" : "Identity Review"}>
      <div className="identity-toolbar">
        <label>
          <span>Search</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={peopleMode ? "Name or alias" : "Filename, person, speaker, or event"} />
        </label>
        <label>
          <span>{peopleMode ? "Person status" : "Review state"}</span>
          <select value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}>
            <option value="">All</option>
            {peopleMode ? (
              <><option value="reviewed">Reviewed</option><option value="provisional">Provisional</option><option value="merged">Merged</option></>
            ) : (
              <><option value="unreviewed">Unreviewed</option><option value="unresolved">Unresolved</option><option value="reviewed">Reviewed</option></>
            )}
          </select>
        </label>
        <div className={`identity-load-state ${loadState.status}`} role="status">
          <strong>{payload.total || 0}</strong>
          <span>{loadState.message}</span>
        </div>
      </div>

      <div className="identity-master-detail">
        {payload.items?.length ? (
          <label className="identity-mobile-picker">
            <span>{peopleMode ? "Person" : "Conversation"}</span>
            <select
              aria-label={peopleMode ? "Select person" : "Select identity review conversation"}
              value={selected?.person_id || selected?.queue_item_id || ""}
              onChange={(event) => setSelectedId(event.target.value)}
            >
              {payload.items.map((item) => {
                const id = item.person_id || item.queue_item_id;
                const itemLabel = peopleMode
                  ? item.primary_name || "Unnamed person"
                  : `${item.original_recording_filename} · ${item.speakers?.length || 0} speakers`;
                return <option key={id} value={id}>{itemLabel}</option>;
              })}
            </select>
          </label>
        ) : <p className="identity-mobile-empty">No records match these filters.</p>}
        <div className="identity-list" aria-label={peopleMode ? "People list" : "Identity review queue"}>
          {(payload.items || []).map((item) => {
            const id = item.person_id || item.queue_item_id;
            return (
              <button className={id === (selected?.person_id || selected?.queue_item_id) ? "active" : ""} key={id} onClick={() => setSelectedId(id)} type="button">
                <span className="identity-list-kicker">{peopleMode ? label(item.status) : `${label(item.review_state)} · priority ${item.priority ?? 0}`}</span>
                <strong>{peopleMode ? item.primary_name || "Unnamed person" : item.original_recording_filename}</strong>
                <small>{peopleMode ? `${item.source_records?.length || 0} sources · ${item.roles?.length || 0} roles` : `${item.speakers?.length || 0} speakers · ${item.calendar_candidates?.length || 0} calendar candidates`}</small>
              </button>
            );
          })}
          {!payload.items?.length && <p className="muted">No records match these filters.</p>}
        </div>

        <div className="identity-detail">
          {!selected ? (payload.items?.length ? <p className="muted">Select a record to inspect it.</p> : null) : peopleMode ? (
            <PeopleDetail person={selected} />
          ) : (
            <>
              <header className="identity-detail-heading">
                <div>
                  <p className="eyebrow">Original recording</p>
                  <h2>{selected.original_recording_filename}</h2>
                </div>
                <span className={`identity-state-pill ${selected.review_state}`}>{label(selected.review_state)} · v{selected.projection_version}</span>
              </header>
              <dl className="identity-lineage">
                <div><dt>Conversation</dt><dd>{selected.conversation_id}</dd></div>
                <div><dt>Recording</dt><dd>{selected.recording_id}</dd></div>
                <div><dt>Artifact SHA-256</dt><dd><code>{compactHash(selected.source_artifact_sha256)}</code></dd></div>
                <div><dt>Media SHA-256</dt><dd><code>{compactHash(selected.source_media_sha256)}</code></dd></div>
                <div><dt>Processing run</dt><dd>{selected.processing_run_id}</dd></div>
                <div><dt>Versions</dt><dd>{[...(selected.model_versions || []), ...(selected.rubric_versions || []), ...(selected.profile_versions || [])].join(" · ") || "None"}</dd></div>
              </dl>

              <section className="identity-section">
                <div className="identity-section-heading"><h3>Calendar alternatives</h3><span>Top three plus no match</span></div>
                <div className="identity-candidate-list">
                  {(selected.calendar_candidates || []).slice(0, 3).map((candidate) => (
                    <article key={candidate.candidate_id}>
                      <div><strong>{candidate.label || candidate.summary || candidate.candidate_id}</strong><span>{strength(candidate.association_strength)}</span></div>
                      <p>{(candidate.attendees || []).join(" · ") || "No attendee snapshot"}</p>
                    </article>
                  ))}
                  <article><div><strong>No matching event</strong><span>Explicit option</span></div><p>Use when none of the candidates fits the recording.</p></article>
                </div>
              </section>

              <section className="identity-section">
                <div className="identity-section-heading"><h3>Participant hypotheses</h3><span>Evidence, not identity truth</span></div>
                <div className="identity-tags">
                  {(selected.participant_hypotheses || []).map((participant) => <span key={participant.hypothesis_id}>{participant.label} · {label(participant.kind)}</span>)}
                </div>
              </section>

              <section className="identity-section">
                <div className="identity-section-heading"><h3>Speakers and evidence</h3><span>Every label needs a disposition</span></div>
                {(selected.speakers || []).map((speaker) => <SpeakerReview key={speaker.speaker_ref} speaker={speaker} />)}
              </section>

              {firstSpeaker && (
                <section className="identity-decision-panel">
                  <div className="identity-section-heading"><h3>Decision and effect preview</h3><span>{firstSpeaker.speaker_ref}</span></div>
                  <div className="identity-decision-form">
                    <label><span>Decision</span><select value={action} onChange={(event) => { setAction(event.target.value); setPreview(null); setPendingSubmission(null); }}>{REVIEW_ACTIONS.map((value) => <option key={value} value={value}>{label(value)}</option>)}</select></label>
                    {action === "choose_existing_person" && <label><span>Person ID</span><input value={personId} onChange={(event) => setPersonId(event.target.value)} required /></label>}
                    <label className="identity-comment"><span>Immutable review comment</span><textarea value={comment} onChange={(event) => setComment(event.target.value)} rows="3" /></label>
                  </div>
                  <div className="identity-decision-actions">
                    <button className="secondary-action" disabled={decisionState.status === "loading" || (action === "choose_existing_person" && !personId.trim())} onClick={previewDecision} type="button">Preview exact effect</button>
                    <button className="primary-action" disabled={!preview || !pendingSubmission || decisionState.status === "loading"} onClick={recordDecision} type="button">Record decision</button>
                  </div>
                  {decisionState.message && <p className={`identity-decision-message ${decisionState.status}`} role="status">{decisionState.message}</p>}
                  {preview && <EffectPreview preview={preview} />}
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function SpeakerReview({ speaker }) {
  const audio = speaker.audio || {};
  const source = audio.media_url ? `${audio.media_url}#t=${Math.max(0, Number(audio.start_ms || 0) / 1000)},${Math.max(0, Number(audio.end_ms || 0) / 1000)}` : "";
  return (
    <article className="speaker-evidence-row">
      <header><div><strong>{speaker.speaker_ref}</strong><span>{speaker.best_guess?.label || "No named proposal"}</span></div><b>{strength(speaker.best_guess?.strength)}</b></header>
      {source ? <audio aria-label={`${speaker.speaker_ref} source-bound sample`} controls preload="none" src={source} /> : <p className="muted">Source-bound audio is unavailable in redacted preview data.</p>}
      <div className="identity-alternatives"><span>Alternatives</span>{(speaker.alternatives || []).map((candidate) => <small key={candidate.person_id || candidate.label}>{candidate.label} · {strength(candidate.strength)}</small>)}</div>
      <ul>{(speaker.evidence || []).map((evidence, index) => <li key={`${evidence.pillar}-${index}`}><span className={evidence.direction}>{label(evidence.direction)}</span><strong>{label(evidence.pillar)}</strong>{evidence.summary}</li>)}</ul>
    </article>
  );
}

function EffectPreview({ preview }) {
  return (
    <div className="effect-preview" aria-label="Exact effect preview">
      <header><strong>Preview only</strong><span>{preview.provider_write_count} provider writes · {preview.raw_deletion_count} raw deletions</span></header>
      {(preview.proposed_effects || []).map((effect, index) => <div key={`${effect.effect_type}-${index}`}><strong>{label(effect.effect_type)}</strong><code>{JSON.stringify(effect, null, 2)}</code></div>)}
      {(preview.warnings || []).map((warning) => <p key={warning}>{warning}</p>)}
    </div>
  );
}

function PeopleDetail({ person }) {
  return (
    <>
      <header className="identity-detail-heading"><div><p className="eyebrow">Person</p><h2>{person.primary_name || "Unnamed person"}</h2><p>{(person.aliases || []).join(" · ") || "No aliases"}</p></div><span className={`identity-state-pill ${person.status}`}>{label(person.status)}</span></header>
      <p className="people-editing-boundary">Tables and explicit forms are the authoritative editing surface. A5 shows the rebuildable local projection; accepted People effects remain gated.</p>
      <PeopleTable title="Source records" columns={["Provider", "Type", "Label", "Status"]} rows={(person.source_records || []).map((row) => [row.provider_kind, row.record_type, row.label || row.external_ref, row.resolution_status])} />
      <PeopleTable title="Roles" columns={["Role", "Organization", "Status", "Evidence"]} rows={(person.roles || []).map((row) => [label(row.role_type), row.organization_id || "—", row.status, (row.evidence_ids || []).length])} />
      <PeopleTable title="Relationships" columns={["Relationship", "Subject", "Object", "Status"]} rows={(person.relationships || []).map((row) => [label(row.relationship_type), row.subject_id, row.object_id, row.status])} />
      <dl className="identity-lineage"><div><dt>Projection watermark</dt><dd><code>{compactHash(person.input_watermark)}</code></dd></div><div><dt>Built</dt><dd>{person.built_at || "Unavailable"}</dd></div><div><dt>Relationship display</dt><dd>Maximum two hops</dd></div></dl>
    </>
  );
}

function PeopleTable({ title, columns, rows }) {
  return (
    <section className="identity-section people-table-section"><div className="identity-section-heading"><h3>{title}</h3><span>{rows.length}</span></div>{rows.length ? <div className="people-table-wrap"><table><thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={`${title}-${rowIndex}`}>{row.map((cell, cellIndex) => <td key={`${title}-${rowIndex}-${cellIndex}`}>{cell ?? "—"}</td>)}</tr>)}</tbody></table></div> : <p className="muted">No {title.toLowerCase()} in this projection.</p>}</section>
  );
}
