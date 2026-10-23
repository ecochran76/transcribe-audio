import React, { useEffect, useRef, useState } from "react";

async function json(url, body, signal) {
  const response = await fetch(url, { method: body === undefined ? "GET" : "POST", body: body === undefined ? undefined : JSON.stringify(body), headers: { "Content-Type": "application/json" }, signal });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "speech_service_unavailable");
  return value;
}
const active = new Set(["submitting", "queued", "processing", "unknown", "cancelling"]);
function message(error) {
  const codes = { speech_unconfigured: "Shared sample tools are not connected.", speech_source_changed: "This recording or speaker timing changed. Refresh before starting again.", speech_origin_forbidden: "This browser address is not enabled for shared sample tools.", speech_diarization_unavailable: "Speaker timing is not available for this recording yet." };
  return codes[error.message] || "The shared speech service could not confirm that request. Check the same operation again.";
}
export function SpeechHarvestPanel({ documentId }) {
  const base = `/api/conversations/${encodeURIComponent(documentId)}/speech-harvest`;
  const [context, setContext] = useState(null), [job, setJob] = useState(null), [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false), [audio, setAudio] = useState({}), [retained, setRetained] = useState({}), [proposals, setProposals] = useState({});
  const [query, setQuery] = useState(""), [people, setPeople] = useState([]), [target, setTarget] = useState(""), [evidence, setEvidence] = useState("");
  const urls = useRef(new Set()), live = useRef(true);
  useEffect(() => {
    live.current = true;
    const controller = new AbortController();
    json(base, undefined, controller.signal).then((value) => {
      if (!live.current) return;
      setContext(value); setJob(value.jobs?.[0] || null);
      const kept = value.retentions || [];
      setRetained(Object.fromEntries(kept.map((item) => [item.candidateId, item])));
      setProposals(Object.fromEntries((value.proposals || []).flatMap((item) => { const sample = kept.find((entry) => entry.requestId === item.retentionRequestId); return sample ? [[sample.candidateId, item]] : []; })));
    }).catch((error) => { if (live.current && error.name !== "AbortError") setNotice(message(error)); });
    return () => { live.current = false; controller.abort(); for (const url of urls.current) URL.revokeObjectURL(url); urls.current.clear(); };
  }, [base]);
  useEffect(() => {
    if (!job || !["submitting", "queued", "processing"].includes(job.state)) return;
    let cancelled = false;
    const timer = setTimeout(async () => {
      try { const value = await json(`${base}/${encodeURIComponent(job.requestId)}`); if (!cancelled && live.current) setJob(value); }
      catch { if (!cancelled && live.current) setNotice("Progress could not be checked. The operation is saved; check it again."); }
    }, 750);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [base, job]);
  async function run(action) {
    if (busy) return;
    setBusy(true); setNotice("");
    try { await action(); } catch (error) { if (live.current) setNotice(message(error)); }
    finally { if (live.current) setBusy(false); }
  }
  async function start() {
    const requestId = crypto.randomUUID(), prepared = context.prepared;
    setJob({ requestId, state: "submitting" });
    try {
      const value = await json(base, { requestId, sourceSha256: prepared.source.sha256, sourceRevision: prepared.source.revision, diarizationSha256: prepared.diarizationSha256 });
      if (live.current) setJob(value);
    } catch (error) {
      try { const value = await json(`${base}/${requestId}`); if (live.current) setJob(value); }
      catch { if (live.current) setJob({ requestId, state: "unknown" }); }
      throw error;
    }
  }
  async function play(candidate) {
    const value = await json(`${base}/${job.requestId}/candidates/${candidate.id}`);
    const bytes = Uint8Array.from(atob(value.dataBase64), (character) => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: value.mediaType }));
    if (!live.current) { URL.revokeObjectURL(url); return; }
    if (audio[candidate.id]) { URL.revokeObjectURL(audio[candidate.id]); urls.current.delete(audio[candidate.id]); }
    urls.current.add(url); setAudio((previous) => ({ ...previous, [candidate.id]: url }));
  }
  async function keep(candidate) {
    const requestId = crypto.randomUUID();
    setRetained((previous) => ({ ...previous, [candidate.id]: { requestId, state: "pending" } }));
    let value;
    try { value = await json(`${base}/${job.requestId}/retain`, { requestId, candidateId: candidate.id, reviewed: true, name: `${candidate.speaker} meeting sample` }); }
    catch { value = { requestId, candidateId: candidate.id, state: "unknown" }; }
    if (live.current) setRetained((previous) => ({ ...previous, [candidate.id]: value }));
  }
  async function checkKeep(candidate) {
    const value = await json(`${base}/${retained[candidate.id].requestId}`);
    if (live.current) setRetained((items) => ({ ...items, [candidate.id]: value }));
  }
  async function propose(candidate) {
    const requestId = crypto.randomUUID();
    setProposals((previous) => ({ ...previous, [candidate.id]: { requestId, state: "pending" } }));
    let value;
    try { value = await json(`${base}/${retained[candidate.id].requestId}/propose`, { requestId, target: JSON.parse(target), evidence }); }
    catch { value = { requestId, state: "unknown" }; }
    if (live.current) {
      setProposals((previous) => ({ ...previous, [candidate.id]: value }));
      setNotice(value.state === "proposed" ? "Person link proposed in Speech. Meeting speaker choices are unchanged." : "The proposal outcome needs checking; its exact operation is saved.");
    }
  }
  const candidates = job?.result?.candidates || [];
  return <section className="workflow-action-panel" aria-label="Shared voice samples">
    <div className="workflow-view-heading"><div><span>Shared voice samples</span><h3>Keep useful speaker clips</h3></div><div className="workflow-action-row">
      <button type="button" disabled={!context || busy || active.has(job?.state)} onClick={() => run(start)}>Find speaker samples</button>
      {job ? <button type="button" disabled={busy} onClick={() => run(async () => { const value = await json(`${base}/${job.requestId}`); if (live.current) setJob(value); })}>Check progress</button> : null}
      {job && active.has(job.state) ? <button type="button" disabled={busy} onClick={() => run(async () => { const value = await json(`${base}/${job.requestId}/cancel`, {}); if (live.current) setJob(value); })}>Stop sample task</button> : null}
    </div></div>
    <p className="muted">Speech prepares and keeps samples. Speaker labels stay local to this recording; meeting assignments are reviewed here separately.</p>
    {job ? <p role="status">{job.state === "completed" ? `${candidates.length} sample candidates ready.` : `Sample task: ${job.state}`}</p> : null}
    {notice ? <p role="status">{notice}</p> : null}
    {context?.foreignObservations?.length ? <details><summary>Speech observations awaiting meeting review ({context.foreignObservations.length})</summary>
      <p>These attributed observations do not change meeting participants. Review the recording and make meeting speaker choices separately.</p>
      {context.foreignObservations.map((item) => <p key={item.requestId}>{item.state === "conflict" ? "Conflicting observation" : "Proposed observation"} · sample {item.sampleId} · {item.evidence}</p>)}
    </details> : null}
    <div className="identity-list">{candidates.map((candidate) => <article key={candidate.id}>
      <strong>{candidate.speaker} · {(candidate.startMs / 1000).toFixed(2)}–{(candidate.endMs / 1000).toFixed(2)} seconds</strong>
      <small>{candidate.quality.boundary === "word-aligned" ? "Boundaries follow supplied words." : "Word boundaries need review."} {candidate.quality.reasons?.join(", ")}</small>
      <div className="workflow-action-row"><button type="button" disabled={busy} onClick={() => run(() => play(candidate))}>Listen to candidate</button>
        {!retained[candidate.id] ? <button type="button" disabled={busy} onClick={() => run(() => keep(candidate))}>Keep reviewed sample</button> : <span>{retained[candidate.id].state === "applied" ? "Kept in Speech" : "Keep outcome unconfirmed"}</span>}
        {retained[candidate.id] && retained[candidate.id].state !== "applied" ? <button type="button" disabled={busy} onClick={() => run(() => checkKeep(candidate))}>Check keep outcome</button> : null}
      </div>
      {audio[candidate.id] ? <audio controls src={audio[candidate.id]} aria-label={`${candidate.speaker} sample candidate`} /> : null}
      {retained[candidate.id]?.state === "applied" ? <details><summary>Propose a person link</summary>
        <label>Find a Contacts person<input value={query} onChange={(event) => setQuery(event.target.value)} maxLength={120} /></label>
        <button type="button" disabled={busy || !query.trim()} onClick={() => run(async () => { const value = await json(`${base}/people/search?q=${encodeURIComponent(query)}`); if (live.current) setPeople(value.people || []); })}>Find person</button>
        <label>Person candidate<select value={target} onChange={(event) => setTarget(event.target.value)}><option value="">Choose a person</option>{people.map((person) => <option key={person.reference.id} value={JSON.stringify(person.reference)}>{person.name}</option>)}</select></label>
        <label>Why this person?<input value={evidence} onChange={(event) => setEvidence(event.target.value)} maxLength={2000} /></label>
        <button type="button" disabled={busy || !target || !evidence.trim() || Boolean(proposals[candidate.id])} onClick={() => run(() => propose(candidate))}>Send link proposal</button>
        {proposals[candidate.id] ? <span>{proposals[candidate.id].state === "proposed" ? "Proposal sent" : "Proposal outcome unconfirmed"}</span> : null}
        {proposals[candidate.id] && proposals[candidate.id].state !== "proposed" ? <button type="button" disabled={busy} onClick={() => run(async () => { const value = await json(`${base}/${proposals[candidate.id].requestId}`); if (live.current) setProposals((previous) => ({ ...previous, [candidate.id]: value })); })}>Check proposal outcome</button> : null}
      </details> : null}
      <details><summary>Source details</summary><p>Speech operation: {job.speechRequestId}</p><p>Source revision: {job.source?.revision}</p>{retained[candidate.id]?.sampleId ? <p>Speech sample: {retained[candidate.id].sampleId}</p> : null}</details>
    </article>)}</div>
  </section>;
}
