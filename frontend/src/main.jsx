import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const NAV_ITEMS = [
  "Library",
  "Review Queue",
  "Context Runs",
  "Contacts",
  "Provenance",
  "Intelligence",
  "Depositions",
  "Settings"
];

const FALLBACK_LIBRARY = [
  {
    id: "demo-transcript",
    kind: "transcript",
    title: "Weekly Product Sync",
    generated_at: "2026-05-16T14:30:00-05:00",
    metadata: { event: { summary: "Weekly Product Sync" } },
    media_blob: { playback_url: "", download_url: "" },
    source_path: "redacted-local-artifact"
  },
  {
    id: "demo-contextual",
    kind: "contextual_readout",
    title: "SoyLei / Tempo contextual readout",
    generated_at: "2026-05-15T11:20:00-05:00",
    metadata: { route: { label: "SoyLei Tempo technical collaboration" } },
    media_blob: {},
    source_path: "redacted-local-artifact"
  }
];

const FALLBACK_REVIEW_QUEUE = {
  total_open: 68,
  buckets: [
  { label: "Filename conflicts", count: 0, status: "clear", detail: "Reviewed: 8 keep target, 2 preserve both" },
  { id: "first_pass_summaries", label: "First-pass summaries", count: 68, status: "pending", detail: "Stored transcripts waiting for first-pass summaries" },
  { label: "Memory harvest", count: 0, status: "gated", detail: "Requires explicit review file approval" },
  { label: "Speaker IDs", count: 0, status: "planned", detail: "Contact dedupe tables are planned in P09" }
  ],
  items: []
};

function formatDate(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

function statusLabel(status) {
  return status.replaceAll("_", " ");
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function App() {
  const [activeNav, setActiveNav] = useState("Library");
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [query, setQuery] = useState("");
  const [library, setLibrary] = useState({ items: FALLBACK_LIBRARY, total: FALLBACK_LIBRARY.length });
  const [reviewQueue, setReviewQueue] = useState(FALLBACK_REVIEW_QUEUE);
  const [selectedId, setSelectedId] = useState(FALLBACK_LIBRARY[0].id);
  const [health, setHealth] = useState({ status: "offline", store_dir: "fallback demo data" });
  const [apiError, setApiError] = useState("");
  const [reviewAction, setReviewAction] = useState({ status: "idle", message: "", manifest: "", batchId: "" });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [healthPayload, libraryPayload, reviewPayload] = await Promise.all([
          fetchJson("/api/health"),
          fetchJson("/api/library?limit=25"),
          fetchJson("/api/review-queue?limit=100")
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setLibrary(libraryPayload);
        setReviewQueue(reviewPayload);
        setSelectedId(libraryPayload.items?.[0]?.id || "");
        setApiError("");
      } catch (error) {
        if (cancelled) return;
        setApiError(`Using redacted fixture data because the local API is unavailable: ${error.message}`);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const visibleItems = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return library.items || [];
    return (library.items || []).filter((item) => {
      const haystack = `${item.title || ""} ${item.kind || ""} ${item.source_path || ""}`.toLowerCase();
      return haystack.includes(needle);
    });
  }, [library.items, query]);

  const selected = visibleItems.find((item) => item.id === selectedId) || visibleItems[0] || null;
  const reviewBuckets = reviewQueue.buckets || FALLBACK_REVIEW_QUEUE.buckets;

  async function prepareFirstPassBatch() {
    setReviewAction({ status: "running", message: "Preparing a 5-item dry-run batch...", manifest: "", batchId: "" });
    try {
      const payload = await postJson("/api/review-queue/first-pass-summaries/prepare", { limit: 5, store: true });
      setReviewAction({
        status: "prepared",
        message: `Prepared ${payload.request_count} dry-run requests; no provider work was submitted.`,
        manifest: payload.manifest || "",
        batchId: payload.batch_id || ""
      });
      setApiError("");
    } catch (error) {
      setReviewAction({ status: "error", message: `Prepare failed: ${error.message}`, manifest: "", batchId: "" });
    }
  }

  async function submitFirstPassBatch() {
    if (!reviewAction.manifest) return;
    const approved = window.confirm("Submit this prepared first-pass summary batch to the configured provider?");
    if (!approved) return;
    setReviewAction((current) => ({ ...current, status: "submitting", message: "Submitting prepared batch..." }));
    try {
      const payload = await postJson("/api/review-queue/first-pass-summaries/submit", {
        manifest: reviewAction.manifest,
        approval_token: "SUBMIT_FIRST_PASS_SUMMARY_BATCH"
      });
      setReviewAction({
        status: payload.status || "submitted",
        message: `Submitted ${payload.request_count} requests; batch ${payload.batch_id || "pending id"}.`,
        manifest: payload.manifest || reviewAction.manifest,
        batchId: payload.batch_id || ""
      });
    } catch (error) {
      setReviewAction((current) => ({ ...current, status: "error", message: `Submit failed: ${error.message}` }));
    }
  }

  async function refreshFirstPassBatch() {
    if (!reviewAction.manifest) return;
    setReviewAction((current) => ({ ...current, status: "checking", message: "Checking prepared batch status..." }));
    try {
      const payload = await postJson("/api/review-queue/first-pass-summaries/status", {
        manifest: reviewAction.manifest,
        materialize: true
      });
      const counts = payload.batch_counts || {};
      const countText = Object.entries(counts).map(([key, value]) => `${key}: ${value}`).join(", ");
      setReviewAction({
        status: payload.status || "checked",
        message: countText
          ? `Batch status ${payload.status}; ${countText}. Materialized ${payload.materialized?.length || 0}.`
          : `Batch status ${payload.status}.`,
        manifest: payload.manifest || reviewAction.manifest,
        batchId: payload.batch_id || reviewAction.batchId || ""
      });
    } catch (error) {
      setReviewAction((current) => ({ ...current, status: "error", message: `Status check failed: ${error.message}` }));
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">tr</span>
          <div>
            <strong>Transcript Console</strong>
            <small>{health.status === "ok" ? "live local API" : "redacted preview mode"}</small>
          </div>
        </div>
        <nav className="nav-tabs" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <button
              className={activeNav === item ? "active" : ""}
              key={item}
              onClick={() => setActiveNav(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </nav>
        <label className="global-search">
          <span>Search</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="meeting, person, matter..." />
        </label>
      </header>

      <section
        className={[
          "workspace",
          leftCollapsed ? "left-collapsed" : "",
          rightCollapsed ? "right-collapsed" : ""
        ].join(" ")}
      >
        <aside className="left-pane">
          <button className="pane-toggle" onClick={() => setLeftCollapsed((value) => !value)} type="button">
            {leftCollapsed ? "Filters +" : "Collapse filters"}
          </button>
          <div className="pane-content">
            <p className="eyebrow">{activeNav}</p>
            <h2>Workflow filters</h2>
            <div className="filter-card">
              <span>Kind</span>
              <button type="button">Transcripts</button>
              <button type="button">Summaries</button>
              <button type="button">Contextual readouts</button>
            </div>
            <div className="filter-card">
              <span>Review buckets</span>
              {reviewBuckets.map((bucket) => (
                <button key={bucket.id || bucket.label} type="button">
                  {bucket.label}
                  <strong>{bucket.count}</strong>
                </button>
              ))}
            </div>
            <div className="runtime-card">
              <span>Runtime</span>
              <code>{health.store_dir}</code>
              {apiError && <p>{apiError}</p>}
            </div>
          </div>
        </aside>

        <section className="center-pane">
          <div className="view-heading">
            <div>
              <p className="eyebrow">Operator Surface</p>
              <h1>{activeNav === "Review Queue" ? "Review queue" : "Transcript library"}</h1>
            </div>
            <div className="summary-strip">
              <span>{library.total ?? visibleItems.length} stored rows</span>
              <span>{reviewQueue.total_open ?? reviewBuckets.reduce((total, item) => total + item.count, 0)} open reviews</span>
            </div>
          </div>

          {activeNav === "Review Queue" ? (
            <ReviewQueue
              queue={reviewQueue}
              reviewAction={reviewAction}
              onPrepareFirstPass={prepareFirstPassBatch}
              onSubmitFirstPass={submitFirstPassBatch}
              onRefreshFirstPass={refreshFirstPassBatch}
            />
          ) : (
            <LibraryTable items={visibleItems} selectedId={selected?.id} onSelect={setSelectedId} />
          )}
        </section>

        <aside className="right-pane">
          <button className="pane-toggle" onClick={() => setRightCollapsed((value) => !value)} type="button">
            {rightCollapsed ? "Inspector +" : "Collapse inspector"}
          </button>
          <Inspector item={selected} activeNav={activeNav} reviewQueue={reviewQueue} />
        </aside>
      </section>
    </main>
  );
}

function LibraryTable({ items, selectedId, onSelect }) {
  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th>Title</th>
            <th>Kind</th>
            <th>Calendar / route</th>
            <th>Generated</th>
            <th>Media</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const calendar = item.metadata?.event?.summary || item.metadata?.route?.label || "No context yet";
            return (
              <tr className={selectedId === item.id ? "selected" : ""} key={item.id} onClick={() => onSelect(item.id)}>
                <td>
                  <strong>{item.title || "Untitled artifact"}</strong>
                  <small>{item.id}</small>
                </td>
                <td><span className="chip">{statusLabel(item.kind || "unknown")}</span></td>
                <td>{calendar}</td>
                <td>{formatDate(item.generated_at || item.updated_at)}</td>
                <td>{item.media_blob?.playback_url ? "Blob linked" : "No blob"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ReviewQueue({ queue, reviewAction, onPrepareFirstPass, onSubmitFirstPass, onRefreshFirstPass }) {
  const buckets = queue.buckets || [];
  const items = queue.items || [];
  return (
    <>
      <div className="review-grid">
        {buckets.map((bucket) => (
          <article className={`review-card ${bucket.status}`} key={bucket.id || bucket.label}>
            <span>{bucket.status}</span>
            <strong>{bucket.count}</strong>
            <h3>{bucket.label}</h3>
            <p>{bucket.detail}</p>
            {bucket.id === "first_pass_summaries" && (
              <button
                className="inline-action"
                disabled={!bucket.count || reviewAction.status === "running"}
                onClick={onPrepareFirstPass}
                type="button"
              >
                {reviewAction.status === "running" ? "Preparing..." : "Prepare batch"}
              </button>
            )}
          </article>
        ))}
      </div>
      {reviewAction.message && (
        <div className={`action-notice ${reviewAction.status}`}>
          <strong>{reviewAction.message}</strong>
          {reviewAction.manifest && <code>{reviewAction.manifest}</code>}
          <div className="notice-actions">
            {reviewAction.manifest && !reviewAction.batchId && (
              <button disabled={reviewAction.status === "submitting"} onClick={onSubmitFirstPass} type="button">
                Submit prepared batch
              </button>
            )}
            {reviewAction.manifest && (
              <button disabled={reviewAction.status === "checking"} onClick={onRefreshFirstPass} type="button">
                Check and materialize
              </button>
            )}
          </div>
        </div>
      )}
      <div className="queue-list">
        <div className="queue-list-heading">
          <h2>Route review items</h2>
          <span>{items.length} loaded</span>
        </div>
        {items.length ? (
          items.map((item) => (
            <article className={`queue-row ${item.status}`} key={item.id}>
              <div>
                <strong>{item.label}</strong>
                <small>{item.reason}</small>
              </div>
              <span>{item.route_decision_exists ? "route available" : "stale route reference"}</span>
              <code>{item.route_decision_path || item.review_path}</code>
            </article>
          ))
        ) : (
          <p className="muted">No route review files are currently loaded.</p>
        )}
      </div>
    </>
  );
}

function Inspector({ item, activeNav, reviewQueue }) {
  if (activeNav === "Review Queue") {
    const routeBucket = (reviewQueue.buckets || []).find((bucket) => bucket.id === "route_reviews");
    const filenameBucket = (reviewQueue.buckets || []).find((bucket) => bucket.id === "filename_conflicts");
    const summaryBucket = (reviewQueue.buckets || []).find((bucket) => bucket.id === "first_pass_summaries");
    return (
      <div className="inspector-content">
        <p className="eyebrow">Review Inspector</p>
        <h2>Live queue state</h2>
        <p>Apply actions stay gated behind CLI review files, approval tokens, and dry-run previews.</p>
        <dl>
          <dt>State root</dt>
          <dd>{reviewQueue.state_dir || "Unavailable"}</dd>
          <dt>Route reviews</dt>
          <dd>{routeBucket?.detail || "No route review summary."}</dd>
          <dt>Filename conflicts</dt>
          <dd>{filenameBucket?.detail || "No filename conflict summary."}</dd>
          <dt>First-pass summaries</dt>
          <dd>{summaryBucket ? `${summaryBucket.count} pending; ${summaryBucket.duplicate_count || 0} duplicate queue entries skipped.` : "No summary queue data."}</dd>
        </dl>
      </div>
    );
  }
  if (!item) {
    return <div className="inspector-content"><h2>No selection</h2></div>;
  }
  return (
    <div className="inspector-content">
      <p className="eyebrow">Inspector</p>
      <h2>{item.title || "Untitled artifact"}</h2>
      <dl>
        <dt>Kind</dt>
        <dd>{statusLabel(item.kind || "unknown")}</dd>
        <dt>Source</dt>
        <dd>{item.source_path || "Unknown"}</dd>
        <dt>Blob</dt>
        <dd>{item.media_blob?.id || "No media blob linked"}</dd>
      </dl>
      {item.media_blob?.playback_url ? (
        <div className="player-card">
          <audio controls src={item.media_blob.playback_url} />
          <a href={item.media_blob.download_url}>Download source recording</a>
        </div>
      ) : (
        <p className="muted">Playback appears here once a stored blob is linked.</p>
      )}
      <div className="action-stack">
        <button type="button">Open context packet</button>
        <button type="button">Prepare share link</button>
        <button type="button">Review speakers</button>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
