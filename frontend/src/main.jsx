import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const NAV_ITEMS = [
  { id: "Library", label: "Library", enabled: true },
  { id: "Review Queue", label: "Review Queue", enabled: true },
  { id: "Context Runs", label: "Context Runs", enabled: false },
  { id: "Contacts", label: "Contacts", enabled: false },
  { id: "Provenance", label: "Provenance", enabled: false },
  { id: "Intelligence", label: "Intelligence", enabled: true },
  { id: "Depositions", label: "Depositions", enabled: false },
  { id: "Settings", label: "Settings", enabled: false }
];

const LIBRARY_KIND_FILTERS = [
  { id: "all", label: "All artifacts" },
  { id: "transcript", label: "Transcripts" },
  { id: "readout", label: "Summaries" },
  { id: "contextual_readout", label: "Contextual readouts" }
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
  { id: "app_intelligence_human_review", label: "App Intelligence review", count: 0, status: "clear", detail: "No App Intelligence human-review decisions pending" },
  { id: "first_pass_summaries", label: "First-pass summaries", count: 68, status: "pending", detail: "Stored transcripts waiting for first-pass summaries" },
  { label: "Memory harvest", count: 0, status: "gated", detail: "Requires explicit review file approval" },
  { label: "Speaker IDs", count: 0, status: "planned", detail: "Contact dedupe tables are planned in P09" }
  ],
  items: []
};

const FALLBACK_CONVERSATIONS = {
  items: buildConversationRows(FALLBACK_LIBRARY),
  total: FALLBACK_LIBRARY.length
};

const CONVERSATION_PAGE_SIZE = 100;

const FALLBACK_INTELLIGENCE = {
  config: {
    schema_version: "transcribe-audio.intelligence-config.v1",
    config_path: "~/.local/state/transcribe-audio/intelligence.config.json",
    tasks: {
      first_pass_summary: {
        task: "first_pass_summary",
        provider: "openai-compatible",
        model: "gpt-4o-mini",
        timeout: 120,
        temperature: 0.1,
        fallbacks: ["codex-exec"],
        human_review: "on_warning",
        requires_ledger: false,
        source: "fallback"
      },
      app_supervisor: {
        task: "app_supervisor",
        provider: "codex-app-server",
        model: "",
        timeout: 120,
        temperature: 0,
        fallbacks: ["codex-exec"],
        human_review: "phase_policy",
        requires_ledger: true,
        source: "fallback"
      }
    }
  },
  providers: {
    providers: [
      { id: "openai-compatible", label: "OpenAI-compatible API", status: "configured-by-env", capabilities: ["summarize", "extract"] },
      { id: "codex-app-server", label: "Codex app-server", status: "ready", ready: true, capabilities: { persistent_sessions: true, branching: true } }
    ],
    default_supervisor: "codex-app-server"
  },
  runs: { items: [], total: 0 },
  smokes: { latest_report: null, reports: [], runs: [], report_count: 0, run_count: 0 },
  smokeJobs: { items: [], total: 0, available_job_types: [] }
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

function filterCount(items, kind) {
  if (kind === "all") return items.length;
  return items.filter((item) => item.kind === kind).length;
}

function capabilityLabels(capabilities) {
  if (Array.isArray(capabilities)) return capabilities;
  if (capabilities && typeof capabilities === "object") {
    return Object.entries(capabilities)
      .filter(([, enabled]) => Boolean(enabled))
      .map(([name]) => statusLabel(name));
  }
  return [];
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function hasActiveSmokeJob(smokeJobs) {
  return (smokeJobs?.items || []).some((job) => ["queued", "running"].includes(job.status));
}

function cleanupSummaryLabel(summary) {
  if (!summary) return "";
  const mode = summary.apply ? "applied" : "dry-run";
  return `${mode}: delete ${summary.delete_run_count || 0}/${summary.matched_run_count || 0} runs · delete ${summary.delete_evidence_count || 0}/${summary.matched_evidence_count || 0} evidence · keep ${summary.keep_evidence || 0} newest/${summary.evidence_days || 0}d`;
}

function smokeJobTiming(job) {
  const createdAt = job?.created_at || "";
  const startedAt = job?.started_at || "";
  const finishedAt = job?.finished_at || "";
  const start = new Date(startedAt || createdAt);
  const end = new Date(finishedAt || "");
  const parts = [];
  if (createdAt) parts.push(`queued ${formatDate(createdAt)}`);
  if (finishedAt) parts.push(`finished ${formatDate(finishedAt)}`);
  if (!Number.isNaN(start.getTime()) && !Number.isNaN(end.getTime())) {
    parts.push(`${Math.max(0, Math.round((end.getTime() - start.getTime()) / 1000))}s runtime`);
  }
  return parts.join(" · ");
}

function groupSmokeJobsByType(jobs) {
  const groups = [];
  for (const job of jobs || []) {
    const key = job?.job_type || "smoke_job";
    let group = groups.find((item) => item.key === key);
    if (!group) {
      group = { key, label: statusLabel(key), jobs: [] };
      groups.push(group);
    }
    group.jobs.push(job);
  }
  return groups;
}

function filterSmokeJobs(jobs, filter) {
  if (filter === "failed") return jobs.filter((job) => job.status === "failed");
  if (filter === "write_bearing") return jobs.filter((job) => job.will_execute_write_bearing_action);
  if (filter === "evidence") return jobs.filter((job) => Boolean(job.evidence_summary));
  return jobs;
}

function sourceArtifactPath(item) {
  return item?.metadata?.source_artifact_path || item?.json_payload?.source_artifact_path || "";
}

function findSourceDocument(item, items) {
  const sourcePath = sourceArtifactPath(item);
  if (!sourcePath) return null;
  return (items || []).find((candidate) => candidate.source_path === sourcePath) || null;
}

function relatedSourceDocument(relatedDocuments) {
  return relatedDocuments?.source_document || null;
}

function mediaForItem(item, sourceDocument) {
  return item?.media_blob?.playback_url ? item.media_blob : sourceDocument?.media_blob?.playback_url ? sourceDocument.media_blob : null;
}

function documentSummaryText(detail, { allowTranscriptFallback = false } = {}) {
  const payload = detail?.json_payload || {};
  if (payload.summary || payload.readout) return payload.summary || payload.readout;
  if (allowTranscriptFallback && detail?.kind !== "transcript") return detail?.text_content || "";
  return "";
}

function conversationGroupKey(item) {
  return sourceArtifactPath(item) || item?.source_path || item?.id || "";
}

function buildConversationRows(items) {
  const groups = new Map();
  (items || []).forEach((item) => {
    const key = conversationGroupKey(item);
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        transcript: null,
        readouts: [],
        contextualReadouts: [],
        artifacts: []
      });
    }
    const group = groups.get(key);
    group.artifacts.push(item);
    if (item.kind === "transcript") group.transcript = item;
    if (item.kind === "readout") group.readouts.push(item);
    if (item.kind === "contextual_readout") group.contextualReadouts.push(item);
  });
  return Array.from(groups.values()).map((group) => {
    const representative = group.contextualReadouts[0] || group.readouts[0] || group.transcript || group.artifacts[0];
    const source = group.transcript || group.artifacts[0];
    const latestArtifact = group.artifacts
      .slice()
      .sort((a, b) => String(b.generated_at || b.updated_at || "").localeCompare(String(a.generated_at || a.updated_at || "")))[0];
    return {
      ...group,
      representative,
      source,
      latestArtifact,
      hasTranscript: Boolean(group.transcript),
      hasSummary: group.readouts.length > 0,
      hasContextualReadout: group.contextualReadouts.length > 0,
      title: representative?.title || source?.title || "Untitled conversation"
    };
  });
}

function normalizeConversationRow(row) {
  if (!row || !row.representative) return row;
  return {
    key: row.key,
    representative: row.representative,
    source: row.source || row.representative,
    latestArtifact: row.latest_artifact || row.latestArtifact || row.representative,
    artifacts: row.artifacts || [],
    hasTranscript: Boolean(row.workflow?.transcript ?? row.hasTranscript),
    hasSummary: Boolean(row.workflow?.summary ?? row.hasSummary),
    hasContextualReadout: Boolean(row.workflow?.contextual_readout ?? row.hasContextualReadout),
    title: row.title || row.representative?.title || "Untitled conversation",
    calendarLabel: row.calendar_label || row.calendarLabel || "No context yet",
    mediaBlob: row.media_blob?.playback_url ? row.media_blob : row.mediaBlob?.playback_url ? row.mediaBlob : {},
    mediaReady: Boolean(row.media_ready ?? row.mediaReady),
    updatedAt: row.updated_at || row.updatedAt || row.latest_artifact?.generated_at || row.representative?.generated_at || ""
  };
}

function conversationSearchParams(kindFilter, query, offset = 0) {
  const params = new URLSearchParams({
    limit: String(CONVERSATION_PAGE_SIZE),
    offset: String(offset)
  });
  if (kindFilter !== "all") params.set("kind", kindFilter);
  if (query.trim()) params.set("query", query.trim());
  return params;
}

function mergeConversationItems(currentItems, nextItems) {
  const merged = [];
  const seen = new Set();
  [...(currentItems || []), ...(nextItems || [])].forEach((item) => {
    const key = item?.key || item?.representative?.id || JSON.stringify(item);
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(item);
  });
  return merged;
}

function speakerClassName(speaker) {
  const text = String(speaker || "speaker").trim();
  let seed = 0;
  for (let index = 0; index < text.length; index += 1) seed += text.charCodeAt(index);
  return `speaker-${seed % 6}`;
}

function transcriptTurns(detail) {
  const payload = detail?.json_payload || {};
  const text = String(payload.transcript_text || detail?.text_content || "");
  const turns = [];
  let current = null;
  text.split(/\r?\n/).forEach((line) => {
    const match = line.match(/^(.{1,48}?)\s+\[([^\]]+)\]:\s*(.*)$/);
    if (match) {
      if (current) turns.push(current);
      current = {
        speaker: match[1].trim(),
        time: match[2].trim(),
        text: match[3].trim()
      };
      return;
    }
    if (current && line.trim()) {
      current.text = `${current.text} ${line.trim()}`.trim();
    }
  });
  if (current) turns.push(current);
  if (turns.length) return turns;
  return text
    .split(/\n{2,}/)
    .map((chunk, index) => ({ speaker: "Transcript", time: index === 0 ? "header" : "", text: chunk.trim() }))
    .filter((turn) => turn.text);
}

function transcriptMeta(detail) {
  const payload = detail?.json_payload || {};
  return {
    duration: payload.duration_seconds ? `${Math.round(Number(payload.duration_seconds) / 60)} min` : "",
    event: payload.event?.summary || "",
    recordingStart: payload.recording_start || payload.event?.start || "",
    recordingEnd: payload.recording_end || payload.event?.end || "",
    backend: payload.backend || "",
    utteranceCount: payload.utterance_count || transcriptTurns(detail).length
  };
}

function displayLabel(value, fallback = "Item") {
  if (typeof value === "string") return value;
  if (value && typeof value === "object") {
    return value.name || value.email || value.label || value.title || value.text || value.task || fallback;
  }
  return fallback;
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
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).error || "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function App() {
  const [activeNav, setActiveNav] = useState("Library");
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [leftPaneWidth, setLeftPaneWidth] = useState(300);
  const [rightPaneWidth, setRightPaneWidth] = useState(380);
  const [kindFilter, setKindFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [library, setLibrary] = useState({ items: FALLBACK_LIBRARY, total: FALLBACK_LIBRARY.length });
  const [conversations, setConversations] = useState(FALLBACK_CONVERSATIONS);
  const [conversationSearchStatus, setConversationSearchStatus] = useState({ status: "idle", message: "Conversation search has not loaded yet." });
  const [reviewQueue, setReviewQueue] = useState(FALLBACK_REVIEW_QUEUE);
  const [selectedId, setSelectedId] = useState(FALLBACK_LIBRARY[0].id);
  const [health, setHealth] = useState({ status: "offline", store_dir: "fallback demo data" });
  const [apiError, setApiError] = useState("");
  const [reviewAction, setReviewAction] = useState({ status: "idle", message: "", manifest: "", batchId: "", payload: null });
  const [firstPassBatchManifests, setFirstPassBatchManifests] = useState({ items: [], total: 0, limit: 0 });
  const [humanReviewAction, setHumanReviewAction] = useState({ status: "idle", message: "", payload: null });
  const [intelligence, setIntelligence] = useState(FALLBACK_INTELLIGENCE);
  const [selectedTask, setSelectedTask] = useState("first_pass_summary");
  const [taskDraft, setTaskDraft] = useState({ provider: "", model: "", timeout: "", temperature: "", fallbacks: "", human_review: "", requires_ledger: false });
  const [configAction, setConfigAction] = useState({ status: "idle", message: "", preview: null });
  const [runAction, setRunAction] = useState({ status: "idle", message: "", runId: "" });
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedRunDetail, setSelectedRunDetail] = useState(null);
  const [runReplayManifest, setRunReplayManifest] = useState(null);
  const [runDetailAction, setRunDetailAction] = useState({ status: "idle", message: "" });
  const [sessionPreflight, setSessionPreflight] = useState({ status: "idle", message: "", payload: null });
  const [sessionStartAction, setSessionStartAction] = useState({ status: "idle", message: "", payload: null });
  const [modelTurnAction, setModelTurnAction] = useState({ status: "idle", message: "", payload: null });
  const [selectedPacketId, setSelectedPacketId] = useState("");
  const [packetReview, setPacketReview] = useState({ status: "idle", message: "", payload: null });
  const [sendPreflight, setSendPreflight] = useState({ status: "idle", message: "", payload: null });
  const [sendAction, setSendAction] = useState({ status: "idle", message: "", payload: null });
  const [turnStatusAction, setTurnStatusAction] = useState({ status: "idle", message: "", payload: null });
  const [decisionValidation, setDecisionValidation] = useState({ status: "idle", message: "", payload: null });
  const [decisionApply, setDecisionApply] = useState({ status: "idle", message: "", payload: null });
  const [forkPreflightAction, setForkPreflightAction] = useState({ status: "idle", message: "", payload: null });
  const [rollbackPreflightAction, setRollbackPreflightAction] = useState({ status: "idle", message: "", payload: null });
  const [runArtifactAction, setRunArtifactAction] = useState({ status: "idle", message: "", payload: null });
  const [smokeJobAction, setSmokeJobAction] = useState({ status: "idle", message: "", payload: null });
  const [smokeTailAction, setSmokeTailAction] = useState({ status: "idle", message: "", payload: null });
  const [selectedDocumentDetail, setSelectedDocumentDetail] = useState(null);
  const [selectedDocumentDetailAction, setSelectedDocumentDetailAction] = useState({ status: "idle", message: "" });
  const [selectedRelatedDocuments, setSelectedRelatedDocuments] = useState(null);
  const [selectedConversationDetail, setSelectedConversationDetail] = useState(null);
  const [selectedConversationDetailAction, setSelectedConversationDetailAction] = useState({ status: "idle", message: "" });
  const [conversationOpen, setConversationOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [healthPayload, libraryPayload, reviewPayload, batchManifestPayload, providerPayload, configPayload, runsPayload, smokesPayload, smokeJobsPayload] = await Promise.all([
          fetchJson("/api/health"),
          fetchJson("/api/library?limit=200"),
          fetchJson("/api/review-queue?limit=100"),
          fetchJson("/api/review-queue/first-pass-summaries/manifests?limit=5"),
          fetchJson("/api/intelligence/providers"),
          fetchJson("/api/intelligence/config"),
          fetchJson("/api/intelligence/runs?limit=8"),
          fetchJson("/api/intelligence/smokes?limit=5"),
          fetchJson("/api/intelligence/smoke-jobs?limit=20")
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setLibrary(libraryPayload);
        setReviewQueue(reviewPayload);
        setFirstPassBatchManifests(batchManifestPayload);
        setIntelligence({ providers: providerPayload, config: configPayload, runs: runsPayload, smokes: smokesPayload, smokeJobs: smokeJobsPayload });
        setSelectedId(libraryPayload.items?.[0]?.id || "");
        setSelectedRunId(runsPayload.items?.[0]?.run_id || "");
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

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      if (activeNav !== "Library") return;
      const params = conversationSearchParams(kindFilter, query, 0);
      setConversationSearchStatus({
        status: "loading",
        message: query.trim()
          ? `Searching conversations for "${query.trim()}"...`
          : "Loading conversations..."
      });
      try {
        const payload = await fetchJson(`/api/conversations?${params.toString()}`);
        if (cancelled) return;
        setConversations(payload);
        setSelectedId((currentId) => {
          const rows = payload.items || [];
          if (rows.some((row) => (row.artifacts || []).some((artifact) => artifact.id === currentId))) return currentId;
          return rows[0]?.representative?.id || currentId;
        });
        setConversationSearchStatus({
          status: "loaded",
          message: `Loaded ${(payload.items || []).length} of ${payload.total ?? (payload.items || []).length} matching conversations.`
        });
        setApiError("");
      } catch (error) {
        if (cancelled) return;
        setConversationSearchStatus({
          status: "error",
          message: `Conversation search failed: ${error.message}`
        });
        setApiError(`Conversation search failed; using current local rows: ${error.message}`);
      }
    }, query.trim() ? 250 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeNav, kindFilter, query]);

  useEffect(() => {
    let cancelled = false;
    async function loadRunDetail() {
      if (!selectedRunId) {
        setSelectedRunDetail(null);
        setRunReplayManifest(null);
        setRunDetailAction({ status: "idle", message: "" });
        return;
      }
      setRunDetailAction({ status: "loading", message: "Loading selected run ledger..." });
      try {
        const [payload, replayManifest] = await Promise.all([
          fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`),
          fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/replay-manifest`)
        ]);
        if (cancelled) return;
        setSelectedRunDetail(payload);
        setRunReplayManifest(replayManifest);
        setRunDetailAction({ status: "loaded", message: "" });
        setSessionPreflight({ status: "idle", message: "", payload: null });
        setSessionStartAction({ status: "idle", message: "", payload: null });
        setModelTurnAction({ status: "idle", message: "", payload: null });
        setSelectedPacketId(payload.run?.prompt_packets?.slice(-1)[0]?.packet_id || "");
        setPacketReview({ status: "idle", message: "", payload: null });
        setSendPreflight({ status: "idle", message: "", payload: null });
        setSendAction({ status: "idle", message: "", payload: null });
        setTurnStatusAction({ status: "idle", message: "", payload: null });
        setDecisionValidation({ status: "idle", message: "", payload: null });
        setDecisionApply({ status: "idle", message: "", payload: null });
        setForkPreflightAction({ status: "idle", message: "", payload: null });
        setRollbackPreflightAction({ status: "idle", message: "", payload: null });
        setRunArtifactAction({ status: "idle", message: "", payload: null });
      } catch (error) {
        if (cancelled) return;
        setSelectedRunDetail(null);
        setRunReplayManifest(null);
        setRunDetailAction({ status: "error", message: `Run detail failed: ${error.message}` });
      }
    }
    loadRunDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    let cancelled = false;
    async function loadPacketReview() {
      if (!selectedRunId || !selectedPacketId) {
        setPacketReview({ status: "idle", message: "", payload: null });
        return;
      }
      setPacketReview({ status: "loading", message: "Loading prompt packet...", payload: null });
      try {
        const payload = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/prompt-packets/${encodeURIComponent(selectedPacketId)}`);
        if (cancelled) return;
        setPacketReview({ status: "loaded", message: "", payload });
      } catch (error) {
        if (cancelled) return;
        setPacketReview({ status: "error", message: `Prompt packet load failed: ${error.message}`, payload: null });
      }
    }
    loadPacketReview();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, selectedPacketId]);

  async function refreshSelectedRun(runId = selectedRunId) {
    if (!runId) return null;
    const [detail, replayManifest] = await Promise.all([
      fetchJson(`/api/intelligence/runs/${encodeURIComponent(runId)}?event_limit=12`),
      fetchJson(`/api/intelligence/runs/${encodeURIComponent(runId)}/replay-manifest`)
    ]);
    setSelectedRunDetail(detail);
    setRunReplayManifest(replayManifest);
    return detail;
  }

  async function refreshSmokeEvidence() {
    const [smokesPayload, smokeJobsPayload, runsPayload] = await Promise.all([
      fetchJson("/api/intelligence/smokes?limit=5"),
      fetchJson("/api/intelligence/smoke-jobs?limit=20"),
      fetchJson("/api/intelligence/runs?limit=8")
    ]);
    setIntelligence((current) => ({ ...current, smokes: smokesPayload, smokeJobs: smokeJobsPayload, runs: runsPayload }));
    return { smokes: smokesPayload, smokeJobs: smokeJobsPayload };
  }

  async function startSmokeJob(jobType, { applyCleanup = false } = {}) {
    const labels = {
      api_replay_smoke: "API replay smoke",
      browser_replay_smoke: "browser replay smoke",
      first_pass_resume_ui_smoke: "first-pass resume UI smoke",
      cleanup_smokes: applyCleanup ? "smoke cleanup apply" : "smoke cleanup dry-run"
    };
    if (jobType === "browser_replay_smoke" || jobType === "first_pass_resume_ui_smoke") {
      const approved = window.confirm(`Run the ${labels[jobType]} through agent-browser? This is a fixed command and records local smoke artifacts.`);
      if (!approved) return;
    }
    if (jobType === "cleanup_smokes" && applyCleanup) {
      const confirmation = window.prompt(
        "Apply smoke artifact cleanup? This deletes only allowlisted disposable smoke artifacts. Type CLEANUP_APP_SMOKE_ARTIFACTS to continue:"
      );
      if (confirmation !== "CLEANUP_APP_SMOKE_ARTIFACTS") {
        setSmokeJobAction({ status: "idle", message: "Smoke cleanup apply cancelled; typed confirmation did not match.", payload: null });
        return;
      }
    }
    setSmokeJobAction({ status: "queueing", message: `Queueing ${labels[jobType] || jobType}...`, payload: null });
    try {
      const payload = await postJson("/api/intelligence/smoke-jobs", {
        job_type: jobType,
        approval_token: jobType === "cleanup_smokes" && applyCleanup ? "CLEANUP_APP_SMOKE_ARTIFACTS" : "RUN_APP_SMOKE_JOB",
        cleanup: true,
        apply_cleanup: applyCleanup
      });
      await refreshSmokeEvidence();
      setSmokeJobAction({
        status: "queued",
        message: `Queued ${payload.job?.job_id || labels[jobType]}; polling will continue until it finishes.`,
        payload
      });
    } catch (error) {
      setSmokeJobAction({ status: "error", message: `Smoke job failed to queue: ${error.message}`, payload: null });
    }
  }

  async function loadSmokeJobTail(jobId, stream = "stderr") {
    if (!jobId) return;
    setSmokeTailAction({ status: "loading", message: `Loading ${stream} tail for ${jobId}...`, payload: null });
    try {
      const payload = await fetchJson(
        `/api/intelligence/smoke-jobs/${encodeURIComponent(jobId)}/tail?stream=${encodeURIComponent(stream)}&chars=6000`
      );
      setSmokeTailAction({
        status: "loaded",
        message: `Loaded ${stream} tail for ${jobId}; no arbitrary file read was allowed.`,
        payload
      });
    } catch (error) {
      setSmokeTailAction({ status: "error", message: `Smoke job tail failed: ${error.message}`, payload: null });
    }
  }

  function resizePane(pane, startEvent) {
    startEvent.preventDefault();
    const startX = startEvent.clientX;
    const initialWidth = pane === "left" ? leftPaneWidth : rightPaneWidth;
    const onPointerMove = (event) => {
      const delta = event.clientX - startX;
      if (pane === "left") {
        setLeftPaneWidth(clamp(initialWidth + delta, 240, 520));
      } else {
        setRightPaneWidth(clamp(initialWidth - delta, 300, 560));
      }
    };
    const onPointerUp = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp, { once: true });
  }

  function resizePaneWithKeyboard(pane, event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    if (pane === "left") {
      setLeftPaneWidth((value) => clamp(value + direction * 16, 240, 520));
    } else {
      setRightPaneWidth((value) => clamp(value - direction * 16, 300, 560));
    }
  }

  const visibleItems = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (library.items || []).filter((item) => {
      if (kindFilter !== "all" && item.kind !== kindFilter) return false;
      if (!needle) return true;
      const haystack = `${item.title || ""} ${item.kind || ""} ${item.source_path || ""}`.toLowerCase();
      return haystack.includes(needle);
    });
  }, [kindFilter, library.items, query]);
  const visibleConversationRows = useMemo(() => {
    const usingApiRows = conversations.schema_version === "transcribe-audio.conversations.v1";
    if (usingApiRows) return (conversations.items || []).map(normalizeConversationRow);
    const needle = query.trim().toLowerCase();
    return buildConversationRows(visibleItems).filter((row) => {
      if (kindFilter !== "all" && !(row.artifacts || []).some((artifact) => artifact.kind === kindFilter)) return false;
      if (!needle) return true;
      const artifactText = (row.artifacts || [])
        .map((artifact) => `${artifact.title || ""} ${artifact.source_path || ""}`)
        .join(" ");
      const haystack = `${row.title || ""} ${row.calendarLabel || ""} ${artifactText}`.toLowerCase();
      return haystack.includes(needle);
    });
  }, [conversations.items, conversations.schema_version, kindFilter, query, visibleItems]);
  const usingApiConversations = conversations.schema_version === "transcribe-audio.conversations.v1";
  const loadedConversationCount = usingApiConversations ? (conversations.items || []).length : visibleConversationRows.length;
  const totalConversationCount = usingApiConversations ? conversations.total ?? loadedConversationCount : visibleConversationRows.length;
  const conversationSearchLoading = conversationSearchStatus.status === "loading" || conversationSearchStatus.status === "loading_more";
  const canLoadMoreConversations =
    activeNav === "Library" &&
    usingApiConversations &&
    !conversationSearchLoading &&
    loadedConversationCount < totalConversationCount;
  const selectedConversation =
    visibleConversationRows.find((row) => (row.artifacts || []).some((artifact) => artifact.id === selectedId)) ||
    visibleConversationRows[0] ||
    null;
  const selected = selectedConversation?.representative || visibleItems.find((item) => item.id === selectedId) || visibleItems[0] || null;
  const reviewBuckets = reviewQueue.buckets || FALLBACK_REVIEW_QUEUE.buckets;
  const taskEntries = Object.entries(intelligence.config?.tasks || {});
  const selectedTaskConfig = intelligence.config?.tasks?.[selectedTask] || taskEntries[0]?.[1] || null;
  const selectedProvider = (intelligence.providers?.providers || []).find((provider) => provider.id === selectedTaskConfig?.provider);
  const selectedTaskFingerprint = selectedTaskConfig ? JSON.stringify(selectedTaskConfig) : "";
  const smokeJobsActive = hasActiveSmokeJob(intelligence.smokeJobs);

  async function loadMoreConversations() {
    if (!canLoadMoreConversations) return;
    const offset = loadedConversationCount;
    setConversationSearchStatus({
      status: "loading_more",
      message: `Loading conversations ${offset + 1}-${Math.min(offset + CONVERSATION_PAGE_SIZE, totalConversationCount)}...`
    });
    try {
      const payload = await fetchJson(`/api/conversations?${conversationSearchParams(kindFilter, query, offset).toString()}`);
      setConversations((current) => {
        const items = mergeConversationItems(current.items, payload.items);
        return {
          ...payload,
          items,
          offset: 0,
          limit: items.length,
          total: payload.total ?? current.total ?? items.length
        };
      });
      setConversationSearchStatus({
        status: "loaded",
        message: `Loaded ${Math.min(offset + (payload.items || []).length, payload.total ?? totalConversationCount)} of ${payload.total ?? totalConversationCount} matching conversations.`
      });
      setApiError("");
    } catch (error) {
      setConversationSearchStatus({
        status: "error",
        message: `Loading more conversations failed: ${error.message}`
      });
      setApiError(`Loading more conversations failed; keeping current rows: ${error.message}`);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function loadSelectedDocumentDetail() {
      if (!selected?.id || activeNav !== "Library") {
        setSelectedDocumentDetail(null);
        setSelectedRelatedDocuments(null);
        setSelectedDocumentDetailAction({ status: "idle", message: "" });
        return;
      }
      setSelectedDocumentDetailAction({ status: "loading", message: "Loading document details..." });
      try {
        const [payload, relatedPayload] = await Promise.all([
          fetchJson(`/api/documents/${encodeURIComponent(selected.id)}`),
          fetchJson(`/api/documents/${encodeURIComponent(selected.id)}/related`)
        ]);
        if (cancelled) return;
        setSelectedDocumentDetail(payload);
        setSelectedRelatedDocuments(relatedPayload);
        setSelectedDocumentDetailAction({ status: "loaded", message: "" });
      } catch (error) {
        if (cancelled) return;
        setSelectedDocumentDetail(null);
        setSelectedRelatedDocuments(null);
        setSelectedDocumentDetailAction({ status: "error", message: `Document detail failed: ${error.message}` });
      }
    }
    loadSelectedDocumentDetail();
    return () => {
      cancelled = true;
    };
  }, [activeNav, selected?.id]);

  useEffect(() => {
    if (activeNav !== "Library") setConversationOpen(false);
  }, [activeNav]);

  useEffect(() => {
    let cancelled = false;
    async function loadConversationDetail() {
      if (!conversationOpen || !selected?.id || activeNav !== "Library") {
        setSelectedConversationDetail(null);
        setSelectedConversationDetailAction({ status: "idle", message: "" });
        return;
      }
      setSelectedConversationDetailAction({ status: "loading", message: "Loading conversation workspace..." });
      try {
        const payload = await fetchJson(`/api/conversations/${encodeURIComponent(selected.id)}`);
        if (cancelled) return;
        setSelectedConversationDetail(payload);
        setSelectedConversationDetailAction({ status: "loaded", message: "" });
      } catch (error) {
        if (cancelled) return;
        setSelectedConversationDetail(null);
        setSelectedConversationDetailAction({ status: "error", message: `Conversation detail failed: ${error.message}` });
      }
    }
    loadConversationDetail();
    return () => {
      cancelled = true;
    };
  }, [activeNav, conversationOpen, selected?.id]);

  useEffect(() => {
    if (!conversationOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") setConversationOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [conversationOpen]);

  useEffect(() => {
    if (!selectedTaskConfig) return;
    setTaskDraft({
      provider: selectedTaskConfig.provider || "",
      model: selectedTaskConfig.model || "",
      timeout: selectedTaskConfig.timeout ?? "",
      temperature: selectedTaskConfig.temperature ?? "",
      fallbacks: (selectedTaskConfig.fallbacks || []).join(", "),
      human_review: selectedTaskConfig.human_review || "",
      requires_ledger: Boolean(selectedTaskConfig.requires_ledger)
    });
    setConfigAction({ status: "idle", message: "", preview: null });
  }, [selectedTask, selectedTaskFingerprint]);

  useEffect(() => {
    if (!smokeJobsActive) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        await refreshSmokeEvidence();
        if (!cancelled) {
          setSmokeJobAction((current) => (
            current.status === "queued" || current.status === "polling"
              ? { ...current, status: "polling", message: "Smoke job running; polling every 2 seconds..." }
              : current
          ));
        }
      } catch (error) {
        if (!cancelled) {
          setSmokeJobAction({ status: "error", message: `Smoke job polling failed: ${error.message}`, payload: null });
        }
      }
    };
    const interval = window.setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [smokeJobsActive]);

  async function prepareFirstPassBatch() {
    setReviewAction({ status: "running", message: "Preparing a 5-item dry-run batch...", manifest: "", batchId: "", payload: null });
    try {
      const payload = await postJson("/api/review-queue/first-pass-summaries/prepare", { limit: 5, store: true });
      const manifests = await fetchJson("/api/review-queue/first-pass-summaries/manifests?limit=5");
      setFirstPassBatchManifests(manifests);
      setReviewAction({
        status: "prepared",
        message: `Prepared ${payload.request_count} dry-run requests; no provider work was submitted.`,
        manifest: payload.manifest || "",
        batchId: payload.batch_id || "",
        payload
      });
      setApiError("");
    } catch (error) {
      setReviewAction({ status: "error", message: `Prepare failed: ${error.message}`, manifest: "", batchId: "", payload: null });
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
      const manifests = await fetchJson("/api/review-queue/first-pass-summaries/manifests?limit=5");
      setFirstPassBatchManifests(manifests);
      setReviewAction({
        status: payload.status || "submitted",
        message: `Submitted ${payload.request_count} requests; batch ${payload.batch_id || "pending id"}.`,
        manifest: payload.manifest || reviewAction.manifest,
        batchId: payload.batch_id || "",
        payload
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
      const manifests = await fetchJson("/api/review-queue/first-pass-summaries/manifests?limit=5");
      setFirstPassBatchManifests(manifests);
      const counts = payload.batch_counts || {};
      const countText = Object.entries(counts).map(([key, value]) => `${key}: ${value}`).join(", ");
      setReviewAction({
        status: payload.status || "checked",
        message: countText
          ? `Batch status ${payload.status}; ${countText}. Materialized ${payload.materialized?.length || 0}.`
          : `Batch status ${payload.status}.`,
        manifest: payload.manifest || reviewAction.manifest,
        batchId: payload.batch_id || reviewAction.batchId || "",
        payload
      });
    } catch (error) {
      setReviewAction((current) => ({ ...current, status: "error", message: `Status check failed: ${error.message}` }));
    }
  }

  function selectFirstPassBatchManifest(item) {
    if (!item?.manifest) return;
    setReviewAction({
      status: item.status || "selected",
      message: `Selected saved first-pass batch manifest with ${item.request_count || 0} requests.`,
      manifest: item.manifest,
      batchId: item.batch_id || "",
      payload: {
        ...item,
        batch_id: item.batch_id || null,
        batch_counts: item.batch_counts || {},
        materialized: Array.from({ length: item.materialized_count || 0 }),
        materialization_errors: Array.from({ length: item.materialization_error_count || 0 })
      }
    });
  }

  async function recordHumanReviewDecision(item, reviewActionName) {
    if (!item?.run_id || !item?.decision_id) return;
    const label = reviewActionName === "resolve" ? "Resolve" : reviewActionName === "reopen" ? "Reopen" : "Annotate";
    const defaultNote = reviewActionName === "annotate" ? "" : `${label} human-review item.`;
    const note = window.prompt(`${label} this App Intelligence human-review item. Add a local ledger note:`, defaultNote);
    if (note === null) return;
    if (!note.trim() && reviewActionName !== "annotate") {
      setHumanReviewAction({ status: "error", message: `${label} requires a note.`, payload: null });
      return;
    }
    setHumanReviewAction({ status: "running", message: `${label} human-review item...`, payload: null });
    try {
      const payload = await postJson(
        `/api/intelligence/runs/${encodeURIComponent(item.run_id)}/structured-decisions/${encodeURIComponent(item.decision_id)}/human-review`,
        {
          approval_token: "RECORD_HUMAN_REVIEW_DECISION",
          review_action: reviewActionName,
          reviewer: "operator",
          note
        }
      );
      const reviewPayload = await fetchJson("/api/review-queue?limit=100");
      const runsPayload = await fetchJson("/api/intelligence/runs?limit=8");
      setReviewQueue(reviewPayload);
      setIntelligence((current) => ({ ...current, runs: runsPayload }));
      setHumanReviewAction({
        status: "recorded",
        message: `${label} recorded in the local App Intelligence ledger; no external action was executed.`,
        payload
      });
    } catch (error) {
      setHumanReviewAction({ status: "error", message: `${label} failed: ${error.message}`, payload: null });
    }
  }

  function taskUpdatePayload() {
    return {
      provider: taskDraft.provider,
      model: taskDraft.model,
      timeout: taskDraft.timeout === "" ? "" : Number(taskDraft.timeout),
      temperature: taskDraft.temperature === "" ? "" : Number(taskDraft.temperature),
      fallbacks: taskDraft.fallbacks.split(",").map((item) => item.trim()).filter(Boolean),
      human_review: taskDraft.human_review,
      requires_ledger: taskDraft.requires_ledger
    };
  }

  async function previewConfigUpdate() {
    setConfigAction({ status: "running", message: "Previewing intelligence routing update...", preview: null });
    try {
      const payload = await postJson("/api/intelligence/config/preview", {
        task: selectedTask,
        update: taskUpdatePayload()
      });
      setConfigAction({
        status: "previewed",
        message: `Preview ready for ${selectedTask}; no config was written.`,
        preview: payload
      });
    } catch (error) {
      setConfigAction({ status: "error", message: `Preview failed: ${error.message}`, preview: null });
    }
  }

  async function applyConfigUpdate() {
    const preview = configAction.preview;
    if (!preview) return;
    const approved = window.confirm(`Apply intelligence routing update for ${selectedTask}?`);
    if (!approved) return;
    setConfigAction((current) => ({ ...current, status: "applying", message: "Applying intelligence routing update..." }));
    try {
      const payload = await postJson("/api/intelligence/config/apply", {
        task: selectedTask,
        update: taskUpdatePayload(),
        approval_token: "APPLY_INTELLIGENCE_CONFIG_UPDATE"
      });
      const configPayload = await fetchJson("/api/intelligence/config");
      setIntelligence((current) => ({ ...current, config: configPayload }));
      setConfigAction({
        status: "applied",
        message: `Applied ${payload.task}; rollback metadata is available in the last preview response.`,
        preview: payload
      });
    } catch (error) {
      setConfigAction((current) => ({ ...current, status: "error", message: `Apply failed: ${error.message}` }));
    }
  }

  async function prepareAppRun() {
    const taskLabel = statusLabel(selectedTask);
    setRunAction({ status: "running", message: `Preparing ${taskLabel} run ledger...`, runId: "" });
    try {
      const payload = await postJson("/api/intelligence/runs/prepare", {
        workflow: selectedTask,
        task: selectedTask,
        purpose: `Prepare a supervised ${taskLabel} intelligence run from the review console.`,
        document_id: selected?.id || "",
        created_by: "review-console"
      });
      const runsPayload = await fetchJson("/api/intelligence/runs?limit=8");
      setIntelligence((current) => ({ ...current, runs: runsPayload }));
      setSelectedRunId(payload.run?.run_id || "");
      setSelectedRunDetail(payload);
      setRunReplayManifest(null);
      setRunAction({
        status: "prepared",
        message: `Prepared local run ledger ${payload.run?.run_id || ""}; no provider work was started.`,
        runId: payload.run?.run_id || ""
      });
    } catch (error) {
      setRunAction({ status: "error", message: `Run prepare failed: ${error.message}`, runId: "" });
    }
  }

  async function runSessionPreflight({ appendEvent = false } = {}) {
    if (!selectedRunId) return;
    setSessionPreflight({
      status: appendEvent ? "recording" : "running",
      message: appendEvent ? "Recording non-starting session preflight event..." : "Running session-start preflight...",
      payload: sessionPreflight.payload
    });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/session-start-preflight`, {
        approval_token: appendEvent ? "APPEND_SESSION_START_PREFLIGHT_EVENT" : "START_APP_SERVER_SESSION",
        append_event: appendEvent
      });
      if (appendEvent) {
        await refreshSelectedRun();
      }
      setSessionPreflight({
        status: payload.ok ? "ok" : "blocked",
        message: appendEvent
          ? "Recorded preflight event; no app-server session was started."
          : payload.ok
            ? "Preflight passed; session start still requires a future explicit action."
            : `Preflight blocked: ${payload.blocking_checks?.join(", ") || "unknown check"}.`,
        payload
      });
    } catch (error) {
      setSessionPreflight({ status: "error", message: `Session preflight failed: ${error.message}`, payload: null });
    }
  }

  async function startAppServerSession() {
    if (!selectedRunId) return;
    const approved = window.confirm("Start the Codex app-server control-plane daemon for this prepared ledger? This does not start a model turn.");
    if (!approved) return;
    setSessionStartAction({ status: "starting", message: "Starting app-server control plane...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/session-start`, {
        approval_token: "START_APP_SERVER_SESSION",
        transport: "stdio"
      });
      await refreshSelectedRun();
      setSessionStartAction({
        status: payload.ok ? "started" : "blocked",
        message: payload.ok
          ? "App-server control plane started; no model turn was started."
          : "App-server session start did not complete.",
        payload
      });
    } catch (error) {
      setSessionStartAction({ status: "error", message: `Session start failed: ${error.message}`, payload: null });
    }
  }

  async function prepareModelTurnPacket() {
    if (!selectedRunId) return;
    const approved = window.confirm("Prepare a reviewed prompt packet for this run? This writes a local artifact but does not send a prompt.");
    if (!approved) return;
    setModelTurnAction({ status: "preparing", message: "Preparing reviewed prompt packet...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/model-turn-preflight`, {
        approval_token: "PREPARE_MODEL_TURN_PREFLIGHT",
        task: selectedTask,
        document_id: selected?.id || selectedRunDetail?.run?.document_id || ""
      });
      await refreshSelectedRun();
      setModelTurnAction({
        status: "prepared",
        message: "Prompt packet prepared for review; no prompt was sent.",
        payload
      });
      setSelectedPacketId(payload.packet?.packet_id || "");
    } catch (error) {
      setModelTurnAction({ status: "error", message: `Prompt packet preflight failed: ${error.message}`, payload: null });
    }
  }

  async function runModelTurnSendPreflight() {
    if (!selectedRunId || !selectedPacketId) return;
    setSendPreflight({ status: "running", message: "Checking model-turn send preflight; no prompt will be sent...", payload: null });
    try {
      const payload = await postJson(
        `/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/prompt-packets/${encodeURIComponent(selectedPacketId)}/send-preflight`,
        { approval_token: "SEND_APP_SERVER_MODEL_TURN" }
      );
      setSendPreflight({
        status: payload.ok ? "ok" : "blocked",
        message: payload.ok
          ? "Send preflight passed; no prompt was sent and no event was written."
          : `Send preflight blocked: ${payload.blocking_checks?.join(", ") || "unknown check"}.`,
        payload
      });
    } catch (error) {
      setSendPreflight({ status: "error", message: `Send preflight failed: ${error.message}`, payload: null });
    }
  }

  async function sendModelTurn() {
    if (!selectedRunId || !selectedPacketId) return;
    const approved = window.confirm("Send this reviewed prompt packet to Codex app-server? This starts a model turn and records ledger events, but will not execute downstream writes.");
    if (!approved) return;
    setSendAction({ status: "sending", message: "Sending reviewed packet to Codex app-server...", payload: null });
    try {
      const payload = await postJson(
        `/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/prompt-packets/${encodeURIComponent(selectedPacketId)}/send`,
        { approval_token: "SEND_APP_SERVER_MODEL_TURN" }
      );
      await refreshSelectedRun();
      setSendAction({
        status: payload.ok ? "started" : "blocked",
        message: payload.ok
          ? "Model turn started and ledger events were recorded; no downstream action was executed."
          : "Model turn send did not start.",
        payload
      });
    } catch (error) {
      setSendAction({ status: "error", message: `Model turn send failed: ${error.message}`, payload: null });
    }
  }

  async function captureTurnStatus() {
    if (!selectedRunId) return;
    setTurnStatusAction({ status: "capturing", message: "Capturing Codex turn status and output...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/turn-status`, {
        approval_token: "CAPTURE_MODEL_TURN_STATUS"
      });
      await refreshSelectedRun();
      setTurnStatusAction({
        status: payload.completed ? "completed" : "captured",
        message: payload.completed
          ? "Turn completion/output captured; no structured decision was executed."
          : "Turn status captured; no structured decision was executed.",
        payload
      });
    } catch (error) {
      setTurnStatusAction({ status: "error", message: `Turn status capture failed: ${error.message}`, payload: null });
    }
  }

  async function validateStructuredDecision() {
    if (!selectedRunId) return;
    setDecisionValidation({ status: "validating", message: "Validating captured structured decision...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/structured-decision/validate`, {
        approval_token: "VALIDATE_STRUCTURED_DECISION"
      });
      await refreshSelectedRun();
      setDecisionValidation({
        status: payload.valid ? "valid" : "rejected",
        message: payload.valid
          ? "Structured decision validated; no host action was executed."
          : `Structured decision rejected: ${payload.errors?.join(", ") || "schema validation failed"}.`,
        payload
      });
    } catch (error) {
      setDecisionValidation({ status: "error", message: `Structured decision validation failed: ${error.message}`, payload: null });
    }
  }

  async function applyStructuredDecision(decisionId) {
    if (!selectedRunId || !decisionId) return;
    const approved = window.confirm("Record this validated ledger-only decision? This will not fork, rollback, write memory, route artifacts, or touch external systems.");
    if (!approved) return;
    setDecisionApply({ status: "applying", message: "Recording ledger-only structured decision...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/structured-decisions/${encodeURIComponent(decisionId)}/apply`, {
        approval_token: "APPLY_STRUCTURED_DECISION",
        reviewer: "operator"
      });
      await refreshSelectedRun();
      setDecisionApply({
        status: "applied",
        message: "Ledger-only structured decision recorded; no external or write-bearing action was executed.",
        payload
      });
    } catch (error) {
      setDecisionApply({ status: "error", message: `Structured decision apply failed: ${error.message}`, payload: null });
    }
  }

  async function runForkPreflight(decisionId) {
    if (!selectedRunId || !decisionId) return;
    setForkPreflightAction({ status: "running", message: "Previewing branch fork plan...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/structured-decisions/${encodeURIComponent(decisionId)}/fork-preflight`, {
        approval_token: "PREVIEW_FORK_BRANCHES",
        reviewer: "operator"
      });
      await refreshSelectedRun();
      setForkPreflightAction({
        status: "previewed",
        message: "Fork preflight recorded; no threads, branches, or provider work were started.",
        payload
      });
    } catch (error) {
      setForkPreflightAction({ status: "error", message: `Fork preflight failed: ${error.message}`, payload: null });
    }
  }

  async function runRollbackPreflight(decisionId) {
    if (!selectedRunId || !decisionId) return;
    setRollbackPreflightAction({ status: "running", message: "Previewing rollback plan...", payload: null });
    try {
      const payload = await postJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/structured-decisions/${encodeURIComponent(decisionId)}/rollback-preflight`, {
        approval_token: "PREVIEW_ROLLBACK",
        reviewer: "operator"
      });
      await refreshSelectedRun();
      setRollbackPreflightAction({
        status: "previewed",
        message: "Rollback preflight recorded; no branches, artifacts, threads, or provider work were changed.",
        payload
      });
    } catch (error) {
      setRollbackPreflightAction({ status: "error", message: `Rollback preflight failed: ${error.message}`, payload: null });
    }
  }

  async function loadRunArtifact(artifactPath) {
    if (!selectedRunId || !artifactPath) return;
    setRunArtifactAction({ status: "loading", message: "Loading registered run artifact...", payload: null });
    try {
      const payload = await fetchJson(
        `/api/intelligence/runs/${encodeURIComponent(selectedRunId)}/artifacts?path=${encodeURIComponent(artifactPath)}`
      );
      setRunArtifactAction({
        status: "loaded",
        message: `Loaded ${payload.relative_path || "artifact"}; no write or external action was executed.`,
        payload
      });
    } catch (error) {
      setRunArtifactAction({ status: "error", message: `Artifact load failed: ${error.message}`, payload: null });
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
              className={activeNav === item.id ? "active" : ""}
              aria-current={activeNav === item.id ? "page" : undefined}
              disabled={!item.enabled}
              key={item.id}
              onClick={() => item.enabled && setActiveNav(item.id)}
              title={item.enabled ? "" : `${item.label} is planned and not wired yet.`}
              type="button"
            >
              {item.label}
              {!item.enabled && <span>planned</span>}
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
        style={{
          "--left-pane-width": `${leftPaneWidth}px`,
          "--right-pane-width": `${rightPaneWidth}px`
        }}
      >
        <aside className="left-pane">
          <PaneToggleButton
            collapsed={leftCollapsed}
            label={leftCollapsed ? "Expand filters pane" : "Collapse filters pane"}
            onClick={() => setLeftCollapsed((value) => !value)}
            side="left"
          />
          {!leftCollapsed && (
            <PaneResizeHandle
              label="Resize filters pane"
              onKeyDown={(event) => resizePaneWithKeyboard("left", event)}
              onPointerDown={(event) => resizePane("left", event)}
              side="left"
            />
          )}
          <div className="pane-content">
            <p className="eyebrow">{activeNav}</p>
            <h2>Workflow filters</h2>
            {activeNav === "Intelligence" ? (
              <div className="filter-card task-filter">
                <span>Task routing</span>
                {taskEntries.map(([task, config]) => (
                  <button className={selectedTask === task ? "selected-filter" : ""} key={task} onClick={() => setSelectedTask(task)} type="button">
                    {statusLabel(task)}
                    <strong>{config.provider}</strong>
                  </button>
                ))}
              </div>
            ) : (
              <>
                <div className="filter-card">
                  <span>Kind</span>
                  {LIBRARY_KIND_FILTERS.map((filter) => (
                    <button
                      aria-pressed={kindFilter === filter.id}
                      className={kindFilter === filter.id ? "selected-filter" : ""}
                      key={filter.id}
                      onClick={() => setKindFilter(filter.id)}
                      type="button"
                    >
                      {filter.label}
                      <strong>{filterCount(library.items || [], filter.id)}</strong>
                    </button>
                  ))}
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
              </>
            )}
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
              <h1>{activeNav === "Review Queue" ? "Review queue" : activeNav === "Intelligence" ? "Intelligence routing" : "Transcript library"}</h1>
            </div>
            <div className="summary-strip">
              <span>{conversations.total ?? visibleConversationRows.length} conversations</span>
              <span>{library.total ?? visibleItems.length} artifacts</span>
              <span>{reviewQueue.total_open ?? reviewBuckets.reduce((total, item) => total + item.count, 0)} open reviews</span>
              {activeNav === "Intelligence" && <span>{taskEntries.length} task routes</span>}
            </div>
          </div>
          <TestStatusStrip
            activeNav={activeNav}
            apiStatus={health.status}
            kindFilter={kindFilter}
            query={query}
            visibleCount={activeNav === "Library" ? visibleConversationRows.length : visibleItems.length}
            totalCount={activeNav === "Library" ? totalConversationCount : library.total ?? (library.items || []).length}
            latestSmoke={intelligence.smokes?.latest_report}
            latestSmokeJob={intelligence.smokeJobs?.items?.[0]}
          />

          {activeNav === "Review Queue" ? (
            <ReviewQueue
              queue={reviewQueue}
              reviewAction={reviewAction}
              batchManifests={firstPassBatchManifests}
              onPrepareFirstPass={prepareFirstPassBatch}
              onSubmitFirstPass={submitFirstPassBatch}
              onRefreshFirstPass={refreshFirstPassBatch}
              onSelectFirstPassManifest={selectFirstPassBatchManifest}
              humanReviewAction={humanReviewAction}
              onRecordHumanReview={recordHumanReviewDecision}
            />
          ) : activeNav === "Intelligence" ? (
            <IntelligencePanel
              config={intelligence.config}
              providers={intelligence.providers}
              runs={intelligence.runs}
              smokes={intelligence.smokes}
              smokeJobs={intelligence.smokeJobs}
              smokeJobAction={smokeJobAction}
              smokeTailAction={smokeTailAction}
              onStartSmokeJob={startSmokeJob}
              onLoadSmokeJobTail={loadSmokeJobTail}
              onRefreshSmokeEvidence={refreshSmokeEvidence}
              selectedTask={selectedTask}
              selectedTaskConfig={selectedTaskConfig}
              selectedProvider={selectedProvider}
              selectedDocument={selected}
              taskDraft={taskDraft}
              setTaskDraft={setTaskDraft}
              configAction={configAction}
              runAction={runAction}
              selectedRunId={selectedRunId}
              onPreview={previewConfigUpdate}
              onApply={applyConfigUpdate}
              onPrepareRun={prepareAppRun}
              onSelectRun={setSelectedRunId}
            />
          ) : (
            <LibraryTable
              rows={visibleConversationRows}
              allItems={library.items || []}
              searchStatus={conversationSearchStatus}
              usingApiRows={usingApiConversations}
              loadedCount={loadedConversationCount}
              totalCount={totalConversationCount}
              canLoadMore={canLoadMoreConversations}
              selectedId={selected?.id}
              onLoadMore={loadMoreConversations}
              onOpenConversation={() => setConversationOpen(true)}
              onSelect={setSelectedId}
            />
          )}
        </section>

        <aside className="right-pane">
          <PaneToggleButton
            collapsed={rightCollapsed}
            label={rightCollapsed ? "Expand inspector pane" : "Collapse inspector pane"}
            onClick={() => setRightCollapsed((value) => !value)}
            side="right"
          />
          {!rightCollapsed && (
            <PaneResizeHandle
              label="Resize inspector pane"
              onKeyDown={(event) => resizePaneWithKeyboard("right", event)}
              onPointerDown={(event) => resizePane("right", event)}
              side="right"
            />
          )}
          <Inspector
            item={selected}
            items={library.items || []}
            activeNav={activeNav}
            documentDetail={selectedDocumentDetail}
            documentDetailAction={selectedDocumentDetailAction}
            relatedDocuments={selectedRelatedDocuments}
            onOpenConversation={() => setConversationOpen(true)}
            onSelectDocument={setSelectedId}
            reviewQueue={reviewQueue}
            selectedTask={selectedTask}
            selectedTaskConfig={selectedTaskConfig}
            selectedProvider={selectedProvider}
            configAction={configAction}
            runAction={runAction}
            selectedRunId={selectedRunId}
            selectedRunDetail={selectedRunDetail}
            runReplayManifest={runReplayManifest}
            runDetailAction={runDetailAction}
            sessionPreflight={sessionPreflight}
            onRunSessionPreflight={runSessionPreflight}
            sessionStartAction={sessionStartAction}
            onStartAppServerSession={startAppServerSession}
            modelTurnAction={modelTurnAction}
            onPrepareModelTurnPacket={prepareModelTurnPacket}
            selectedPacketId={selectedPacketId}
            onSelectPacket={setSelectedPacketId}
            packetReview={packetReview}
            sendPreflight={sendPreflight}
            onRunModelTurnSendPreflight={runModelTurnSendPreflight}
            sendAction={sendAction}
            onSendModelTurn={sendModelTurn}
            turnStatusAction={turnStatusAction}
            onCaptureTurnStatus={captureTurnStatus}
            decisionValidation={decisionValidation}
            onValidateStructuredDecision={validateStructuredDecision}
            decisionApply={decisionApply}
            onApplyStructuredDecision={applyStructuredDecision}
            forkPreflightAction={forkPreflightAction}
            onRunForkPreflight={runForkPreflight}
            rollbackPreflightAction={rollbackPreflightAction}
            onRunRollbackPreflight={runRollbackPreflight}
            runArtifactAction={runArtifactAction}
            onLoadRunArtifact={loadRunArtifact}
            intelligence={intelligence}
          />
        </aside>
      </section>
      {conversationOpen && activeNav === "Library" && selected ? (
        <ConversationWorkflowModal
          conversationDetail={selectedConversationDetail}
          conversationDetailAction={selectedConversationDetailAction}
          documentDetail={selectedDocumentDetail}
          documentDetailAction={selectedDocumentDetailAction}
          relatedDocuments={selectedRelatedDocuments}
          item={selected}
          items={library.items || []}
          onClose={() => setConversationOpen(false)}
          onSelectDocument={setSelectedId}
        />
      ) : null}
    </main>
  );
}

function TestStatusStrip({
  activeNav,
  apiStatus,
  kindFilter,
  query,
  visibleCount,
  totalCount,
  latestSmoke,
  latestSmokeJob
}) {
  const target =
    activeNav === "Intelligence"
      ? "Queue a smoke, inspect the tail, then verify the latest report."
      : activeNav === "Review Queue"
        ? "Pick one queue bucket, run a dry preview, then materialize only after review."
        : "Search or filter, select a row, then verify playback and source metadata in the inspector.";
  return (
    <section className="test-status-strip" aria-label="Operator test status">
      <div>
        <span>API</span>
        <strong>{apiStatus === "ok" ? "Live" : "Preview"}</strong>
      </div>
      <div>
        <span>Rows in scope</span>
        <strong>{visibleCount} / {totalCount}</strong>
      </div>
      <div>
        <span>Filter</span>
        <strong>{statusLabel(kindFilter)}{query ? ` + "${query}"` : ""}</strong>
      </div>
      <div>
        <span>Latest smoke</span>
        <strong>{latestSmokeJob?.status || latestSmoke?.status || "none"}</strong>
      </div>
      <p>{target}</p>
    </section>
  );
}

function PaneToggleButton({ collapsed, label, onClick, side }) {
  const points = side === "left"
    ? collapsed ? "10 8 16 12 10 16" : "16 8 10 12 16 16"
    : collapsed ? "14 8 8 12 14 16" : "8 8 14 12 8 16";
  return (
    <button
      aria-label={label}
      aria-pressed={collapsed}
      className={`pane-toggle icon-pane-toggle ${side}`}
      onClick={onClick}
      title={label}
      type="button"
    >
      <svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">
        <rect x="4" y="5" width="16" height="14" rx="3" />
        <line x1={side === "left" ? "9" : "15"} y1="6" x2={side === "left" ? "9" : "15"} y2="18" />
        <polyline points={points} />
      </svg>
      <span>{side === "left" ? "Filters" : "Inspector"}</span>
    </button>
  );
}

function PaneResizeHandle({ label, onKeyDown, onPointerDown, side }) {
  return (
    <div
      aria-label={label}
      aria-orientation="vertical"
      className={`pane-resizer ${side}`}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      role="separator"
      tabIndex={0}
      title={`${label}. Drag or use arrow keys.`}
    />
  );
}

function IntelligencePanel({
  config,
  providers,
  runs,
  smokes,
  smokeJobs,
  smokeJobAction,
  smokeTailAction,
  onStartSmokeJob,
  onLoadSmokeJobTail,
  onRefreshSmokeEvidence,
  selectedTask,
  selectedTaskConfig,
  selectedProvider,
  selectedDocument,
  taskDraft,
  setTaskDraft,
  configAction,
  runAction,
  selectedRunId,
  onPreview,
  onApply,
  onPrepareRun,
  onSelectRun
}) {
  const [smokeJobFilter, setSmokeJobFilter] = useState("all");
  const providerList = providers?.providers || [];
  const taskEntries = Object.entries(config?.tasks || {});
  const recentRuns = runs?.items || [];
  const recentSmokeJobs = smokeJobs?.items || [];
  const totalSmokeJobs = smokeJobs?.total || recentSmokeJobs.length;
  const filteredSmokeJobs = filterSmokeJobs(recentSmokeJobs, smokeJobFilter);
  const failedSmokeJobs = filteredSmokeJobs.filter((job) => job.status === "failed");
  const latestSmoke = smokes?.latest_report || null;
  const latestSmokeChecks = latestSmoke?.checks && typeof latestSmoke.checks === "object" ? Object.entries(latestSmoke.checks) : [];
  const selectedCapabilities = capabilityLabels(selectedProvider?.capabilities);
  const selectedChecks = selectedProvider?.checks && typeof selectedProvider.checks === "object" ? Object.entries(selectedProvider.checks) : [];
  const smokeJobGroups = groupSmokeJobsByType(filteredSmokeJobs.filter((job) => job.status !== "failed"));
  const smokeJobFilters = [
    { id: "all", label: "All", count: recentSmokeJobs.length },
    { id: "failed", label: "Failed", count: recentSmokeJobs.filter((job) => job.status === "failed").length },
    { id: "write_bearing", label: "Write-bearing", count: recentSmokeJobs.filter((job) => job.will_execute_write_bearing_action).length },
    { id: "evidence", label: "Evidence", count: recentSmokeJobs.filter((job) => Boolean(job.evidence_summary)).length }
  ];
  return (
    <div className="intelligence-grid">
      <section className="intelligence-card task-editor">
        <p className="eyebrow">Task Config</p>
        <h2>{statusLabel(selectedTask)}</h2>
        <p>Preview changes first. Apply writes only to the user-scoped intelligence config with an approval token.</p>
        <div className="editor-grid">
          <label>
            <span>Provider</span>
            <select value={taskDraft.provider} onChange={(event) => setTaskDraft((draft) => ({ ...draft, provider: event.target.value }))}>
              {[...new Set([taskDraft.provider, ...providerList.map((provider) => provider.id)])].filter(Boolean).map((providerId) => (
                <option key={providerId} value={providerId}>{providerId}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Model</span>
            <input value={taskDraft.model} onChange={(event) => setTaskDraft((draft) => ({ ...draft, model: event.target.value }))} placeholder="provider default" />
          </label>
          <label>
            <span>Timeout</span>
            <input type="number" value={taskDraft.timeout} onChange={(event) => setTaskDraft((draft) => ({ ...draft, timeout: event.target.value }))} />
          </label>
          <label>
            <span>Temperature</span>
            <input type="number" step="0.1" value={taskDraft.temperature} onChange={(event) => setTaskDraft((draft) => ({ ...draft, temperature: event.target.value }))} />
          </label>
          <label>
            <span>Fallbacks</span>
            <input value={taskDraft.fallbacks} onChange={(event) => setTaskDraft((draft) => ({ ...draft, fallbacks: event.target.value }))} placeholder="codex-exec, openai-compatible" />
          </label>
          <label>
            <span>Human review</span>
            <input value={taskDraft.human_review} onChange={(event) => setTaskDraft((draft) => ({ ...draft, human_review: event.target.value }))} />
          </label>
          <label className="checkbox-line">
            <input type="checkbox" checked={taskDraft.requires_ledger} onChange={(event) => setTaskDraft((draft) => ({ ...draft, requires_ledger: event.target.checked }))} />
            <span>Requires run ledger</span>
          </label>
        </div>
        <div className="notice-actions">
          <button onClick={onPreview} disabled={configAction.status === "running"} type="button">Preview update</button>
          <button onClick={onApply} disabled={!configAction.preview || configAction.status === "applying"} type="button">Apply with approval</button>
          <button onClick={onPrepareRun} disabled={runAction.status === "running"} type="button">Prepare run ledger</button>
        </div>
        {configAction.message && <div className={`action-notice ${configAction.status}`}><strong>{configAction.message}</strong></div>}
        {runAction.message && <div className={`action-notice ${runAction.status}`}><strong>{runAction.message}</strong></div>}
      </section>

      <section className="intelligence-card provider-map">
        <p className="eyebrow">Provider Status</p>
        <h2>{selectedProvider?.label || selectedTaskConfig?.provider || "No provider"}</h2>
        <ProviderDetails provider={selectedProvider} capabilities={selectedCapabilities} checks={selectedChecks} />
        <div className="provider-list">
          {providerList.map((provider) => (
            <article className={provider.id === selectedTaskConfig?.provider ? "provider-row active" : "provider-row"} key={provider.id}>
              <div>
                <strong>{provider.label || provider.id}</strong>
                <small>{provider.id}</small>
              </div>
              <span>{provider.status || (provider.ready ? "ready" : "unknown")}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="intelligence-card task-map">
        <p className="eyebrow">Resolved Routes</p>
        <h2>{config?.schema_version}</h2>
        <p className="muted">Ledger prepare uses the selected task route and links the currently selected document when one is available.</p>
        {selectedDocument && <p className="linked-document">Selected document: <strong>{selectedDocument.title || selectedDocument.id}</strong></p>}
        <div className="task-table">
          {taskEntries.map(([task, route]) => (
            <article className={task === selectedTask ? "task-row active" : "task-row"} key={task}>
              <strong>{statusLabel(task)}</strong>
              <span>{route.provider}</span>
              <small>{route.model || "provider default"} · {route.source}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="intelligence-card run-ledgers">
        <p className="eyebrow">Prepared Ledgers</p>
        <h2>Recent app runs</h2>
        <div className="run-list">
          {recentRuns.length ? (
            recentRuns.map((run) => (
              <button
                className={selectedRunId === run.run_id ? "run-row active" : "run-row"}
                key={run.run_id}
                onClick={() => onSelectRun(run.run_id)}
                type="button"
              >
                <div>
                  <strong>{run.workflow || run.run_id}</strong>
                  <small>{run.run_id}</small>
                </div>
                <span>{run.phase || run.status}</span>
              </button>
            ))
          ) : (
            <p className="muted">No prepared app-intelligence run ledgers yet.</p>
          )}
        </div>
      </section>

      <section className="intelligence-card run-ledgers">
        <p className="eyebrow">Smoke Status</p>
        <h2>{latestSmoke ? `${latestSmoke.status || "unknown"} latest smoke` : "No smoke evidence"}</h2>
        {latestSmoke ? (
          <>
            <dl>
              <dt>Run</dt>
              <dd>{latestSmoke.run_id || "Unavailable"}</dd>
              <dt>Report</dt>
              <dd>{latestSmoke.path || "Unavailable"}</dd>
              <dt>Screenshot</dt>
              <dd>{latestSmoke.screenshot_exists ? latestSmoke.screenshot_path : "Unavailable"}</dd>
              <dt>Evidence</dt>
              <dd>{smokes?.report_count || 0} reports · {smokes?.run_count || 0} smoke runs</dd>
            </dl>
            {latestSmokeChecks.length ? (
              <div className="provider-checks">
                {latestSmokeChecks.slice(0, 7).map(([name, ok]) => (
                  <span className={ok ? "check-ok" : "check-warn"} key={name}>{statusLabel(name)}</span>
                ))}
              </div>
            ) : null}
          </>
        ) : (
          <p className="muted">Run `python scripts/smoke_app_replay_manifest_ui.py --cleanup` to generate browser-smoke evidence.</p>
        )}
        <div className="notice-actions">
          <button onClick={() => onStartSmokeJob("api_replay_smoke")} disabled={smokeJobAction.status === "queueing"} type="button">
            Queue API smoke
          </button>
          <button onClick={() => onStartSmokeJob("browser_replay_smoke")} disabled={smokeJobAction.status === "queueing"} type="button">
            Queue browser smoke
          </button>
          <button onClick={() => onStartSmokeJob("first_pass_resume_ui_smoke")} disabled={smokeJobAction.status === "queueing"} type="button">
            Queue resume UI smoke
          </button>
          <button onClick={() => onStartSmokeJob("cleanup_smokes")} disabled={smokeJobAction.status === "queueing"} type="button">
            Cleanup dry-run
          </button>
          <button onClick={() => onStartSmokeJob("cleanup_smokes", { applyCleanup: true })} disabled={smokeJobAction.status === "queueing"} type="button">
            Apply cleanup
          </button>
          <button onClick={onRefreshSmokeEvidence} type="button">
            Refresh smoke state
          </button>
        </div>
        {smokeJobAction.message && (
          <div className={`action-notice ${smokeJobAction.status}`}>
            <strong>{smokeJobAction.message}</strong>
            {smokeJobAction.payload?.job && <code>{JSON.stringify(smokeJobAction.payload.job, null, 2)}</code>}
          </div>
        )}
        {recentSmokeJobs.length ? (
          <div className="task-table">
            <div className="smoke-risk-legend" aria-label="Smoke job risk legend">
              <span className="risk-badge write-bearing">write-bearing</span>
              <small>Can delete allowlisted disposable smoke artifacts after typed approval.</small>
              <span className="risk-badge read-only">read-only</span>
              <small>Inspects status, tails, or dry-run cleanup counts without deleting artifacts.</small>
            </div>
            <p className="smoke-job-page-hint">
              Showing {filteredSmokeJobs.length} of {recentSmokeJobs.length} loaded smoke jobs from the current API page ({totalSmokeJobs} retained).
            </p>
            <div className="smoke-job-filters" aria-label="Smoke job filters">
              {smokeJobFilters.map((filter) => (
                <button
                  aria-pressed={smokeJobFilter === filter.id}
                  className={smokeJobFilter === filter.id ? "active" : ""}
                  key={filter.id}
                  onClick={() => setSmokeJobFilter(filter.id)}
                  type="button"
                >
                  {filter.label}
                  <strong>{filter.count}</strong>
                </button>
              ))}
            </div>
            {failedSmokeJobs.length ? (
              <section className="smoke-failure-band" aria-label="Failed smoke jobs">
                <div className="smoke-failure-heading">
                  <strong>{failedSmokeJobs.length} failed smoke job{failedSmokeJobs.length === 1 ? "" : "s"} need review</strong>
                  <span>Loaded failures are shown before successful history.</span>
                </div>
                {failedSmokeJobs.map((job) => {
                  const timing = smokeJobTiming(job);
                  return (
                    <article className="task-row smoke-job-row failed" key={job.job_id}>
                      <strong>{statusLabel(job.job_type || "smoke job")}</strong>
                      <span>{job.status}</span>
                      <span className="risk-badge failed">failed</span>
                      <small>{job.job_id}</small>
                      {timing && <small className="job-timing">{timing}</small>}
                      <div className="notice-actions">
                        <button onClick={() => onLoadSmokeJobTail(job.job_id, "stderr")} type="button">
                          stderr tail
                        </button>
                        <button onClick={() => onLoadSmokeJobTail(job.job_id, "stdout")} type="button">
                          stdout tail
                        </button>
                      </div>
                    </article>
                  );
                })}
              </section>
            ) : (
              <p className="smoke-job-page-hint">No failed smoke jobs in this filtered view.</p>
            )}
            {smokeJobGroups.length ? smokeJobGroups.map((group) => (
              <section className="smoke-job-group" key={group.key}>
                <div className="smoke-job-group-heading">
                  <strong>{group.label}</strong>
                  <span>{group.jobs.length} loaded</span>
                </div>
                {group.jobs.map((job) => {
                  const timing = smokeJobTiming(job);
                  return (
                    <article className={`task-row smoke-job-row ${job.will_execute_write_bearing_action ? "write-bearing" : "read-only"}`} key={job.job_id}>
                      <strong>{statusLabel(job.job_type || "smoke job")}</strong>
                      <span>{job.status}</span>
                      <span className={job.will_execute_write_bearing_action ? "risk-badge write-bearing" : "risk-badge read-only"}>
                        {job.will_execute_write_bearing_action ? "write-bearing" : "read-only"}
                      </span>
                      <small>{job.job_id}</small>
                      {timing && <small className="job-timing">{timing}</small>}
                      {job.cleanup_summary && <small>{cleanupSummaryLabel(job.cleanup_summary)}</small>}
                      {job.evidence_summary && (
                        <div className="smoke-evidence-card">
                          <span>Evidence {job.evidence_summary.status}</span>
                          <small>{job.evidence_summary.check_count || 0} checks · {job.evidence_summary.failed_check_count || 0} failed</small>
                          <div className="notice-actions">
                            {job.evidence_summary.report_url && (
                              <a href={job.evidence_summary.report_url} rel="noreferrer" target="_blank">
                                Open report JSON
                              </a>
                            )}
                            {job.evidence_summary.screenshot_url && (
                              <a href={job.evidence_summary.screenshot_url} rel="noreferrer" target="_blank">
                                Open screenshot
                              </a>
                            )}
                          </div>
                        </div>
                      )}
                      <div className="notice-actions">
                        <button onClick={() => onLoadSmokeJobTail(job.job_id, "stderr")} type="button">
                          stderr tail
                        </button>
                        <button onClick={() => onLoadSmokeJobTail(job.job_id, "stdout")} type="button">
                          stdout tail
                        </button>
                      </div>
                    </article>
                  );
                })}
              </section>
            )) : (
              <p className="smoke-job-page-hint">No non-failed smoke jobs match this filter.</p>
            )}
          </div>
        ) : null}
        {smokeTailAction.message && (
          <div className={`action-notice ${smokeTailAction.status}`}>
            <strong>{smokeTailAction.message}</strong>
            {smokeTailAction.payload && <code>{JSON.stringify({
              job_id: smokeTailAction.payload.job_id,
              stream: smokeTailAction.payload.stream,
              exists: smokeTailAction.payload.exists,
              bytes: smokeTailAction.payload.bytes,
              will_read_arbitrary_file: smokeTailAction.payload.will_read_arbitrary_file
            }, null, 2)}</code>}
            {smokeTailAction.payload && (
              <pre className="prompt-preview">{smokeTailAction.payload.tail || "(empty)"}</pre>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function ProviderDetails({ provider, capabilities, checks }) {
  if (!provider) return <p className="muted">No provider registry details are available for this task route.</p>;
  return (
    <div className="provider-details">
      <dl>
        <dt>Control plane</dt>
        <dd>{provider.control_plane || "unknown"}</dd>
        <dt>Readiness</dt>
        <dd>{provider.ready === true ? "ready" : provider.ready === false ? "not ready" : provider.status || "unknown"}</dd>
        {provider.version && (
          <>
            <dt>Version</dt>
            <dd>{provider.version}</dd>
          </>
        )}
      </dl>
      {capabilities.length ? (
        <div className="capability-cloud">
          {capabilities.slice(0, 12).map((capability) => <span key={capability}>{capability}</span>)}
        </div>
      ) : null}
      {provider.notes?.length ? (
        <ul className="provider-notes">
          {provider.notes.slice(0, 3).map((note) => <li key={note}>{note}</li>)}
        </ul>
      ) : null}
      {checks.length ? (
        <div className="provider-checks">
          {checks.slice(0, 4).map(([name, check]) => (
            <span className={check?.ok ? "check-ok" : "check-warn"} key={name}>{statusLabel(name)}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function DecisionHistory({ decisions, onLoadArtifact }) {
  if (!decisions?.length) return <p className="muted">No structured decisions have been validated for this run.</p>;
  return (
    <div className="decision-history">
      {decisions.slice().reverse().map((decision, index) => {
        const applyResult = decision.apply_result || {};
        const isLatest = index === 0;
        return (
          <article className={`decision-row ${decision.status || "unknown"} ${isLatest ? "latest" : ""}`} key={decision.decision_id}>
            <div className="decision-row-heading">
              <strong>{decision.action || "unknown action"}</strong>
              <span>{decision.status || "unknown"}</span>
            </div>
            <small>{isLatest ? "Latest decision" : formatDate(decision.applied_at || decision.created_at)}</small>
            <dl>
              <dt>Decision id</dt>
              <dd>{decision.decision_id || "Unavailable"}</dd>
              <dt>Validation artifact</dt>
              <dd>{decision.artifact_path || "Unavailable"}</dd>
              <dt>Codex turn</dt>
              <dd>{decision.codex_turn_id || "Unavailable"}</dd>
              <dt>Apply artifact</dt>
              <dd>{applyResult.artifact_path || "Not applied"}</dd>
              <dt>Apply event</dt>
              <dd>{decision.apply_event_id || "Not applied"}</dd>
            </dl>
            <div className="notice-actions">
              {decision.artifact_path && (
                <button onClick={() => onLoadArtifact(decision.artifact_path)} type="button">Open validation JSON</button>
              )}
              {applyResult.artifact_path && (
                <button onClick={() => onLoadArtifact(applyResult.artifact_path)} type="button">Open apply JSON</button>
              )}
            </div>
            <code>{JSON.stringify({
              valid: decision.valid,
              will_execute_host_action: decision.will_execute_host_action,
              applied_ledger_state: applyResult.applied_ledger_state ?? null,
              current_branch: applyResult.current_branch || null,
              will_execute_write_bearing_action: applyResult.will_execute_write_bearing_action ?? null
            }, null, 2)}</code>
          </article>
        );
      })}
    </div>
  );
}

function PreflightArtifacts({ events, onLoadArtifact }) {
  const artifacts = (events || [])
    .filter((event) => event?.event_type?.includes("preflight") && event.payload?.artifact_path)
    .slice()
    .reverse();
  if (!artifacts.length) return <p className="muted">No preflight artifacts are recorded for this run.</p>;
  return (
    <div className="preflight-artifacts">
      {artifacts.map((event) => (
        <button
          className="preflight-artifact-row"
          key={event.event_id || `${event.event_type}-${event.created_at}`}
          onClick={() => onLoadArtifact(event.payload.artifact_path)}
          type="button"
        >
          <span>{statusLabel(event.event_type || "preflight")}</span>
          <strong>{event.payload.decision_id || "run preflight"}</strong>
          <small>{event.created_at || event.payload.artifact_path}</small>
        </button>
      ))}
    </div>
  );
}

function ReplayManifest({ manifest, onLoadArtifact }) {
  const artifacts = manifest?.artifacts || [];
  if (!artifacts.length) return <p className="muted">No replay artifacts are registered for this run.</p>;
  return (
    <div className="preflight-artifacts">
      {artifacts.map((artifact) => (
        <button
          className="preflight-artifact-row"
          disabled={!artifact.can_read_via_artifact_endpoint}
          key={artifact.artifact_id || artifact.path}
          onClick={() => onLoadArtifact(artifact.path)}
          type="button"
        >
          <span>{statusLabel(artifact.artifact_role || "artifact")}</span>
          <strong>{artifact.label || artifact.relative_path}</strong>
          <small>{artifact.created_at || artifact.relative_path}</small>
        </button>
      ))}
    </div>
  );
}

function WorkflowIcon({ active, label, title }) {
  return (
    <span className={active ? "workflow-dot active" : "workflow-dot"} title={title || label} aria-label={`${label}: ${active ? "ready" : "not ready"}`}>
      {label.slice(0, 1)}
    </span>
  );
}

const DEFAULT_LIBRARY_COLUMN_WIDTHS = {
  conversation: 420,
  workflow: 140,
  context: 260,
  updated: 150,
  media: 120
};

function LibraryTable({
  rows,
  allItems,
  searchStatus,
  usingApiRows,
  loadedCount,
  totalCount,
  canLoadMore,
  selectedId,
  onLoadMore,
  onOpenConversation,
  onSelect
}) {
  const [columnWidths, setColumnWidths] = useState(DEFAULT_LIBRARY_COLUMN_WIDTHS);
  const status = searchStatus?.status || "idle";
  const statusMessage = searchStatus?.message || "";
  const showLoading = status === "loading";
  const loadingMore = status === "loading_more";
  const showEmpty = status === "loaded" && usingApiRows && rows.length === 0;
  const showFallback = status === "error";
  function startColumnResize(column, event) {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startWidth = columnWidths[column];
    const handlePointerMove = (moveEvent) => {
      const nextWidth = Math.max(96, Math.min(720, startWidth + moveEvent.clientX - startX));
      setColumnWidths((widths) => ({ ...widths, [column]: nextWidth }));
    };
    const handlePointerUp = () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
  }

  function resizeColumnWithKeyboard(column, event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    setColumnWidths((widths) => ({
      ...widths,
      [column]: Math.max(96, Math.min(720, widths[column] + direction * 16))
    }));
  }

  function HeaderCell({ column, children }) {
    return (
      <th>
        <div className="resizable-column-heading">
          <span>{children}</span>
          <button
            aria-label={`Resize ${children} column`}
            className="column-resize-handle"
            onKeyDown={(event) => resizeColumnWithKeyboard(column, event)}
            onPointerDown={(event) => startColumnResize(column, event)}
            type="button"
          />
        </div>
      </th>
    );
  }

  return (
    <div className="table-shell">
      <table className="conversation-table">
        <colgroup>
          <col style={{ width: `${columnWidths.conversation}px` }} />
          <col style={{ width: `${columnWidths.workflow}px` }} />
          <col style={{ width: `${columnWidths.context}px` }} />
          <col style={{ width: `${columnWidths.updated}px` }} />
          <col style={{ width: `${columnWidths.media}px` }} />
        </colgroup>
        <thead>
          <tr>
            <HeaderCell column="conversation">Conversation</HeaderCell>
            <HeaderCell column="workflow">Workflow</HeaderCell>
            <HeaderCell column="context">Calendar / route</HeaderCell>
            <HeaderCell column="updated">Updated</HeaderCell>
            <HeaderCell column="media">Media</HeaderCell>
          </tr>
        </thead>
        <tbody>
          {showLoading ? (
            <tr className="table-state-row">
              <td colSpan={5}>
                <div className="library-table-state loading" role="status">
                  <span className="state-spinner" aria-hidden="true" />
                  <strong>Loading conversations</strong>
                  <p>{statusMessage}</p>
                </div>
              </td>
            </tr>
          ) : null}
          {showEmpty ? (
            <tr className="table-state-row">
              <td colSpan={5}>
                <div className="library-table-state empty" role="status">
                  <strong>No matching conversations</strong>
                  <p>{statusMessage || "Try a broader search or switch the artifact kind filter back to All artifacts."}</p>
                </div>
              </td>
            </tr>
          ) : null}
          {showFallback ? (
            <tr className="table-state-row">
              <td colSpan={5}>
                <div className="library-table-state warning" role="status">
                  <strong>Using fallback rows</strong>
                  <p>{statusMessage || "The conversation search API is unavailable, so the table is using local fixture or previously loaded rows."}</p>
                </div>
              </td>
            </tr>
          ) : null}
          {rows.map((row) => {
            const item = row.representative;
            const calendar =
              row.calendarLabel ||
              row.source?.metadata?.event?.summary ||
              item.metadata?.event?.summary ||
              item.metadata?.route?.label ||
              "No context yet";
            const sourceDocument = row.source || findSourceDocument(item, allItems);
            const linkedMedia = row.mediaBlob?.playback_url ? row.mediaBlob : mediaForItem(item, sourceDocument);
            return (
              <tr
                className={row.artifacts.some((artifact) => artifact.id === selectedId) ? "selected" : ""}
                key={row.key}
                onClick={() => onSelect(item.id)}
                onDoubleClick={() => {
                  onSelect(item.id);
                  onOpenConversation();
                }}
              >
                <td>
                  <strong>{row.title}</strong>
                  <small>{row.source?.title || row.source?.source_path || row.key}</small>
                </td>
                <td>
                  <div className="workflow-progress" aria-label="Workflow progress">
                    <WorkflowIcon active={row.hasTranscript} label="Transcript" title="Transcript stored" />
                    <WorkflowIcon active={row.hasSummary} label="Summary" title="First-pass summary stored" />
                    <WorkflowIcon active={row.hasContextualReadout} label="Context" title="Context-enriched readout stored" />
                  </div>
                </td>
                <td>{calendar}</td>
                <td>{formatDate(row.updatedAt || row.latestArtifact?.generated_at || row.latestArtifact?.updated_at)}</td>
                <td>
                  <button
                    className={linkedMedia ? "media-play-button" : "media-play-button disabled"}
                    disabled={!linkedMedia}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (!linkedMedia) return;
                      onSelect(item.id);
                      onOpenConversation();
                    }}
                    title={linkedMedia ? "Open the conversation player" : "No source recording is linked"}
                    type="button"
                  >
                    <svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z" />
                    </svg>
                    Play
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {usingApiRows && totalCount > 0 ? (
        <div className="library-pagination" aria-label="Library pagination">
          <span>{loadedCount} of {totalCount} conversations loaded</span>
          <button
            disabled={!canLoadMore || loadingMore}
            onClick={onLoadMore}
            type="button"
          >
            {loadingMore ? "Loading..." : canLoadMore ? "Load more" : "All loaded"}
          </button>
        </div>
      ) : null}
    </div>
  );
}

function ReviewQueue({ queue, reviewAction, batchManifests, onPrepareFirstPass, onSubmitFirstPass, onRefreshFirstPass, onSelectFirstPassManifest, humanReviewAction, onRecordHumanReview }) {
  const buckets = queue.buckets || [];
  const items = queue.items || [];
  const recentBatchManifests = batchManifests?.items || [];
  const batchPayload = reviewAction.payload || null;
  const batchCounts = batchPayload?.batch_counts || {};
  const batchCountEntries = Object.entries(batchCounts);
  const materializedCount = batchPayload?.materialized?.length || 0;
  const materializationErrorCount = batchPayload?.materialization_errors?.length || 0;
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
          {batchPayload && (
            <div className="batch-status-panel">
              <div>
                <span>Requests</span>
                <strong>{batchPayload.request_count || 0}</strong>
              </div>
              <div>
                <span>Batch</span>
                <strong>{batchPayload.batch_id || (batchPayload.dry_run ? "prepared only" : "pending")}</strong>
              </div>
              <div>
                <span>Status</span>
                <strong>{batchPayload.status || reviewAction.status}</strong>
              </div>
              <div>
                <span>Materialized</span>
                <strong>{materializedCount}</strong>
              </div>
              {batchCountEntries.length ? (
                <div className="batch-counts">
                  <span>Provider counts</span>
                  <p>{batchCountEntries.map(([key, value]) => `${statusLabel(key)} ${value}`).join(" · ")}</p>
                </div>
              ) : null}
              {materializationErrorCount ? (
                <div className="batch-counts warning">
                  <span>Materialization errors</span>
                  <p>{materializationErrorCount}</p>
                </div>
              ) : null}
            </div>
          )}
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
      {recentBatchManifests.length ? (
        <section className="saved-batch-panel">
          <div className="saved-batch-heading">
            <strong>Recent first-pass batches</strong>
            <span>{recentBatchManifests.length} of {batchManifests.total || recentBatchManifests.length}</span>
          </div>
          <div className="saved-batch-list">
            {recentBatchManifests.map((item) => (
              <button
                className={reviewAction.manifest === item.manifest ? "saved-batch-row active" : "saved-batch-row"}
                key={item.manifest}
                onClick={() => onSelectFirstPassManifest(item)}
                type="button"
              >
                <div>
                  <strong>{item.batch_id || (item.dry_run ? "prepared manifest" : "submitted manifest")}</strong>
                  <small>{item.manifest}</small>
                </div>
                <span>{item.status}</span>
                <small>{item.request_count || 0} requests · {item.materialized_count || 0} materialized</small>
              </button>
            ))}
          </div>
        </section>
      ) : null}
      {humanReviewAction.message && (
        <div className={`action-notice ${humanReviewAction.status}`}>
          <strong>{humanReviewAction.message}</strong>
          {humanReviewAction.payload && <code>{JSON.stringify({
            run_id: humanReviewAction.payload.run_id,
            decision_id: humanReviewAction.payload.decision_id,
            review_action: humanReviewAction.payload.review_action,
            human_review_status: humanReviewAction.payload.human_review_status,
            will_execute_external_action: humanReviewAction.payload.will_execute_external_action
          }, null, 2)}</code>}
        </div>
      )}
      <div className="queue-list">
        <div className="queue-list-heading">
          <h2>Review items</h2>
          <span>{items.length} loaded</span>
        </div>
        {items.length ? (
          items.map((item) => (
            <article className={`queue-row ${item.status}`} key={item.id}>
              <div>
                <strong>{item.label}</strong>
                <small>{item.reason}</small>
              </div>
              <span>{item.type === "app_intelligence_human_review" ? statusLabel(item.decision_status || item.status) : item.route_decision_exists ? "route available" : "stale route reference"}</span>
              <code>{item.artifact_path || item.route_decision_path || item.review_path}</code>
              {item.type === "app_intelligence_human_review" && (
                <div className="notice-actions">
                  <button onClick={() => onRecordHumanReview(item, "annotate")} type="button">Annotate</button>
                  {item.status === "needs_human_review" && (
                    <button onClick={() => onRecordHumanReview(item, "resolve")} type="button">Resolve</button>
                  )}
                  {item.human_review_status === "resolved" && (
                    <button onClick={() => onRecordHumanReview(item, "reopen")} type="button">Reopen</button>
                  )}
                </div>
              )}
            </article>
          ))
        ) : (
          <p className="muted">No route review files are currently loaded.</p>
        )}
      </div>
    </>
  );
}

function Inspector({
  item,
  items,
  activeNav,
  documentDetail,
  documentDetailAction,
  relatedDocuments,
  onOpenConversation,
  onSelectDocument,
  reviewQueue,
  selectedTask,
  selectedTaskConfig,
  selectedProvider,
  configAction,
  selectedRunId,
  selectedRunDetail,
  runReplayManifest,
  runDetailAction,
  sessionPreflight,
  onRunSessionPreflight,
  sessionStartAction,
  onStartAppServerSession,
  modelTurnAction,
  onPrepareModelTurnPacket,
  selectedPacketId,
  onSelectPacket,
  packetReview,
  sendPreflight,
  onRunModelTurnSendPreflight,
  sendAction,
  onSendModelTurn,
  turnStatusAction,
  onCaptureTurnStatus,
  decisionValidation,
  onValidateStructuredDecision,
  decisionApply,
  onApplyStructuredDecision,
  forkPreflightAction,
  onRunForkPreflight,
  rollbackPreflightAction,
  onRunRollbackPreflight,
  runArtifactAction,
  onLoadRunArtifact,
  intelligence
}) {
  if (activeNav === "Intelligence") {
    const preview = configAction.preview;
    const run = selectedRunDetail?.run || null;
    const events = selectedRunDetail?.events || [];
    const policy = run?.policy || {};
    const approvalPolicy = policy.approval_policy || {};
    const evalPolicy = policy.eval_policy || {};
    const decisions = run?.decisions || [];
    const latestDecision = decisions.length ? decisions[decisions.length - 1] : null;
    const latestDecisionCanApply =
      latestDecision?.status === "validated" &&
      (latestDecision.action === "continue_current_branch" || latestDecision.action === "stop" || latestDecision.action === "ask_for_human_review");
    const latestDecisionCanForkPreflight =
      latestDecision?.status === "validated" && latestDecision.action === "fork_branches";
    const latestDecisionCanRollbackPreflight =
      latestDecision?.status === "validated" && latestDecision.action === "rollback";
    return (
      <div className="inspector-content">
        <p className="eyebrow">Intelligence Inspector</p>
        <h2>{statusLabel(selectedTask || "task")}</h2>
        <dl>
          <dt>Provider</dt>
          <dd>{selectedTaskConfig?.provider || "Unknown"}</dd>
          <dt>Model</dt>
          <dd>{selectedTaskConfig?.model || "Provider default"}</dd>
          <dt>Config path</dt>
          <dd>{intelligence.config?.config_path || "Unavailable"}</dd>
          <dt>Provider status</dt>
          <dd>{selectedProvider?.status || "No provider status"}</dd>
          <dt>Requires ledger</dt>
          <dd>{selectedTaskConfig?.requires_ledger ? "yes" : "no"}</dd>
        </dl>
        {preview ? (
          <div className="preview-card">
            <span>Preview</span>
            <strong>{preview.will_write ? "Apply response" : "No write preview"}</strong>
            <p>{preview.resolved_before?.provider} → {preview.resolved_after?.provider}</p>
            <code>{JSON.stringify(preview.rollback || {}, null, 2)}</code>
          </div>
        ) : (
          <p className="muted">Preview a task edit to see rollback metadata here.</p>
        )}
        <div className="run-detail-card">
          <span>Selected Run</span>
          {run ? (
            <>
              <strong>{run.workflow || run.run_id}</strong>
              <small>{run.run_id}</small>
              <dl>
                <dt>Phase</dt>
                <dd>{run.phase || run.status}</dd>
                <dt>Provider</dt>
                <dd>{run.provider || "Unknown"}</dd>
                <dt>Document</dt>
                <dd>{run.document_id || "None linked"}</dd>
                <dt>Ledger path</dt>
                <dd>{selectedRunDetail.path || "Unavailable"}</dd>
                <dt>Allowed actions</dt>
                <dd>{(policy.allowed_actions || []).join(", ") || "None"}</dd>
                <dt>Remote transport</dt>
                <dd>{policy.remote_transport || "Unspecified"}</dd>
              </dl>
              <div className="approval-gate">
                <strong>Next gate: start app-server session</strong>
                <p>Not enabled in this UI. Starting a session must be added as a separate reviewed action with an explicit approval token and ledger event.</p>
                <code>{JSON.stringify({ approval_policy: approvalPolicy, eval_policy: evalPolicy }, null, 2)}</code>
                <div className="notice-actions">
                  <button onClick={() => onRunSessionPreflight({ appendEvent: false })} disabled={sessionPreflight.status === "running"} type="button">
                    Dry-run preflight
                  </button>
                  <button onClick={() => onRunSessionPreflight({ appendEvent: true })} disabled={sessionPreflight.status === "recording"} type="button">
                    Record preflight event
                  </button>
                  <button onClick={onStartAppServerSession} disabled={sessionStartAction.status === "starting" || run.phase !== "prepared"} type="button">
                    Start control plane
                  </button>
                  <button onClick={onPrepareModelTurnPacket} disabled={modelTurnAction.status === "preparing" || run.phase !== "session_started"} type="button">
                    Prepare prompt packet
                  </button>
                </div>
                {sessionPreflight.message && (
                  <div className={`action-notice ${sessionPreflight.status}`}>
                    <strong>{sessionPreflight.message}</strong>
                    {sessionPreflight.payload && <code>{JSON.stringify(sessionPreflight.payload.checks || {}, null, 2)}</code>}
                  </div>
                )}
                {sessionStartAction.message && (
                  <div className={`action-notice ${sessionStartAction.status}`}>
                    <strong>{sessionStartAction.message}</strong>
                    {sessionStartAction.payload && <code>{JSON.stringify({ transport: sessionStartAction.payload.transport, will_start_model_turn: sessionStartAction.payload.will_start_model_turn }, null, 2)}</code>}
                  </div>
                )}
                {modelTurnAction.message && (
                  <div className={`action-notice ${modelTurnAction.status}`}>
                    <strong>{modelTurnAction.message}</strong>
                    {modelTurnAction.payload && <code>{JSON.stringify({ packet_path: modelTurnAction.payload.packet_path, will_send_prompt: modelTurnAction.payload.will_send_prompt }, null, 2)}</code>}
                  </div>
                )}
              </div>
              {run.prompt_packets?.length ? (
                <div className="event-list">
                  <span>Prompt Packets</span>
                  {run.prompt_packets.slice(-3).map((packet) => (
                    <article className={selectedPacketId === packet.packet_id ? "active" : ""} key={packet.packet_id} onClick={() => onSelectPacket(packet.packet_id)}>
                      <strong>{packet.task}</strong>
                      <small>{packet.packet_path}</small>
                      <div className="notice-actions">
                        {packet.packet_path ? (
                          <button
                            onClick={(event) => {
                              event.stopPropagation();
                              onLoadRunArtifact(packet.packet_path);
                            }}
                            type="button"
                          >
                            Open packet JSON
                          </button>
                        ) : null}
                        {packet.prompt_path ? (
                          <button
                            onClick={(event) => {
                              event.stopPropagation();
                              onLoadRunArtifact(packet.prompt_path);
                            }}
                            type="button"
                          >
                            Open prompt text
                          </button>
                        ) : null}
                      </div>
                    </article>
                  ))}
                </div>
              ) : null}
              <div className="event-list decision-history-card">
                <span>Decision History</span>
                <DecisionHistory decisions={decisions} onLoadArtifact={onLoadRunArtifact} />
                <div className="preflight-picker">
                  <span>Replay Manifest</span>
                  <ReplayManifest manifest={runReplayManifest} onLoadArtifact={onLoadRunArtifact} />
                </div>
                <div className="preflight-picker">
                  <span>Preflight Artifacts</span>
                  <PreflightArtifacts events={events} onLoadArtifact={onLoadRunArtifact} />
                </div>
                {runArtifactAction.message ? (
                  <div className={`action-notice ${runArtifactAction.status}`}>
                    <strong>{runArtifactAction.message}</strong>
                    {runArtifactAction.payload && <code>{JSON.stringify({
                      relative_path: runArtifactAction.payload.relative_path,
                      artifact_type: runArtifactAction.payload.artifact_type,
                      bytes: runArtifactAction.payload.bytes,
                      will_execute_write_bearing_action: runArtifactAction.payload.will_execute_write_bearing_action
                    }, null, 2)}</code>}
                    {runArtifactAction.payload && (
                      <pre className="prompt-preview">
                        {runArtifactAction.payload.artifact_type === "json"
                          ? JSON.stringify(runArtifactAction.payload.json, null, 2)
                          : runArtifactAction.payload.text}
                      </pre>
                    )}
                  </div>
                ) : null}
              </div>
              {packetReview.payload ? (
                <div className="preview-card">
                  <span>Packet Review</span>
                  <strong>{packetReview.payload.packet_id}</strong>
                  <p>Future send token: {packetReview.payload.future_required_approval_token_for_send}</p>
                  <code>{JSON.stringify({
                    packet_path: packetReview.payload.packet_path,
                    prompt_path: packetReview.payload.prompt_path,
                    will_send_prompt: packetReview.payload.will_send_prompt
                  }, null, 2)}</code>
                  <button
                    className="gate-button"
                    disabled={sendPreflight.status === "running"}
                    onClick={onRunModelTurnSendPreflight}
                    type="button"
                  >
                    Dry-run send preflight
                  </button>
                  {sendPreflight.message ? (
                    <div className={`action-notice ${sendPreflight.status}`}>
                      <strong>{sendPreflight.message}</strong>
                      {sendPreflight.payload && <code>{JSON.stringify({
                        checks: sendPreflight.payload.checks,
                        will_send_prompt: sendPreflight.payload.will_send_prompt,
                        will_write_event: sendPreflight.payload.will_write_event,
                        prompt_char_count: sendPreflight.payload.prompt_char_count
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  <button
                    className="gate-button danger-gate"
                    disabled={sendAction.status === "sending" || packetReview.payload.packet?.will_send_prompt === true}
                    onClick={onSendModelTurn}
                    type="button"
                  >
                    Send reviewed packet
                  </button>
                  {sendAction.message ? (
                    <div className={`action-notice ${sendAction.status}`}>
                      <strong>{sendAction.message}</strong>
                      {sendAction.payload && <code>{JSON.stringify({
                        codex_thread_id: sendAction.payload.codex_thread_id,
                        codex_turn_id: sendAction.payload.codex_turn_id,
                        captured_event_count: sendAction.payload.captured_event_count,
                        will_execute_downstream_action: sendAction.payload.will_execute_downstream_action
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  <button
                    className="gate-button"
                    disabled={turnStatusAction.status === "capturing" || !run.state?.latest_turn_id}
                    onClick={onCaptureTurnStatus}
                    type="button"
                  >
                    Capture turn status
                  </button>
                  {turnStatusAction.message ? (
                    <div className={`action-notice ${turnStatusAction.status}`}>
                      <strong>{turnStatusAction.message}</strong>
                      {turnStatusAction.payload && <code>{JSON.stringify({
                        status: turnStatusAction.payload.status,
                        completed: turnStatusAction.payload.completed,
                        artifact_path: turnStatusAction.payload.artifact_path,
                        will_execute_structured_decision: turnStatusAction.payload.will_execute_structured_decision
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  {run.latest_model_turn_status ? (
                    <div className="action-notice ok">
                      <strong>Latest turn status: {run.latest_model_turn_status.status || "unknown"}</strong>
                      {run.latest_model_turn_status.artifact_path ? (
                        <div className="notice-actions">
                          <button onClick={() => onLoadRunArtifact(run.latest_model_turn_status.artifact_path)} type="button">
                            Open status JSON
                          </button>
                        </div>
                      ) : null}
                      <code>{JSON.stringify(run.latest_model_turn_status, null, 2)}</code>
                    </div>
                  ) : null}
                  <button
                    className="gate-button"
                    disabled={decisionValidation.status === "validating" || !run.latest_model_turn_status?.artifact_path}
                    onClick={onValidateStructuredDecision}
                    type="button"
                  >
                    Validate structured decision
                  </button>
                  {decisionValidation.message ? (
                    <div className={`action-notice ${decisionValidation.status}`}>
                      <strong>{decisionValidation.message}</strong>
                      {decisionValidation.payload && <code>{JSON.stringify({
                        valid: decisionValidation.payload.valid,
                        action: decisionValidation.payload.decision?.action || "",
                        errors: decisionValidation.payload.errors || [],
                        will_execute_host_action: decisionValidation.payload.will_execute_host_action
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  {latestDecision ? (
                    <div className="action-notice ok">
                      <strong>Latest structured decision: {latestDecision.action || "unknown"} ({latestDecision.status})</strong>
                      <code>{JSON.stringify({
                        decision_id: latestDecision.decision_id,
                        action: latestDecision.action,
                        status: latestDecision.status,
                        will_execute_host_action: latestDecision.will_execute_host_action,
                        apply_result: latestDecision.apply_result || null
                      }, null, 2)}</code>
                    </div>
                  ) : null}
                  <button
                    className="gate-button"
                    disabled={decisionApply.status === "applying" || !latestDecisionCanApply}
                    onClick={() => onApplyStructuredDecision(latestDecision?.decision_id)}
                    type="button"
                  >
                    Apply ledger-only decision
                  </button>
                  {decisionApply.message ? (
                    <div className={`action-notice ${decisionApply.status}`}>
                      <strong>{decisionApply.message}</strong>
                      {decisionApply.payload && <code>{JSON.stringify({
                        decision_id: decisionApply.payload.decision_id,
                        action: decisionApply.payload.decision_action,
                        applied_ledger_state: decisionApply.payload.applied_ledger_state,
                        will_execute_external_action: decisionApply.payload.will_execute_external_action,
                        will_execute_write_bearing_action: decisionApply.payload.will_execute_write_bearing_action,
                        will_fork_or_rollback: decisionApply.payload.will_fork_or_rollback
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  <button
                    className="gate-button"
                    disabled={forkPreflightAction.status === "running" || !latestDecisionCanForkPreflight}
                    onClick={() => onRunForkPreflight(latestDecision?.decision_id)}
                    type="button"
                  >
                    Preview fork plan
                  </button>
                  {forkPreflightAction.message ? (
                    <div className={`action-notice ${forkPreflightAction.status}`}>
                      <strong>{forkPreflightAction.message}</strong>
                      {forkPreflightAction.payload && <code>{JSON.stringify({
                        decision_id: forkPreflightAction.payload.decision_id,
                        planned_branch_count: forkPreflightAction.payload.planned_branches?.length || 0,
                        will_create_thread: forkPreflightAction.payload.will_create_thread,
                        will_modify_branches: forkPreflightAction.payload.will_modify_branches,
                        will_run_provider: forkPreflightAction.payload.will_run_provider
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  <button
                    className="gate-button"
                    disabled={rollbackPreflightAction.status === "running" || !latestDecisionCanRollbackPreflight}
                    onClick={() => onRunRollbackPreflight(latestDecision?.decision_id)}
                    type="button"
                  >
                    Preview rollback plan
                  </button>
                  {rollbackPreflightAction.message ? (
                    <div className={`action-notice ${rollbackPreflightAction.status}`}>
                      <strong>{rollbackPreflightAction.message}</strong>
                      {rollbackPreflightAction.payload && <code>{JSON.stringify({
                        decision_id: rollbackPreflightAction.payload.decision_id,
                        current_branch: rollbackPreflightAction.payload.current_branch,
                        target_branch: rollbackPreflightAction.payload.target_branch,
                        target_event_id: rollbackPreflightAction.payload.target_event_id,
                        target_turn_id: rollbackPreflightAction.payload.target_turn_id,
                        warning_count: rollbackPreflightAction.payload.warnings?.length || 0,
                        will_modify_branches: rollbackPreflightAction.payload.will_modify_branches,
                        will_revert_artifacts: rollbackPreflightAction.payload.will_revert_artifacts,
                        will_run_provider: rollbackPreflightAction.payload.will_run_provider
                      }, null, 2)}</code>}
                    </div>
                  ) : null}
                  <pre className="prompt-preview">{packetReview.payload.prompt_text}</pre>
                </div>
              ) : packetReview.message ? (
                <p className="muted">{packetReview.message}</p>
              ) : null}
              <div className="event-list">
                <span>Recent Events</span>
                {events.length ? events.map((event) => (
                  <article key={event.event_id || `${event.event_type}-${event.created_at}`}>
                    <strong>{event.event_type}</strong>
                    <small>{event.created_at}</small>
                  </article>
                )) : <p className="muted">No events recorded.</p>}
              </div>
            </>
          ) : (
            <p className="muted">{runDetailAction.message || (selectedRunId ? "Run detail is unavailable." : "Select or prepare a run ledger to inspect policy and events.")}</p>
          )}
        </div>
      </div>
    );
  }
  if (activeNav === "Review Queue") {
    const routeBucket = (reviewQueue.buckets || []).find((bucket) => bucket.id === "route_reviews");
    const appReviewBucket = (reviewQueue.buckets || []).find((bucket) => bucket.id === "app_intelligence_human_review");
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
          <dt>App Intelligence review</dt>
          <dd>{appReviewBucket ? `${appReviewBucket.count} pending; ${appReviewBucket.pending_apply_count || 0} still need ledger-only apply.` : "No App Intelligence review summary."}</dd>
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
  const sourceDocument = relatedSourceDocument(relatedDocuments) || findSourceDocument(item, items);
  const linkedMedia = mediaForItem(item, sourceDocument);
  const summaryText = documentSummaryText(documentDetail);
  const payload = documentDetail?.json_payload || {};
  const metadata = item.metadata || {};
  const participants = Array.isArray(payload.participants) ? payload.participants : [];
  const topics = Array.isArray(payload.topics) ? payload.topics : [];
  const actionItems = Array.isArray(payload.action_items) ? payload.action_items : [];
  const risks = Array.isArray(payload.risks) ? payload.risks : [];
  return (
    <div className="inspector-content">
      <p className="eyebrow">Inspector</p>
      <h2>{item.title || "Untitled artifact"}</h2>
      <dl>
        <dt>Selected</dt>
        <dd>{statusLabel(item.kind || "unknown")}</dd>
        <dt>Meeting</dt>
        <dd>{metadata.event?.summary || payload.event?.summary || "No calendar title"}</dd>
        <dt>When</dt>
        <dd>{formatDate(item.generated_at || item.updated_at)}</dd>
        <dt>Source</dt>
        <dd>{item.source_path || "Unknown"}</dd>
        <dt>Audio</dt>
        <dd>
          {item.media_blob?.id || (sourceDocument?.media_blob?.id ? `Inherited from source transcript ${sourceDocument.id}` : "No media blob linked")}
        </dd>
      </dl>
      <button className="conversation-launch" onClick={onOpenConversation} type="button">
        Open conversation workspace
      </button>
      {linkedMedia ? (
        <div className="player-card">
          <span>{item.media_blob?.playback_url ? "Source recording" : "Source transcript recording"}</span>
          <audio controls src={linkedMedia.playback_url} />
          <a href={linkedMedia.download_url}>Download source recording</a>
          {sourceDocument && sourceDocument.id !== item.id ? (
            <button className="inline-action" onClick={() => onSelectDocument(sourceDocument.id)} type="button">
              Open source transcript
            </button>
          ) : null}
        </div>
      ) : (
        <div className="media-diagnostic-card">
          <span>No linked audio</span>
          <p>
            {sourceArtifactPath(item)
              ? "This artifact points to a source transcript, but the current library page does not include a matching transcript with a stored blob."
              : "This artifact has no direct source recording link yet. Run media backfill or select the source transcript when available."}
          </p>
        </div>
      )}
      <section className="readout-card" aria-label="Conversation summary">
        <span>Conversation summary</span>
        {documentDetailAction.status === "loading" ? (
          <p className="muted">Loading summary...</p>
        ) : summaryText ? (
          <p>{summaryText.length > 900 ? `${summaryText.slice(0, 900)}...` : summaryText}</p>
        ) : (
          <p className="muted">{documentDetailAction.message || "No first-pass summary is stored for this conversation yet."}</p>
        )}
        {participants.length ? (
          <div className="mini-section">
            <strong>People</strong>
            <div className="chip-cloud">
              {participants.slice(0, 8).map((participant) => (
                <span key={displayLabel(participant, "Participant")}>{displayLabel(participant, "Participant")}</span>
              ))}
            </div>
          </div>
        ) : null}
        {topics.length ? (
          <div className="mini-section">
            <strong>Topics</strong>
            <div className="chip-cloud">
              {topics.slice(0, 8).map((topic) => <span key={displayLabel(topic, "Topic")}>{displayLabel(topic, "Topic")}</span>)}
            </div>
          </div>
        ) : null}
        {actionItems.length || risks.length ? (
          <div className="readout-columns">
            {actionItems.length ? (
              <div>
                <strong>Actions</strong>
                <ul>
                  {actionItems.slice(0, 4).map((action, index) => (
                    <li key={`${index}-${displayLabel(action, "Action item")}`}>{displayLabel(action, "Action item")}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {risks.length ? (
              <div>
                <strong>Risks</strong>
                <ul>
                  {risks.slice(0, 4).map((risk, index) => (
                    <li key={`${index}-${displayLabel(risk, "Risk")}`}>{displayLabel(risk, "Risk")}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
      <div className="action-stack">
        <a className="developer-link" href={`/api/documents/${encodeURIComponent(item.id)}/context?context_chunks=2`} rel="noreferrer" target="_blank">
          Developer: raw context JSON
        </a>
        <button disabled title="Share-link workflow is planned but not wired yet." type="button">Prepare share link (planned)</button>
        <button disabled title="Speaker/contact review is planned but not wired yet." type="button">Review speakers (planned)</button>
      </div>
    </div>
  );
}

function ConversationWorkflowModal({
  conversationDetail,
  conversationDetailAction,
  documentDetail,
  documentDetailAction,
  relatedDocuments,
  item,
  items,
  onClose,
  onSelectDocument
}) {
  const [retranscriptionBackend, setRetranscriptionBackend] = useState("faster_whisper");
  const [retranscriptionPreflight, setRetranscriptionPreflight] = useState({ status: "idle", message: "", payload: null });
  const [retranscriptionQueue, setRetranscriptionQueue] = useState({ status: "idle", message: "", payload: null });
  const [activeWorkflowView, setActiveWorkflowView] = useState("transcript");
  const [sourceDetail, setSourceDetail] = useState(null);
  const [sourceDetailAction, setSourceDetailAction] = useState({ status: "idle", message: "" });
  const selectedDetail = conversationDetail?.selected_document || documentDetail;
  const sourceDocument = conversationDetail?.transcript_document || relatedSourceDocument(relatedDocuments) || findSourceDocument(item, items);
  const summaryDetail = conversationDetail?.summary_document || (item.kind === "readout" ? selectedDetail : null);
  const contextualDetail = conversationDetail?.contextual_readout_document || (item.kind === "contextual_readout" ? selectedDetail : null);
  const linkedMedia = conversationDetail?.media_blob?.playback_url ? conversationDetail.media_blob : mediaForItem(item, sourceDocument);
  const readoutDetail = contextualDetail || summaryDetail || selectedDetail;
  const payload = readoutDetail?.json_payload || {};
  const transcriptDetail = conversationDetail?.transcript_document || (item.kind === "transcript" ? selectedDetail : sourceDetail);
  const turns = transcriptTurns(transcriptDetail);
  const meta = transcriptMeta(transcriptDetail);
  const summaryText = documentSummaryText(summaryDetail || selectedDetail);
  const finalReadoutText = documentSummaryText(contextualDetail || selectedDetail, { allowTranscriptFallback: true });
  const participants = Array.isArray(conversationDetail?.participants) && conversationDetail.participants.length
    ? conversationDetail.participants
    : Array.isArray(payload.participants) ? payload.participants : [];
  const topics = Array.isArray(payload.topics) ? payload.topics : [];
  const actionItems = Array.isArray(payload.action_items) ? payload.action_items : [];
  const risks = Array.isArray(payload.risks) ? payload.risks : [];
  const finalReadoutReady = Boolean(contextualDetail) || item.kind === "contextual_readout" || Boolean(payload.contextualization?.status);
  const workflowViews = [
    { id: "transcript", label: "Transcript" },
    { id: "summary", label: "First-pass summary" },
    { id: "context", label: "Context workbench" },
    { id: "speakers", label: "Speakers" },
    { id: "output", label: "Final readout" }
  ];

  useEffect(() => {
    let cancelled = false;
    async function loadSourceDetail() {
      if (conversationDetail?.transcript_document || !sourceDocument?.id || sourceDocument.id === item.id || item.kind === "transcript") {
        setSourceDetail(null);
        setSourceDetailAction({ status: "idle", message: "" });
        return;
      }
      setSourceDetailAction({ status: "loading", message: "Loading source transcript..." });
      try {
        const payload = await fetchJson(`/api/documents/${encodeURIComponent(sourceDocument.id)}`);
        if (cancelled) return;
        setSourceDetail(payload);
        setSourceDetailAction({ status: "loaded", message: "" });
      } catch (error) {
        if (cancelled) return;
        setSourceDetail(null);
        setSourceDetailAction({ status: "error", message: `Source transcript failed: ${error.message}` });
      }
    }
    loadSourceDetail();
    return () => {
      cancelled = true;
    };
  }, [conversationDetail?.transcript_document, item.id, item.kind, sourceDocument?.id]);
  async function previewRetranscription() {
    setRetranscriptionPreflight({ status: "running", message: "Previewing retranscription plan...", payload: null });
    setRetranscriptionQueue({ status: "idle", message: "", payload: null });
    try {
      const payload = await postJson(
        `/api/documents/${encodeURIComponent(item.id)}/retranscription/preflight`,
        { backend: retranscriptionBackend }
      );
      setRetranscriptionPreflight({
        status: payload.ok ? "ok" : "blocked",
        message: payload.ok
          ? "Preflight ready; no transcription was queued."
          : `Preflight blocked: ${(payload.blocking_checks || []).join(", ") || "source unavailable"}.`,
        payload
      });
    } catch (error) {
      setRetranscriptionPreflight({
        status: "error",
        message: `Preflight failed: ${error.message}`,
        payload: null
      });
    }
  }

  async function queueRetranscription() {
    setRetranscriptionQueue({ status: "running", message: "Writing re-transcription job manifest...", payload: null });
    try {
      const payload = await postJson(
        `/api/documents/${encodeURIComponent(item.id)}/retranscription/queue`,
        {
          backend: retranscriptionBackend,
          approval_token: retranscriptionPreflight.payload?.future_required_approval_token_for_queue || "QUEUE_RETRANSCRIPTION_JOB"
        }
      );
      setRetranscriptionQueue({
        status: payload.ok ? "ok" : "blocked",
        message: payload.ok
          ? "Re-transcription job manifest queued; no backend was started."
          : `Queue blocked: ${(payload.blocking_checks || []).join(", ") || "source unavailable"}.`,
        payload
      });
    } catch (error) {
      setRetranscriptionQueue({
        status: "error",
        message: `Queue failed: ${error.message}`,
        payload: null
      });
    }
  }

  return (
    <div className="conversation-modal-backdrop full-viewport" onMouseDown={onClose}>
      <section
        aria-labelledby="conversation-modal-title"
        aria-modal="true"
        className="conversation-modal conversation-workspace"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header className="conversation-modal-header">
          <div>
            <p className="eyebrow">Conversation Workspace</p>
            <h2 id="conversation-modal-title">{conversationDetail?.conversation?.title || item.title || sourceDocument?.title || "Untitled conversation"}</h2>
            <p className="muted">
              {conversationDetailAction.message || meta.event || statusLabel(item.kind || "artifact")} · {formatDate(item.generated_at || item.updated_at)}
            </p>
          </div>
          <button aria-label="Close conversation workspace" className="modal-close" onClick={onClose} type="button">
            <svg aria-hidden="true" focusable="false" viewBox="0 0 24 24">
              <line x1="7" y1="7" x2="17" y2="17" />
              <line x1="17" y1="7" x2="7" y2="17" />
            </svg>
          </button>
        </header>

        <div className="conversation-workspace-body">
          <aside className="conversation-rail">
            <section className="rail-card">
              <span>Source recording</span>
              {linkedMedia ? (
                <>
                  <audio controls src={linkedMedia.playback_url} />
                  <a href={linkedMedia.download_url}>Download source recording</a>
                  {sourceDocument && sourceDocument.id !== item.id ? (
                    <button onClick={() => onSelectDocument(sourceDocument.id)} type="button">
                      Select source transcript
                    </button>
                  ) : null}
                </>
              ) : (
                <p className="muted">No source recording is linked yet.</p>
              )}
            </section>
            <section className="rail-card">
              <span>Metadata</span>
              <dl>
                <dt>Transcript</dt>
                <dd>{sourceDocument?.id || (item.kind === "transcript" ? item.id : "Not linked")}</dd>
                <dt>Summary</dt>
                <dd>{summaryDetail?.id || "Not prepared"}</dd>
                <dt>Final readout</dt>
                <dd>{contextualDetail?.id || "Not generated"}</dd>
                <dt>Turns</dt>
                <dd>{meta.utteranceCount || turns.length || "Unknown"}</dd>
                <dt>Duration</dt>
                <dd>{meta.duration || "Unknown"}</dd>
                <dt>Backend</dt>
                <dd>{meta.backend || "Unknown"}</dd>
              </dl>
            </section>
            <section className="rail-card">
              <span>Re-transcription</span>
              <label className="workflow-field">
                Backend
                <select
                  value={retranscriptionBackend}
                  onChange={(event) => {
                    setRetranscriptionBackend(event.target.value);
                    setRetranscriptionPreflight({ status: "idle", message: "", payload: null });
                    setRetranscriptionQueue({ status: "idle", message: "", payload: null });
                  }}
                >
                  <option value="faster_whisper">faster-whisper local</option>
                  <option value="assemblyai">AssemblyAI</option>
                </select>
              </label>
              <div className="workflow-action-row">
                <button onClick={previewRetranscription} disabled={retranscriptionPreflight.status === "running"} type="button">
                  Preview
                </button>
                <button
                  disabled={!retranscriptionPreflight.payload?.ok || retranscriptionQueue.status === "running"}
                  onClick={queueRetranscription}
                  type="button"
                >
                  Queue manifest
                </button>
              </div>
              {[retranscriptionPreflight, retranscriptionQueue].map((action, index) => (
                action.message ? (
                  <div className={`action-notice ${action.status}`} key={`${index}-${action.status}`}>
                    <strong>{action.message}</strong>
                    {action.payload?.job ? <small>{action.payload.job.path}</small> : null}
                  </div>
                ) : null
              ))}
            </section>
          </aside>

          <main className="conversation-main">
            <nav className="workflow-view-tabs" aria-label="Conversation workflow views">
              {workflowViews.map((view) => (
                <button
                  aria-pressed={activeWorkflowView === view.id}
                  className={activeWorkflowView === view.id ? "active" : ""}
                  key={view.id}
                  onClick={() => setActiveWorkflowView(view.id)}
                  type="button"
                >
                  {view.label}
                </button>
              ))}
            </nav>

            {activeWorkflowView === "transcript" ? (
              <section className="workflow-view transcript-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Transcript</span>
                    <h3>{transcriptDetail?.title || sourceDocument?.title || item.title}</h3>
                  </div>
                  <strong>{turns.length} turns</strong>
                </div>
                {sourceDetailAction.status === "loading" ? (
                  <p className="muted">Loading source transcript...</p>
                ) : turns.length ? (
                  <div className="transcript-frame" tabIndex={0}>
                    {turns.map((turn, index) => (
                      <article className={`transcript-turn ${speakerClassName(turn.speaker)}`} key={`${index}-${turn.time}-${turn.speaker}`}>
                        <div className="speaker-badge">
                          <strong>{turn.speaker}</strong>
                          {turn.time ? <small>{turn.time}</small> : null}
                        </div>
                        <p>{turn.text}</p>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted">{sourceDetailAction.message || "No transcript text is available for this conversation."}</p>
                )}
              </section>
            ) : null}

            {activeWorkflowView === "summary" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>First-pass summary</span>
                    <h3>{summaryText ? "Summary ready" : "Summary not prepared"}</h3>
                  </div>
                </div>
                {conversationDetailAction.status === "loading" || documentDetailAction.status === "loading" ? (
                  <p className="muted">Loading summary...</p>
                ) : summaryText ? (
                  <p className="summary-prose">{summaryText}</p>
                ) : (
                  <p className="muted">{documentDetailAction.message || "No first-pass summary is stored yet."}</p>
                )}
                {topics.length ? (
                  <div className="chip-cloud">{topics.slice(0, 16).map((topic) => <span key={displayLabel(topic, "Topic")}>{displayLabel(topic, "Topic")}</span>)}</div>
                ) : null}
              </section>
            ) : null}

            {activeWorkflowView === "context" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Context workbench</span>
                    <h3>Provenance and context gathering</h3>
                  </div>
                </div>
                <p>Gather calendar, GWS, msgcli, Odollo, Graphiti, local-store, and matter-routing evidence before the final readout.</p>
                {conversationDetail?.conversation?.artifacts?.length ? (
                  <div className="event-list">
                    <span>Conversation artifacts</span>
                    {conversationDetail.conversation.artifacts.map((artifact) => (
                      <article key={artifact.id}>
                        <strong>{artifact.title || artifact.id}</strong>
                        <small>{statusLabel(artifact.kind || "artifact")} · {formatDate(artifact.generated_at || artifact.updated_at)}</small>
                      </article>
                    ))}
                  </div>
                ) : null}
                <div className="workflow-action-row">
                  <a href={`/api/documents/${encodeURIComponent(item.id)}/context?context_chunks=2`} rel="noreferrer" target="_blank">Open raw context JSON</a>
                  <button disabled title="Context-run creation needs a reviewed backend contract." type="button">Start context run (planned)</button>
                  <button disabled title="Recurring-meeting recipes are planned for deterministic context acquisition." type="button">Apply recurring recipe (planned)</button>
                </div>
              </section>
            ) : null}

            {activeWorkflowView === "speakers" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Speakers and contacts</span>
                    <h3>Identity resolution</h3>
                  </div>
                </div>
                {participants.length ? (
                  <div className="identity-list">
                    {participants.map((participant, index) => (
                      <article key={`${index}-${displayLabel(participant, "Participant")}`}>
                        <strong>{displayLabel(participant, "Participant")}</strong>
                        <small>{participant.role || "Contact linking is planned; retain as extracted participant for now."}</small>
                        <button disabled title="Contact DB and merge workflow are planned in P09." type="button">Link contact (planned)</button>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No participants were extracted for this artifact yet.</p>
                )}
              </section>
            ) : null}

            {activeWorkflowView === "output" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Final readout</span>
                    <h3>{finalReadoutReady ? "Context-enriched readout" : "Context-enriched readout not generated yet"}</h3>
                  </div>
                </div>
                {finalReadoutReady && finalReadoutText ? <p className="summary-prose">{finalReadoutText}</p> : null}
                <div className="readout-columns">
                  {actionItems.length ? (
                    <div>
                      <strong>Actions</strong>
                      <ul>{actionItems.slice(0, 8).map((action, index) => <li key={`${index}-${displayLabel(action, "Action item")}`}>{displayLabel(action, "Action item")}</li>)}</ul>
                    </div>
                  ) : null}
                  {risks.length ? (
                    <div>
                      <strong>Risks</strong>
                      <ul>{risks.slice(0, 8).map((risk, index) => <li key={`${index}-${displayLabel(risk, "Risk")}`}>{displayLabel(risk, "Risk")}</li>)}</ul>
                    </div>
                  ) : null}
                </div>
                <div className="workflow-action-row">
                  <button disabled title="Final readout generation needs context-run output and a reviewed provider action." type="button">Generate final readout (planned)</button>
                  <button disabled title="Share links are planned after scoped artifact sharing is wired." type="button">Share final readout (planned)</button>
                </div>
              </section>
            ) : null}
          </main>
        </div>
      </section>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
