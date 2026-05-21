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
  { id: "app_intelligence_human_review", label: "App Intelligence review", count: 0, status: "clear", detail: "No App Intelligence human-review decisions pending" },
  { id: "first_pass_summaries", label: "First-pass summaries", count: 68, status: "pending", detail: "Stored transcripts waiting for first-pass summaries" },
  { label: "Memory harvest", count: 0, status: "gated", detail: "Requires explicit review file approval" },
  { label: "Speaker IDs", count: 0, status: "planned", detail: "Contact dedupe tables are planned in P09" }
  ],
  items: []
};

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
  runs: { items: [], total: 0 }
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

function capabilityLabels(capabilities) {
  if (Array.isArray(capabilities)) return capabilities;
  if (capabilities && typeof capabilities === "object") {
    return Object.entries(capabilities)
      .filter(([, enabled]) => Boolean(enabled))
      .map(([name]) => statusLabel(name));
  }
  return [];
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
  const [query, setQuery] = useState("");
  const [library, setLibrary] = useState({ items: FALLBACK_LIBRARY, total: FALLBACK_LIBRARY.length });
  const [reviewQueue, setReviewQueue] = useState(FALLBACK_REVIEW_QUEUE);
  const [selectedId, setSelectedId] = useState(FALLBACK_LIBRARY[0].id);
  const [health, setHealth] = useState({ status: "offline", store_dir: "fallback demo data" });
  const [apiError, setApiError] = useState("");
  const [reviewAction, setReviewAction] = useState({ status: "idle", message: "", manifest: "", batchId: "" });
  const [humanReviewAction, setHumanReviewAction] = useState({ status: "idle", message: "", payload: null });
  const [intelligence, setIntelligence] = useState(FALLBACK_INTELLIGENCE);
  const [selectedTask, setSelectedTask] = useState("first_pass_summary");
  const [taskDraft, setTaskDraft] = useState({ provider: "", model: "", timeout: "", temperature: "", fallbacks: "", human_review: "", requires_ledger: false });
  const [configAction, setConfigAction] = useState({ status: "idle", message: "", preview: null });
  const [runAction, setRunAction] = useState({ status: "idle", message: "", runId: "" });
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedRunDetail, setSelectedRunDetail] = useState(null);
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

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [healthPayload, libraryPayload, reviewPayload, providerPayload, configPayload, runsPayload] = await Promise.all([
          fetchJson("/api/health"),
          fetchJson("/api/library?limit=25"),
          fetchJson("/api/review-queue?limit=100"),
          fetchJson("/api/intelligence/providers"),
          fetchJson("/api/intelligence/config"),
          fetchJson("/api/intelligence/runs?limit=8")
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setLibrary(libraryPayload);
        setReviewQueue(reviewPayload);
        setIntelligence({ providers: providerPayload, config: configPayload, runs: runsPayload });
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
    async function loadRunDetail() {
      if (!selectedRunId) {
        setSelectedRunDetail(null);
        setRunDetailAction({ status: "idle", message: "" });
        return;
      }
      setRunDetailAction({ status: "loading", message: "Loading selected run ledger..." });
      try {
      const payload = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
        if (cancelled) return;
        setSelectedRunDetail(payload);
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
      } catch (error) {
        if (cancelled) return;
        setSelectedRunDetail(null);
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
  const taskEntries = Object.entries(intelligence.config?.tasks || {});
  const selectedTaskConfig = intelligence.config?.tasks?.[selectedTask] || taskEntries[0]?.[1] || null;
  const selectedProvider = (intelligence.providers?.providers || []).find((provider) => provider.id === selectedTaskConfig?.provider);
  const selectedTaskFingerprint = selectedTaskConfig ? JSON.stringify(selectedTaskConfig) : "";

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
        const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
        setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
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
      const detail = await fetchJson(`/api/intelligence/runs/${encodeURIComponent(selectedRunId)}?event_limit=12`);
      setSelectedRunDetail(detail);
      setForkPreflightAction({
        status: "previewed",
        message: "Fork preflight recorded; no threads, branches, or provider work were started.",
        payload
      });
    } catch (error) {
      setForkPreflightAction({ status: "error", message: `Fork preflight failed: ${error.message}`, payload: null });
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
              <span>{library.total ?? visibleItems.length} stored rows</span>
              <span>{reviewQueue.total_open ?? reviewBuckets.reduce((total, item) => total + item.count, 0)} open reviews</span>
              {activeNav === "Intelligence" && <span>{taskEntries.length} task routes</span>}
            </div>
          </div>

          {activeNav === "Review Queue" ? (
            <ReviewQueue
              queue={reviewQueue}
              reviewAction={reviewAction}
              onPrepareFirstPass={prepareFirstPassBatch}
              onSubmitFirstPass={submitFirstPassBatch}
              onRefreshFirstPass={refreshFirstPassBatch}
              humanReviewAction={humanReviewAction}
              onRecordHumanReview={recordHumanReviewDecision}
            />
          ) : activeNav === "Intelligence" ? (
            <IntelligencePanel
              config={intelligence.config}
              providers={intelligence.providers}
              runs={intelligence.runs}
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
            <LibraryTable items={visibleItems} selectedId={selected?.id} onSelect={setSelectedId} />
          )}
        </section>

        <aside className="right-pane">
          <button className="pane-toggle" onClick={() => setRightCollapsed((value) => !value)} type="button">
            {rightCollapsed ? "Inspector +" : "Collapse inspector"}
          </button>
          <Inspector
            item={selected}
            activeNav={activeNav}
            reviewQueue={reviewQueue}
            selectedTask={selectedTask}
            selectedTaskConfig={selectedTaskConfig}
            selectedProvider={selectedProvider}
            configAction={configAction}
            runAction={runAction}
            selectedRunId={selectedRunId}
            selectedRunDetail={selectedRunDetail}
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
            intelligence={intelligence}
          />
        </aside>
      </section>
    </main>
  );
}

function IntelligencePanel({
  config,
  providers,
  runs,
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
  const providerList = providers?.providers || [];
  const taskEntries = Object.entries(config?.tasks || {});
  const recentRuns = runs?.items || [];
  const selectedCapabilities = capabilityLabels(selectedProvider?.capabilities);
  const selectedChecks = selectedProvider?.checks && typeof selectedProvider.checks === "object" ? Object.entries(selectedProvider.checks) : [];
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

function ReviewQueue({ queue, reviewAction, onPrepareFirstPass, onSubmitFirstPass, onRefreshFirstPass, humanReviewAction, onRecordHumanReview }) {
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
  activeNav,
  reviewQueue,
  selectedTask,
  selectedTaskConfig,
  selectedProvider,
  configAction,
  selectedRunId,
  selectedRunDetail,
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
      (latestDecision.action === "stop" || latestDecision.action === "ask_for_human_review");
    const latestDecisionCanForkPreflight =
      latestDecision?.status === "validated" && latestDecision.action === "fork_branches";
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
                    </article>
                  ))}
                </div>
              ) : null}
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
