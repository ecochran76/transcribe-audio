import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { IdentityReviewView } from "./identity-review.jsx";
import { Icon } from "./icons.jsx";

const NAV_ITEMS = [
  { id: "Library", label: "Library", icon: "library", enabled: true },
  { id: "Review Queue", label: "Review Queue", icon: "queue", enabled: true },
  { id: "People", label: "Contacts", icon: "people", enabled: true },
  { id: "Provenance", label: "Provenance", enabled: true },
  { id: "Settings", label: "Settings", enabled: true }
];

const PRIMARY_NAV_ITEMS = NAV_ITEMS.filter((item) => ["Library", "Review Queue", "People"].includes(item.id));

const LIBRARY_KIND_FILTERS = [
  { id: "all", label: "All artifacts" },
  { id: "transcript", label: "Transcripts" },
  { id: "readout", label: "Summaries" },
  { id: "contextual_readout", label: "Contextual readouts" }
];

const WORKFLOW_VIEWS = [
  { id: "transcript", label: "Transcript" },
  { id: "summary", label: "First-pass summary" },
  { id: "context", label: "Context workbench" },
  { id: "speakers", label: "Speakers" },
  { id: "output", label: "Final readout" }
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

function readInitialUrlState() {
  const params = new URLSearchParams(window.location.search);
  const requestedView = params.get("view");
  const requestedSection = params.get("section");
  const view = requestedView === "Intelligence"
    ? "Settings"
    : requestedView === "Identity Review"
      ? "Review Queue"
      : requestedView;
  const kind = params.get("kind");
  const workflow = params.get("workflow");
  return {
    activeNav: NAV_ITEMS.some((item) => item.id === view && item.enabled) ? view : "Library",
    activeSettingsSection: requestedView === "Intelligence" ? "intelligence" : requestedSection || "",
    kindFilter: LIBRARY_KIND_FILTERS.some((item) => item.id === kind) ? kind : "all",
    query: params.get("q") || "",
    selectedId: params.get("selected") || FALLBACK_LIBRARY[0].id,
    conversationOpen: params.get("conversation") === "1",
    activeWorkflowView: WORKFLOW_VIEWS.some((item) => item.id === workflow) ? workflow : "transcript"
  };
}

const FALLBACK_INTELLIGENCE = {
  config: {
    schema_version: "transcribe-audio.intelligence-config.v1",
    config_path: "~/.local/state/transcribe-audio/intelligence.config.json",
    profiles: {
      openai_readout: {
        label: "OpenAI readout",
        description: "General transcript summarization and contextual readout profile.",
        provider: "openai-compatible",
        model: "gpt-4o-mini",
        base_url: "",
        timeout: 120,
        temperature: 0.1
      },
      codex_supervisor: {
        label: "Codex supervisor",
        description: "Ledger-backed App Intelligence supervisor profile.",
        provider: "codex-app-server",
        model: "",
        base_url: "",
        timeout: 120,
        temperature: 0
      }
    },
    task_profiles: {
      first_pass_summary: "openai_readout",
      app_supervisor: "codex_supervisor"
    },
    tasks: {
      first_pass_summary: {
        task: "first_pass_summary",
        profile: "openai_readout",
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
        profile: "codex_supervisor",
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
      { id: "auracall", label: "AuraCall", status: "configured-by-runtime-env", capabilities: ["summarize", "browser backed batch"] },
      { id: "codex-app-server", label: "Codex app-server", status: "ready", ready: true, capabilities: { persistent_sessions: true, branching: true } }
    ],
    default_supervisor: "codex-app-server"
  },
  runs: { items: [], total: 0 },
  smokes: { latest_report: null, reports: [], runs: [], report_count: 0, run_count: 0 },
  smokeJobs: { items: [], total: 0, available_job_types: [] }
};

const FALLBACK_PROVENANCE = {
  schema_version: "transcribe-audio.provenance-config.v1",
  config_path: "~/.local/state/transcribe-audio/provenance.config.json",
  exists: false,
  profile: "default",
  config: {
    active_profile: "default",
    profiles: {},
    sources: {},
    mutation_policy: {}
  },
  calendar_metadata: {
    provider_configs: [],
    provenance_calendar_ids: [],
    provenance_ical_urls: [],
    warnings: []
  },
  contact_source_config: {}
};

const FALLBACK_AUTOMATION = {
  schema_version: "transcribe-audio.automation-config.v1",
  config_path: "~/.local/state/transcribe-audio/automation.config.json",
  exists: false,
  profile: "default",
  stage_order: ["ingest_audio", "transcribe_audio", "initial_summary", "speaker_identity", "context_collection", "final_readout"],
  mode_choices: ["manual", "one_click", "automatic"],
  stages: {
    ingest_audio: { label: "Ingest audio", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: false, automatic_available: false } },
    transcribe_audio: { label: "Transcribe audio", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: false, automatic_available: false } },
    initial_summary: { label: "Initial summary", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: true, automatic_available: false } },
    speaker_identity: { label: "Speaker identity", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: false, automatic_available: false } },
    context_collection: { label: "Context collection", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: false, automatic_available: false } },
    final_readout: { label: "Final readout", enabled: false, mode: "manual", requires_review: true, notes: "", capabilities: { one_click_available: false, automatic_available: false } }
  },
  will_execute_workflow_stage: false,
  will_execute_external_action: false
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
  return String(status || "").replaceAll("_", " ");
}

function workflowViewForStage(stage) {
  if (stage === "speakers") return "speakers";
  if (stage === "context") return "context";
  if (stage === "output" || stage === "final_readout") return "output";
  if (stage === "summary" || stage === "first_pass_summary") return "summary";
  return "transcript";
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

function automationStageEntries(automation) {
  const stages = automation?.stages || {};
  const order = automation?.stage_order || Object.keys(stages);
  return order
    .filter((stageId) => stages[stageId])
    .map((stageId) => [stageId, stages[stageId]]);
}

function automationDraftFromConfig(automation) {
  const stageDrafts = {};
  for (const [stageId, stage] of automationStageEntries(automation)) {
    stageDrafts[stageId] = {
      enabled: Boolean(stage.enabled),
      mode: stage.mode || "manual",
      requires_review: stage.requires_review !== false,
      notes: stage.notes || ""
    };
  }
  return {
    profile: automation?.profile || "default",
    stages: stageDrafts
  };
}

function automationUpdateFromDraft(draft) {
  return {
    profile: draft.profile || "default",
    stages: Object.fromEntries(
      Object.entries(draft.stages || {}).map(([stageId, stage]) => [
        stageId,
        {
          enabled: Boolean(stage.enabled),
          mode: stage.mode || "manual",
          requires_review: stage.requires_review !== false,
          notes: stage.notes || ""
        }
      ])
    )
  };
}

function intelligenceUpdateFromDraft(draft) {
  const payload = {
    profile: draft.profile,
    fallbacks: String(draft.fallbacks || "").split(",").map((item) => item.trim()).filter(Boolean),
    human_review: draft.human_review,
    requires_ledger: Boolean(draft.requires_ledger)
  };
  if (draft.provider) payload.provider = draft.provider;
  if (draft.model) payload.model = draft.model;
  if (draft.timeout !== "") payload.timeout = Number(draft.timeout);
  if (draft.temperature !== "") payload.temperature = Number(draft.temperature);
  return payload;
}

function intelligenceProfileEntries(config) {
  const profiles = config?.profiles || {};
  return Object.entries(profiles);
}

function profileDraftFromConfig(profile) {
  return {
    label: profile?.label || "",
    description: profile?.description || "",
    provider: profile?.provider || "",
    model: profile?.model || "",
    base_url: profile?.base_url || "",
    timeout: profile?.timeout ?? "",
    temperature: profile?.temperature ?? ""
  };
}

function profileUpdateFromDraft(draft) {
  return {
    label: draft.label,
    description: draft.description,
    provider: draft.provider,
    model: draft.model,
    base_url: draft.base_url,
    timeout: draft.timeout === "" ? "" : Number(draft.timeout),
    temperature: draft.temperature === "" ? "" : Number(draft.temperature)
  };
}

function taskDraftFromConfig(config) {
  return {
    profile: config?.profile || "",
    provider: config?.provider || "",
    model: config?.model || "",
    timeout: config?.timeout ?? "",
    temperature: config?.temperature ?? "",
    fallbacks: (config?.fallbacks || []).join(", "),
    human_review: config?.human_review || "",
    requires_ledger: Boolean(config?.requires_ledger)
  };
}

function normalizeForCompare(value) {
  return JSON.stringify(value ?? null);
}

function automationDraftDirty(draft, automation) {
  return normalizeForCompare(automationUpdateFromDraft(draft)) !== normalizeForCompare(automationUpdateFromDraft(automationDraftFromConfig(automation)));
}

function intelligenceDraftDirty(draft, selectedTaskConfig) {
  if (!selectedTaskConfig) return false;
  return normalizeForCompare(intelligenceUpdateFromDraft(draft)) !== normalizeForCompare(intelligenceUpdateFromDraft(taskDraftFromConfig(selectedTaskConfig)));
}

function profileDraftDirty(draft, profileConfig) {
  if (!profileConfig) return Boolean(draft?.label || draft?.provider || draft?.model || draft?.description);
  return normalizeForCompare(profileUpdateFromDraft(draft)) !== normalizeForCompare(profileUpdateFromDraft(profileDraftFromConfig(profileConfig)));
}

function profileIdFromLabel(label, fallback = "new_profile") {
  const value = String(label || fallback)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return value || fallback;
}

function uniqueProfileId(baseId, profiles) {
  const existing = new Set(Object.keys(profiles || {}));
  let candidate = profileIdFromLabel(baseId);
  if (!existing.has(candidate)) return candidate;
  let index = 2;
  while (existing.has(`${candidate}_${index}`)) index += 1;
  return `${candidate}_${index}`;
}

function defaultProfileDraft(provider = "openai-compatible") {
  if (provider === "auracall") {
    return {
      label: "AuraCall readout",
      description: "Browser-backed AuraCall agent profile for transcript readouts.",
      provider,
      model: "",
      base_url: "",
      timeout: 300,
      temperature: 0.1
    };
  }
  if (provider === "codex-app-server") {
    return {
      label: "Codex app-server",
      description: "Ledger-backed App Intelligence supervisor profile.",
      provider,
      model: "",
      base_url: "",
      timeout: 120,
      temperature: 0
    };
  }
  if (provider === "codex-exec") {
    return {
      label: "Codex exec",
      description: "Local Codex CLI execution profile.",
      provider,
      model: "",
      base_url: "",
      timeout: 300,
      temperature: 0
    };
  }
  return {
    label: "OpenAI readout",
    description: "General transcript summarization and contextual readout profile.",
    provider,
    model: "gpt-4o-mini",
    base_url: "",
    timeout: 120,
    temperature: 0.1
  };
}

function agentIdFromModel(model) {
  const value = String(model || "").trim();
  return value.startsWith("agent:") ? value.slice("agent:".length).trim() : "";
}

function agentModelFromId(agentId) {
  const value = String(agentId || "").trim();
  return value ? `agent:${value}` : "";
}

function auracallAgentOptions(intelligence) {
  return intelligence?.auracall_readiness?.agent_options || intelligence?.config?.auracall_readiness?.agent_options || [];
}

function shouldUseAuraCallAgentSelector(profileDraft, selectedProfileProvider, agentOptions) {
  return (
    profileDraft.provider === "auracall"
    || agentIdFromModel(profileDraft.model)
    || selectedProfileProvider?.id === "auracall"
    || (agentOptions.length > 0 && String(profileDraft.base_url || "").includes("18095"))
  );
}

function auraCallAgentDescription(agent) {
  if (!agent) return "";
  return agent.settings_description || [
    agent.runtimeProfileId ? `runtime ${agent.runtimeProfileId}` : "",
    agent.browserProfileId ? `browser ${agent.browserProfileId}` : "",
    agent.projectBinding?.label ? `project ${agent.projectBinding.label}` : "",
    agent.ready ? "ready" : "not ready"
  ].filter(Boolean).join("; ");
}

function provenanceDraftDirty(draft, provenance) {
  const currentProfile = provenance?.config?.active_profile || provenance?.profile || "default";
  if ((draft.activeProfile || "default") !== currentProfile) return true;
  if (draft.newIcalId || draft.newIcalLabel || draft.newIcalUrl) return true;
  return provenanceSourceEntries(provenance).some(([sourceId, source]) => {
    const currentEnabled = source.enabled !== false;
    const draftedEnabled = draft.sourceEnabled?.[sourceId];
    return typeof draftedEnabled === "boolean" && draftedEnabled !== currentEnabled;
  });
}

function resetProvenanceDraftFromConfig(provenance) {
  const sourceEnabled = {};
  for (const [sourceId, source] of provenanceSourceEntries(provenance)) {
    sourceEnabled[sourceId] = source.enabled !== false;
  }
  return {
    activeProfile: provenance?.config?.active_profile || provenance?.profile || "default",
    sourceEnabled,
    newIcalId: "",
    newIcalLabel: "",
    newIcalUrl: ""
  };
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

function normalizeSourceId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function contactCandidateText(candidate) {
  const parts = [
    candidate?.label,
    candidate?.email,
    candidate?.organization,
    candidate?.role,
    candidate?.source,
    candidate?.source_type,
    candidate?.source_profile
  ];
  if (Array.isArray(candidate?.merged_sources)) {
    candidate.merged_sources.forEach((source) => {
      parts.push(source?.label, source?.email, source?.source_type, source?.source_profile);
    });
  }
  return parts.filter(Boolean).join(" ").toLowerCase();
}

function contactRankScore(candidate) {
  const value = Number(candidate?.rank_score);
  return Number.isFinite(value) ? value : 0;
}

function contactConfidence(candidate) {
  const value = Number(candidate?.confidence);
  return Number.isFinite(value) ? value : 0;
}

function contactRankingReasons(candidate) {
  const reasons = [];
  if (Array.isArray(candidate?.ranking_reasons)) reasons.push(...candidate.ranking_reasons);
  if (Array.isArray(candidate?.relationship_affinity?.evidence)) reasons.push(...candidate.relationship_affinity.evidence);
  return [...new Set(reasons.map((reason) => String(reason || "").trim()).filter(Boolean))].slice(0, 3);
}

function contactMatchesQuery(candidate, query) {
  const terms = String(query || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const text = contactCandidateText(candidate);
  return terms.every((term) => text.includes(term));
}

function contactCandidateId(candidate) {
  return String(candidate?.contact_id || candidate?.id || candidate?.dedupe_key || "").trim();
}

function contactCandidateIds(candidate) {
  const ids = [contactCandidateId(candidate)];
  if (Array.isArray(candidate?.merged_contact_ids)) {
    candidate.merged_contact_ids.forEach((id) => ids.push(String(id || "").trim()));
  }
  return [...new Set(ids.filter(Boolean))];
}

function contactIdSetHasCandidate(idSet, candidate) {
  return contactCandidateIds(candidate).some((id) => idSet.has(id));
}

function speakerAssignmentMatchesCandidate(speaker, candidate) {
  const assignment = speaker?.assignment || {};
  const candidateIds = contactCandidateIds(candidate);
  const assignmentId = String(assignment.contact_id || "").trim();
  const assignmentEmail = String(assignment.email || assignment.contact_email || "").trim().toLowerCase();
  const assignmentLabel = String(assignment.contact_label || "").trim().toLowerCase();
  const candidateEmail = String(candidate?.email || "").trim().toLowerCase();
  const candidateLabel = String(candidate?.label || "").trim().toLowerCase();
  return (
    (assignmentId && candidateIds.includes(assignmentId))
    || (assignmentEmail && candidateEmail && assignmentEmail === candidateEmail)
    || (assignmentLabel && candidateLabel && assignmentLabel === candidateLabel)
  );
}

function uniqueContactCandidates(candidates) {
  const result = [];
  const seen = new Set();
  candidates.filter(Boolean).forEach((candidate) => {
    const ids = contactCandidateIds(candidate);
    const email = String(candidate?.email || "").trim().toLowerCase();
    const dedupeKey = String(candidate?.dedupe_key || "").trim();
    const keys = [...ids, dedupeKey ? `dedupe:${dedupeKey}` : "", email ? `email:${email}` : ""].filter(Boolean);
    const key = keys[0] || contactCandidateText(candidate);
    if (!key || keys.some((id) => seen.has(id))) return;
    result.push(candidate);
    keys.forEach((id) => seen.add(id));
  });
  return result;
}

function provenanceSourceEntries(provenance) {
  return Object.entries(provenance?.config?.sources || {});
}

function provenanceSourceCounts(provenance) {
  return provenanceSourceEntries(provenance).reduce((counts, [, source]) => {
    const kind = source?.kind || "unknown";
    counts[kind] = (counts[kind] || 0) + 1;
    return counts;
  }, {});
}

function buildProvenanceUpdate(draft, provenance) {
  const config = provenance?.config || {};
  const currentProfile = config.active_profile || provenance?.profile || "default";
  const activeProfile = draft.activeProfile || currentProfile;
  const update = {};
  const sourceUpdates = {};
  for (const [sourceId, source] of provenanceSourceEntries(provenance)) {
    const currentEnabled = source.enabled !== false;
    const draftedEnabled = draft.sourceEnabled?.[sourceId];
    if (typeof draftedEnabled === "boolean" && draftedEnabled !== currentEnabled) {
      sourceUpdates[sourceId] = { enabled: draftedEnabled };
    }
  }
  const newIcalId = normalizeSourceId(draft.newIcalId);
  const newIcalLabel = String(draft.newIcalLabel || "").trim();
  const newIcalUrl = String(draft.newIcalUrl || "").trim();
  if (newIcalId || newIcalLabel || newIcalUrl) {
    if (!newIcalId || !newIcalLabel || !newIcalUrl) {
      throw new Error("iCal source id, label, and URL or env ref are required.");
    }
    sourceUpdates[newIcalId] = {
      kind: "ical_calendar",
      enabled: true,
      label: newIcalLabel,
      capabilities: ["calendar"],
      read_only: true,
      calendar: {
        timezone: "America/Chicago",
        max_events: 250,
        cache_ttl_seconds: 900
      },
      sensitive_fields: newIcalUrl.startsWith("env:") ? ["url_ref"] : ["url"]
    };
    if (newIcalUrl.startsWith("env:")) {
      sourceUpdates[newIcalId].url_ref = newIcalUrl;
    } else {
      sourceUpdates[newIcalId].url = newIcalUrl;
    }
    const profile = config.profiles?.[activeProfile] || {};
    const workflows = profile.workflows || {};
    const calendarWorkflow = workflows.calendar_metadata || {};
    const profileSourceIds = Array.isArray(profile.source_ids) ? profile.source_ids : [];
    const provenanceSources = Array.isArray(calendarWorkflow.provenance_sources)
      ? calendarWorkflow.provenance_sources
      : [];
    update.profiles = {
      [activeProfile]: {
        source_ids: [...new Set([...profileSourceIds, newIcalId])],
        workflows: {
          calendar_metadata: {
            provenance_sources: provenanceSources.some((item) => item?.source_id === newIcalId)
              ? provenanceSources
              : [...provenanceSources, { source_id: newIcalId }]
          }
        }
      }
    };
  }
  if (activeProfile !== currentProfile) update.active_profile = activeProfile;
  if (Object.keys(sourceUpdates).length) update.sources = sourceUpdates;
  return update;
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
  const [initialUrlState] = useState(readInitialUrlState);
  const [activeNav, setActiveNav] = useState(initialUrlState.activeNav);
  const [activeSettingsSection, setActiveSettingsSection] = useState(initialUrlState.activeSettingsSection || "account");
  const [leftCollapsed, setLeftCollapsed] = useState(initialUrlState.activeNav === "Library");
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [leftPaneWidth, setLeftPaneWidth] = useState(300);
  const [rightPaneWidth, setRightPaneWidth] = useState(380);
  const [kindFilter, setKindFilter] = useState(initialUrlState.kindFilter);
  const [query, setQuery] = useState(initialUrlState.query);
  const [library, setLibrary] = useState({ items: FALLBACK_LIBRARY, total: FALLBACK_LIBRARY.length });
  const [conversations, setConversations] = useState(FALLBACK_CONVERSATIONS);
  const [conversationSearchStatus, setConversationSearchStatus] = useState({ status: "idle", message: "Conversation search has not loaded yet." });
  const [reviewQueue, setReviewQueue] = useState(FALLBACK_REVIEW_QUEUE);
  const [selectedId, setSelectedId] = useState(initialUrlState.selectedId);
  const [health, setHealth] = useState({ status: "offline", store_dir: "fallback demo data" });
  const [apiError, setApiError] = useState("");
  const [reviewAction, setReviewAction] = useState({ status: "idle", message: "", manifest: "", batchId: "", payload: null });
  const [firstPassBatchManifests, setFirstPassBatchManifests] = useState({ items: [], total: 0, limit: 0 });
  const [humanReviewAction, setHumanReviewAction] = useState({ status: "idle", message: "", payload: null });
  const [intelligence, setIntelligence] = useState(FALLBACK_INTELLIGENCE);
  const [provenance, setProvenance] = useState(FALLBACK_PROVENANCE);
  const [automation, setAutomation] = useState(FALLBACK_AUTOMATION);
  const [provenanceDoctor, setProvenanceDoctor] = useState(null);
  const [provenanceDraft, setProvenanceDraft] = useState({
    activeProfile: "default",
    sourceEnabled: {},
    newIcalId: "",
    newIcalLabel: "",
    newIcalUrl: ""
  });
  const [provenanceAction, setProvenanceAction] = useState({ status: "idle", message: "", preview: null });
  const [selectedTask, setSelectedTask] = useState("first_pass_summary");
  const [selectedProfile, setSelectedProfile] = useState("openai_readout");
  const [profileDraft, setProfileDraft] = useState(profileDraftFromConfig(FALLBACK_INTELLIGENCE.config.profiles.openai_readout));
  const [taskDraft, setTaskDraft] = useState({ profile: "", provider: "", model: "", timeout: "", temperature: "", fallbacks: "", human_review: "", requires_ledger: false });
  const [configAction, setConfigAction] = useState({ status: "idle", message: "", preview: null });
  const [automationDraft, setAutomationDraft] = useState(automationDraftFromConfig(FALLBACK_AUTOMATION));
  const [automationAction, setAutomationAction] = useState({ status: "idle", message: "", preview: null });
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
  const [conversationOpen, setConversationOpen] = useState(initialUrlState.conversationOpen);
  const [activeWorkflowView, setActiveWorkflowView] = useState(initialUrlState.activeWorkflowView);
  const [shareAction, setShareAction] = useState({ status: "idle", message: "", url: "" });
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [
          healthPayload,
          libraryPayload,
          reviewPayload,
          batchManifestPayload,
          providerPayload,
          configPayload,
          runsPayload,
          smokesPayload,
          smokeJobsPayload,
          provenancePayload,
          automationPayload,
          provenanceDoctorPayload
        ] = await Promise.all([
          fetchJson("/api/health"),
          fetchJson("/api/library?limit=200"),
          fetchJson("/api/review-queue?limit=100"),
          fetchJson("/api/review-queue/first-pass-summaries/manifests?limit=5"),
          fetchJson("/api/intelligence/providers"),
          fetchJson("/api/intelligence/config"),
          fetchJson("/api/intelligence/runs?limit=8"),
          fetchJson("/api/intelligence/smokes?limit=5"),
          fetchJson("/api/intelligence/smoke-jobs?limit=20"),
          fetchJson("/api/provenance/config"),
          fetchJson("/api/automation/config"),
          fetchJson("/api/provenance/config/doctor")
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setLibrary(libraryPayload);
        setReviewQueue(reviewPayload);
        setFirstPassBatchManifests(batchManifestPayload);
        setIntelligence({ providers: providerPayload, config: configPayload, runs: runsPayload, smokes: smokesPayload, smokeJobs: smokeJobsPayload });
        setProvenance(provenancePayload);
        setAutomation(automationPayload);
        setProvenanceDoctor(provenanceDoctorPayload);
        setSelectedId((currentId) => currentId || libraryPayload.items?.[0]?.id || "");
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
    setLeftCollapsed(activeNav === "Library");
  }, [activeNav]);

  useEffect(() => {
    if (!accountMenuOpen) return undefined;
    const closeFromPointer = (event) => {
      if (!event.target.closest(".account-menu-wrap")) setAccountMenuOpen(false);
    };
    const closeFromKeyboard = (event) => {
      if (event.key === "Escape") setAccountMenuOpen(false);
    };
    document.addEventListener("pointerdown", closeFromPointer);
    window.addEventListener("keydown", closeFromKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeFromPointer);
      window.removeEventListener("keydown", closeFromKeyboard);
    };
  }, [accountMenuOpen]);

  useEffect(() => {
    const sourceEnabled = {};
    for (const [sourceId, source] of provenanceSourceEntries(provenance)) {
      sourceEnabled[sourceId] = source.enabled !== false;
    }
    setProvenanceDraft((current) => ({
      ...current,
      activeProfile: provenance.config?.active_profile || provenance.profile || "default",
      sourceEnabled
    }));
  }, [
    JSON.stringify(Object.entries(provenance.config?.sources || {}).map(([sourceId, source]) => [sourceId, source?.enabled !== false])),
    provenance.config?.active_profile,
    provenance.profile
  ]);

  useEffect(() => {
    setAutomationDraft(automationDraftFromConfig(automation));
    setAutomationAction({ status: "idle", message: "", preview: null });
  }, [automation.config_path, automation.profile, JSON.stringify(automationStageEntries(automation).map(([stageId, stage]) => [stageId, stage.enabled, stage.mode, stage.requires_review, stage.notes]))]);

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
          if (currentId && currentId !== FALLBACK_LIBRARY[0].id) return currentId;
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
    if (activeNav === "Intelligence") {
      setActiveNav("Settings");
      setActiveSettingsSection("intelligence");
    }
  }, [activeNav]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (activeNav !== "Library") params.set("view", activeNav);
    if (activeNav === "Settings" && activeSettingsSection !== "account") {
      params.set("section", activeSettingsSection);
    }
    if (activeNav === "Library") {
      if (kindFilter !== "all") params.set("kind", kindFilter);
      if (query.trim()) params.set("q", query.trim());
      if (selectedId) params.set("selected", selectedId);
      if (conversationOpen) {
        params.set("conversation", "1");
        if (activeWorkflowView !== "transcript") params.set("workflow", activeWorkflowView);
      }
    }
    const nextQuery = params.toString();
    const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ""}${window.location.hash}`;
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) window.history.replaceState(null, "", nextUrl);
  }, [activeNav, activeSettingsSection, activeWorkflowView, conversationOpen, kindFilter, query, selectedId]);

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
  const selectedConversation = visibleConversationRows.find((row) => (row.artifacts || []).some((artifact) => artifact.id === selectedId)) || null;
  const selected =
    selectedConversation?.representative ||
    visibleItems.find((item) => item.id === selectedId) ||
    visibleConversationRows[0]?.representative ||
    visibleItems[0] ||
    null;
  const reviewBuckets = reviewQueue.buckets || FALLBACK_REVIEW_QUEUE.buckets;
  const taskEntries = Object.entries(intelligence.config?.tasks || {});
  const profileEntries = intelligenceProfileEntries(intelligence.config);
  const selectedTaskConfig = intelligence.config?.tasks?.[selectedTask] || taskEntries[0]?.[1] || null;
  const selectedProfileConfig = intelligence.config?.profiles?.[selectedProfile] || null;
  const selectedProvider = (intelligence.providers?.providers || []).find((provider) => provider.id === selectedTaskConfig?.provider);
  const selectedTaskFingerprint = selectedTaskConfig ? JSON.stringify(selectedTaskConfig) : "";
  const selectedProfileFingerprint = selectedProfileConfig ? JSON.stringify(selectedProfileConfig) : "";
  const smokeJobsActive = hasActiveSmokeJob(intelligence.smokeJobs);
  const provenanceEntries = provenanceSourceEntries(provenance);
  const provenanceCounts = provenanceSourceCounts(provenance);
  const enabledProvenanceSourceCount = provenanceEntries.filter(([, source]) => source.enabled !== false).length;
  const provenanceStatus = provenanceDoctor?.status || (provenance.exists ? "unknown" : "missing");
  const isSettingsView = activeNav === "Settings";

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

  async function copyCurrentWorkspaceUrl() {
    const url = window.location.href;
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard API is unavailable.");
      await navigator.clipboard.writeText(url);
      setShareAction({ status: "ok", message: "Copied current workspace link.", url });
    } catch (error) {
      setShareAction({
        status: "blocked",
        message: "Clipboard blocked. Select the link below to copy it manually.",
        url
      });
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
    setTaskDraft(taskDraftFromConfig(selectedTaskConfig));
    setConfigAction({ status: "idle", message: "", preview: null });
  }, [selectedTask, selectedTaskFingerprint]);

  useEffect(() => {
    if (!profileEntries.length) return;
    if (!profileEntries.some(([profileId]) => profileId === selectedProfile)) {
      if (profileDraftDirty(profileDraft, null)) return;
      setSelectedProfile(profileEntries[0][0]);
    }
  }, [JSON.stringify(profileEntries.map(([profileId]) => profileId)), selectedProfile, profileDraft]);

  useEffect(() => {
    if (!selectedProfileConfig) return;
    setProfileDraft(profileDraftFromConfig(selectedProfileConfig));
    setConfigAction({ status: "idle", message: "", preview: null });
  }, [selectedProfile, selectedProfileFingerprint]);

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

  function openQueueConversation(item) {
    const documentId = item?.representative_document_id || item?.document_id || "";
    if (!documentId) return;
    setSelectedId(documentId);
    setActiveWorkflowView(workflowViewForStage(item.workflow_stage));
    setConversationOpen(true);
    openAppView("Library");
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
    return intelligenceUpdateFromDraft(taskDraft);
  }

  function profileUpdatePayload() {
    return profileUpdateFromDraft(profileDraft);
  }

  function createProfileDraft(provider = "openai-compatible") {
    const draft = defaultProfileDraft(provider);
    const profileId = uniqueProfileId(draft.label, intelligence.config?.profiles || {});
    setSelectedProfile(profileId);
    setProfileDraft(draft);
    setConfigAction({ status: "idle", message: `Drafting new profile ${draft.label}. Preview and apply to save it.`, preview: null });
  }

  function duplicateProfileDraft() {
    const base = profileDraftFromConfig(selectedProfileConfig || profileDraft);
    const label = `${base.label || selectedProfile || "Profile"} copy`;
    const profileId = uniqueProfileId(label, intelligence.config?.profiles || {});
    setSelectedProfile(profileId);
    setProfileDraft({ ...base, label });
    setConfigAction({ status: "idle", message: `Drafting duplicate profile ${label}. Preview and apply to save it.`, preview: null });
  }

  async function previewConfigUpdate() {
    setConfigAction({ status: "running", message: "Previewing intelligence routing update...", preview: null });
    try {
      const taskDirty = intelligenceDraftDirty(taskDraft, selectedTaskConfig);
      const editedProfile = profileDraftDirty(profileDraft, selectedProfileConfig);
      const targetLabel = [
        editedProfile ? `profile ${selectedProfile}` : "",
        taskDirty ? `component ${selectedTask}` : ""
      ].filter(Boolean).join(" and ");
      const payload = await postJson("/api/intelligence/config/preview", {
        task: taskDirty ? selectedTask : "",
        update: taskDirty ? taskUpdatePayload() : {},
        profile_id: editedProfile ? selectedProfile : "",
        profile_update: editedProfile ? profileUpdatePayload() : {}
      });
      setConfigAction({
        status: "previewed",
        message: `Preview ready for ${targetLabel || "intelligence config"}; no config was written.`,
        preview: payload
      });
    } catch (error) {
      setConfigAction({ status: "error", message: `Preview failed: ${error.message}`, preview: null });
    }
  }

  async function applyConfigUpdate() {
    const preview = configAction.preview;
    if (!preview) return;
    const taskDirty = intelligenceDraftDirty(taskDraft, selectedTaskConfig);
    const editedProfile = profileDraftDirty(profileDraft, selectedProfileConfig);
    const targetLabel = [
      editedProfile ? `profile ${selectedProfile}` : "",
      taskDirty ? `component ${selectedTask}` : ""
    ].filter(Boolean).join(" and ");
    const approved = window.confirm(`Apply intelligence config update for ${targetLabel || "current preview"}?`);
    if (!approved) return;
    setConfigAction((current) => ({ ...current, status: "applying", message: "Applying intelligence routing update..." }));
    try {
      const payload = await postJson("/api/intelligence/config/apply", {
        task: taskDirty ? selectedTask : "",
        update: taskDirty ? taskUpdatePayload() : {},
        profile_id: editedProfile ? selectedProfile : "",
        profile_update: editedProfile ? profileUpdatePayload() : {},
        approval_token: "APPLY_INTELLIGENCE_CONFIG_UPDATE"
      });
      const configPayload = await fetchJson("/api/intelligence/config");
      setIntelligence((current) => ({ ...current, config: configPayload }));
      setConfigAction({
        status: "applied",
        message: `Applied ${targetLabel || "intelligence config"}; rollback metadata is available in the last preview response.`,
        preview: payload
      });
    } catch (error) {
      setConfigAction((current) => ({ ...current, status: "error", message: `Apply failed: ${error.message}` }));
    }
  }

  async function deleteSelectedProfile() {
    if (!selectedProfile) return;
    const approved = window.confirm(`Delete intelligence profile ${selectedProfile}?`);
    if (!approved) return;
    setConfigAction({ status: "running", message: `Deleting profile ${selectedProfile}...`, preview: null });
    try {
      const payload = await postJson("/api/intelligence/config/apply", {
        profile_id: selectedProfile,
        delete_profile: true,
        approval_token: "APPLY_INTELLIGENCE_CONFIG_UPDATE"
      });
      const configPayload = await fetchJson("/api/intelligence/config");
      setIntelligence((current) => ({ ...current, config: configPayload }));
      const nextProfile = Object.keys(configPayload.profiles || {})[0] || "";
      setSelectedProfile(nextProfile);
      if (nextProfile) setProfileDraft(profileDraftFromConfig(configPayload.profiles[nextProfile]));
      setConfigAction({
        status: "applied",
        message: `Deleted profile ${selectedProfile}.`,
        preview: payload
      });
    } catch (error) {
      setConfigAction({ status: "error", message: `Delete failed: ${error.message}`, preview: null });
    }
  }

  async function refreshProvenanceConfig() {
    setProvenanceAction((current) => ({ ...current, status: "refreshing", message: "Refreshing provenance config..." }));
    try {
      const [configPayload, doctorPayload] = await Promise.all([
        fetchJson("/api/provenance/config"),
        fetchJson("/api/provenance/config/doctor")
      ]);
      setProvenance(configPayload);
      setProvenanceDoctor(doctorPayload);
      setProvenanceAction({ status: "loaded", message: "Provenance config refreshed.", preview: null });
    } catch (error) {
      setProvenanceAction({ status: "error", message: `Refresh failed: ${error.message}`, preview: null });
    }
  }

  function provenanceUpdatePayload() {
    const update = buildProvenanceUpdate(provenanceDraft, provenance);
    if (!Object.keys(update).length) throw new Error("No provenance config changes are staged.");
    return update;
  }

  async function previewProvenanceUpdate() {
    setProvenanceAction({ status: "running", message: "Previewing provenance config update...", preview: null });
    try {
      const payload = await postJson("/api/provenance/config/preview", {
        update: provenanceUpdatePayload()
      });
      setProvenanceAction({
        status: "previewed",
        message: "Preview ready; no provenance config was written.",
        preview: payload
      });
    } catch (error) {
      setProvenanceAction({ status: "error", message: `Preview failed: ${error.message}`, preview: null });
    }
  }

  async function applyProvenanceUpdate() {
    const preview = provenanceAction.preview;
    if (!preview) return;
    const approved = window.confirm("Apply provenance config update?");
    if (!approved) return;
    setProvenanceAction((current) => ({ ...current, status: "applying", message: "Applying provenance config update..." }));
    try {
      const payload = await postJson("/api/provenance/config/apply", {
        update: provenanceUpdatePayload(),
        approval_token: "APPLY_PROVENANCE_CONFIG_UPDATE"
      });
      const [configPayload, doctorPayload] = await Promise.all([
        fetchJson("/api/provenance/config"),
        fetchJson("/api/provenance/config/doctor")
      ]);
      setProvenance(configPayload);
      setProvenanceDoctor(doctorPayload);
      setProvenanceDraft((current) => ({
        ...current,
        newIcalId: "",
        newIcalLabel: "",
        newIcalUrl: ""
      }));
      setProvenanceAction({
        status: "applied",
        message: "Applied provenance config update.",
        preview: payload
      });
    } catch (error) {
      setProvenanceAction((current) => ({ ...current, status: "error", message: `Apply failed: ${error.message}` }));
    }
  }

  async function previewAutomationUpdate() {
    setAutomationAction({ status: "running", message: "Previewing automation settings update...", preview: null });
    try {
      const payload = await postJson("/api/automation/config/preview", {
        update: automationUpdateFromDraft(automationDraft)
      });
      setAutomationAction({
        status: "previewed",
        message: "Preview ready; no workflow stage was run and no config was written.",
        preview: payload
      });
    } catch (error) {
      setAutomationAction({ status: "error", message: `Automation preview failed: ${error.message}`, preview: null });
    }
  }

  async function applyAutomationUpdate() {
    if (!automationAction.preview) return;
    const approved = window.confirm("Apply automation settings update?");
    if (!approved) return;
    setAutomationAction((current) => ({ ...current, status: "applying", message: "Applying automation settings update..." }));
    try {
      const payload = await postJson("/api/automation/config/apply", {
        update: automationUpdateFromDraft(automationDraft),
        approval_token: "APPLY_AUTOMATION_CONFIG_UPDATE"
      });
      const automationPayload = await fetchJson("/api/automation/config");
      setAutomation(automationPayload);
      setAutomationAction({
        status: "applied",
        message: "Applied automation settings; no workflow stage was run.",
        preview: payload
      });
    } catch (error) {
      setAutomationAction((current) => ({ ...current, status: "error", message: `Automation apply failed: ${error.message}` }));
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

  function openAppView(view, options = {}) {
    setActiveNav(view);
    if (view === "Settings" && options.section) {
      setActiveSettingsSection(options.section);
    }
    setAccountMenuOpen(false);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">tr</span>
          <div>
            <strong>Transcript Console</strong>
            <small>{health.status === "ok" ? "Local API live" : "Preview data"}</small>
          </div>
        </div>
        <nav className="nav-tabs" aria-label="Primary">
          {PRIMARY_NAV_ITEMS.map((item) => (
            <button
              aria-label={item.label}
              className={activeNav === item.id ? "active" : ""}
              aria-current={activeNav === item.id ? "page" : undefined}
              key={item.id}
              onClick={() => openAppView(item.id)}
              title={item.label}
              type="button"
            >
              <Icon name={item.icon} />
              <span className="visually-hidden">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="account-menu-wrap">
          <button
            aria-expanded={accountMenuOpen}
            aria-haspopup="menu"
            className="account-chip"
            onClick={() => setAccountMenuOpen((open) => !open)}
            type="button"
          >
            <span className="account-avatar" aria-hidden="true">EC</span>
            <span>
              <strong>{automationDraft.profile || provenanceDraft.activeProfile || "default"}</strong>
              <small>{health.status === "ok" ? "connected" : "preview"}</small>
            </span>
          </button>
          {accountMenuOpen ? (
            <div className="account-menu" role="menu" aria-label="Account and settings">
              <div className="account-menu-heading">
                <strong>Account</strong>
                <small>{health.store_dir || "runtime unavailable"}</small>
              </div>
              <button role="menuitem" onClick={() => openAppView("Settings", { section: "account" })} type="button">Settings</button>
              <button role="menuitem" onClick={() => openAppView("Settings", { section: "account" })} type="button">Account management</button>
              <button role="menuitem" onClick={() => openAppView("Provenance")} type="button">Integrations / provenance</button>
              <button role="menuitem" onClick={() => openAppView("Settings", { section: "intelligence" })} type="button">Intelligence</button>
              <button role="menuitem" onClick={() => openAppView("Settings", { section: "automation" })} type="button">Automation</button>
              <button role="menuitem" onClick={() => openAppView("Settings", { section: "validation" })} type="button">Runtime status</button>
            </div>
          ) : null}
        </div>
      </header>

      <section
        className={[
          "workspace",
          activeNav === "Library" ? "library-workspace" : "",
          ["Review Queue", "People"].includes(activeNav) ? "identity-workspace" : "",
          isSettingsView ? "settings-workspace" : "",
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
            <h2>{activeNav === "Library" ? "Filters" : activeNav === "Settings" ? "Settings" : "Workflow filters"}</h2>
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
            ) : activeNav === "Settings" ? null : activeNav === "Provenance" ? (
              <div className="filter-card task-filter">
                <span>Source kinds</span>
                {Object.entries(provenanceCounts).map(([kind, count]) => (
                  <button key={kind} type="button">
                    {statusLabel(kind)}
                    <strong>{count}</strong>
                  </button>
                ))}
                {!Object.keys(provenanceCounts).length && (
                  <button disabled type="button">
                    No sources
                    <strong>0</strong>
                  </button>
                )}
              </div>
            ) : activeNav === "Library" ? (
              <>
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

        <section className={isSettingsView ? "center-pane settings-center-pane" : "center-pane"}>
          <div className="view-heading">
            <div>
              <p className="eyebrow">Operator Surface</p>
              <h1>
                {activeNav === "Review Queue"
                  ? "Review queue"
                  : activeNav === "People"
                      ? "Contacts"
                  : activeNav === "Intelligence"
                    ? "Intelligence routing"
                    : activeNav === "Provenance"
                      ? "Provenance configuration"
                      : activeNav === "Settings"
                        ? "Settings"
                        : "Transcript library"}
              </h1>
            </div>
            {!isSettingsView && !["Review Queue", "People"].includes(activeNav) && (
              <div className="summary-strip">
                <span>{conversations.total ?? visibleConversationRows.length} conversations</span>
                <span>{library.total ?? visibleItems.length} artifacts</span>
                <span>{reviewQueue.total_open ?? reviewBuckets.reduce((total, item) => total + item.count, 0)} open reviews</span>
                {activeNav === "Intelligence" && <span>{taskEntries.length} task routes</span>}
                {activeNav === "Provenance" && <span>{enabledProvenanceSourceCount} enabled sources</span>}
                {activeNav === "Provenance" && <span>{provenanceStatus}</span>}
              </div>
            )}
          </div>
          {activeNav === "Library" ? (
            <LibraryToolbar
              filterOpen={!leftCollapsed}
              kindFilter={kindFilter}
              libraryItems={library.items || []}
              onCopyWorkspaceUrl={copyCurrentWorkspaceUrl}
              onSetKindFilter={setKindFilter}
              onToggleFilters={() => setLeftCollapsed((value) => !value)}
              query={query}
              reviewCount={reviewQueue.total_open ?? reviewBuckets.reduce((total, item) => total + item.count, 0)}
              setQuery={setQuery}
            />
          ) : null}
          {activeNav === "Library" && shareAction.message ? (
            <div className={`share-link-notice ${shareAction.status}`} role="status">
              <span>{shareAction.message}</span>
              {shareAction.url && (
                <input
                  aria-label="Current workspace link"
                  onFocus={(event) => event.target.select()}
                  readOnly
                  value={shareAction.url}
                />
              )}
            </div>
          ) : null}
          {activeNav === "Library" ? (
            <details className="diagnostics-disclosure">
              <summary>Diagnostics</summary>
              <TestStatusStrip
                activeNav={activeNav}
                apiStatus={health.status}
                kindFilter={kindFilter}
                query={query}
                visibleCount={visibleConversationRows.length}
                totalCount={totalConversationCount}
                latestSmoke={intelligence.smokes?.latest_report}
                latestSmokeJob={intelligence.smokeJobs?.items?.[0]}
                provenanceStatus={provenanceStatus}
              />
            </details>
          ) : null}

          {activeNav === "Review Queue" ? (
            <IdentityReviewView mode="review" />
          ) : activeNav === "People" ? (
            <IdentityReviewView mode="people" />
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
          ) : activeNav === "Provenance" ? (
            <ProvenancePanel
              provenance={provenance}
              doctor={provenanceDoctor}
              draft={provenanceDraft}
              setDraft={setProvenanceDraft}
              action={provenanceAction}
              onPreview={previewProvenanceUpdate}
              onApply={applyProvenanceUpdate}
              onRefresh={refreshProvenanceConfig}
            />
          ) : activeNav === "Settings" ? (
            <SettingsPanel
              automation={automation}
              automationAction={automationAction}
              automationDraft={automationDraft}
              activeSettingsSection={activeSettingsSection}
              configAction={configAction}
              health={health}
              intelligence={intelligence}
              provenance={provenance}
              provenanceAction={provenanceAction}
              provenanceDoctor={provenanceDoctor}
              provenanceDraft={provenanceDraft}
              profileDraft={profileDraft}
              selectedTask={selectedTask}
              selectedTaskConfig={selectedTaskConfig}
              selectedProfile={selectedProfile}
              selectedProfileConfig={selectedProfileConfig}
              setAutomationAction={setAutomationAction}
              setConfigAction={setConfigAction}
              setActiveSettingsSection={setActiveSettingsSection}
              setProfileDraft={setProfileDraft}
              setProvenanceAction={setProvenanceAction}
              setProvenanceDraft={setProvenanceDraft}
              setSelectedProfile={setSelectedProfile}
              setSelectedTask={setSelectedTask}
              setTaskDraft={setTaskDraft}
              taskDraft={taskDraft}
              onApplyIntelligence={applyConfigUpdate}
              onApplyAutomation={applyAutomationUpdate}
              onApplyProvenance={applyProvenanceUpdate}
              onCreateProfile={createProfileDraft}
              onDeleteProfile={deleteSelectedProfile}
              onDuplicateProfile={duplicateProfileDraft}
              onOpenIntelligence={() => openAppView("Settings", { section: "intelligence" })}
              onOpenProvenance={() => openAppView("Provenance")}
              onPreviewIntelligence={previewConfigUpdate}
              onPreviewAutomation={previewAutomationUpdate}
              onPreviewProvenance={previewProvenanceUpdate}
              onRefreshProvenance={refreshProvenanceConfig}
              setAutomationDraft={setAutomationDraft}
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
            provenance={provenance}
            provenanceDoctor={provenanceDoctor}
            provenanceAction={provenanceAction}
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
          activeWorkflowView={activeWorkflowView}
          onClose={() => setConversationOpen(false)}
          onSelectDocument={setSelectedId}
          onWorkflowViewChange={setActiveWorkflowView}
        />
      ) : null}
    </main>
  );
}

function LibraryToolbar({
  filterOpen,
  kindFilter,
  libraryItems,
  onCopyWorkspaceUrl,
  onSetKindFilter,
  onToggleFilters,
  query,
  reviewCount,
  setQuery
}) {
  return (
    <section className="library-toolbar" aria-label="Library search and filters">
      <label className="library-search">
        <span>Search library</span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="meeting, person, matter..."
        />
      </label>
      <div className="library-kind-controls" aria-label="Artifact kind">
        {LIBRARY_KIND_FILTERS.map((filter) => (
          <button
            aria-pressed={kindFilter === filter.id}
            className={kindFilter === filter.id ? "selected-filter" : ""}
            key={filter.id}
            onClick={() => onSetKindFilter(filter.id)}
            type="button"
          >
            <span>{filter.label}</span>
            <strong>{filterCount(libraryItems, filter.id)}</strong>
          </button>
        ))}
      </div>
      <div className="library-toolbar-actions">
        <button
          aria-expanded={filterOpen}
          className="toolbar-secondary"
          onClick={onToggleFilters}
          type="button"
        >
          {filterOpen ? "Hide filters" : "Filters"}
          <strong>{reviewCount}</strong>
        </button>
        <button className="share-link-button" onClick={onCopyWorkspaceUrl} type="button">
          Copy workspace link
        </button>
      </div>
    </section>
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
  latestSmokeJob,
  provenanceStatus
}) {
  const target =
    activeNav === "Intelligence"
      ? "Queue a smoke, inspect the tail, then verify the latest report."
      : activeNav === "Provenance"
        ? `Shared provenance config is ${provenanceStatus || "unknown"}.`
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
        <strong>{activeNav === "Provenance" ? provenanceStatus || "unknown" : latestSmokeJob?.status || latestSmoke?.status || "none"}</strong>
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

function SettingsPanel({
  activeSettingsSection,
  automation,
  automationAction,
  automationDraft,
  configAction,
  health,
  intelligence,
  provenance,
  provenanceAction,
  provenanceDoctor,
  provenanceDraft,
  profileDraft,
  selectedProfile,
  selectedProfileConfig,
  selectedTask,
  selectedTaskConfig,
  setAutomationAction,
  setActiveSettingsSection,
  setConfigAction,
  setProfileDraft,
  setProvenanceAction,
  setProvenanceDraft,
  setSelectedProfile,
  setSelectedTask,
  setTaskDraft,
  taskDraft,
  onApplyIntelligence,
  onApplyAutomation,
  onApplyProvenance,
  onCreateProfile,
  onDeleteProfile,
  onDuplicateProfile,
  onOpenIntelligence,
  onOpenProvenance,
  onPreviewIntelligence,
  onPreviewAutomation,
  onPreviewProvenance,
  onRefreshProvenance,
  setAutomationDraft
}) {
  const [activeSection, setActiveSection] = useState(activeSettingsSection || "account");
  const stages = automationStageEntries(automation);
  const modeChoices = automation?.mode_choices || ["manual", "one_click", "automatic"];
  const intelligenceConfig = intelligence?.config || {};
  const taskEntries = Object.entries(intelligenceConfig?.tasks || {});
  const profileEntries = intelligenceProfileEntries(intelligenceConfig);
  const profileMap = intelligenceConfig.profiles || {};
  const defaultProfileIds = new Set(intelligenceConfig.default_profile_ids || []);
  const providerList = intelligence?.providers?.providers || [];
  const sources = provenanceSourceEntries(provenance);
  const provenanceCounts = provenanceSourceCounts(provenance);
  const enabledSourceCount = sources.filter(([, source]) => source.enabled !== false).length;
  const provenanceStatus = provenanceDoctor?.status || (provenance?.exists ? "unknown" : "missing");
  const latestValidation = intelligence?.smokes?.latest_report || null;
  const enabledCount = Object.values(automationDraft.stages || {}).filter((stage) => stage.enabled).length;
  const automationDirty = automationDraftDirty(automationDraft, automation);
  const intelligenceDirty = intelligenceDraftDirty(taskDraft, selectedTaskConfig);
  const profileDirty = profileDraftDirty(profileDraft, selectedProfileConfig);
  const provenanceDirty = provenanceDraftDirty(provenanceDraft, provenance);
  const dirtySections = [
    automationDirty ? "Automation" : "",
    intelligenceDirty || profileDirty ? "Intelligence" : "",
    provenanceDirty ? "Provenance" : ""
  ].filter(Boolean);
  const selectedProfileProvider = providerList.find((provider) => provider.id === profileDraft.provider) || null;
  const auraCallReadiness = intelligence?.auracall_readiness || intelligence?.config?.auracall_readiness || {};
  const auraCallAgents = auracallAgentOptions(intelligence);
  const selectedAuraCallAgentId = agentIdFromModel(profileDraft.model);
  const selectedAuraCallAgent = auraCallAgents.find((agent) => agent.id === selectedAuraCallAgentId) || null;
  const auraCallAgentChoices = selectedAuraCallAgentId && !selectedAuraCallAgent
    ? [
        {
          id: selectedAuraCallAgentId,
          label: selectedAuraCallAgentId,
          ready: false,
          settings_description: "Configured agent; the AuraCall choices API did not return current settings for it."
        },
        ...auraCallAgents
      ]
    : auraCallAgents;
  const selectedAuraCallAgentDetail = selectedAuraCallAgent || auraCallAgentChoices.find((agent) => agent.id === selectedAuraCallAgentId) || null;
  const showAuraCallAgentSelector = shouldUseAuraCallAgentSelector(profileDraft, selectedProfileProvider, auraCallAgents);
  const isNewProfile = Boolean(selectedProfile && !profileMap[selectedProfile]);
  const selectedProfileInUse = taskEntries.some(([, task]) => task.profile === selectedProfile);
  const canDeleteProfile = Boolean(selectedProfile && !isNewProfile && !defaultProfileIds.has(selectedProfile) && !selectedProfileInUse);
  const showBaseUrlField = ["openai-compatible", "auracall"].includes(profileDraft.provider);
  const showTemperatureField = ["openai-compatible", "auracall"].includes(profileDraft.provider);
  const showModelField = !showAuraCallAgentSelector && profileDraft.provider !== "codex-app-server";
  const providerProfileNote = {
    "openai-compatible": "OpenAI-compatible API profiles use model, optional base URL, timeout, and temperature.",
    auracall: "AuraCall profiles select a runtime-advertised agent when available, with API transport read from runtime settings.",
    "codex-exec": "Codex exec profiles dispatch local CLI work; model and base URL are usually left to the Codex runtime.",
    "codex-app-server": "Codex app-server profiles use the ledger-backed local supervisor; provider defaults usually own model selection."
  }[profileDraft.provider] || "Provider-specific settings appear when the provider exposes configurable fields.";
  const activeProfileName = automationDraft.profile || provenanceDraft.activeProfile || "default";
  const settingsState = dirtySections.length
    ? `${dirtySections.length} staged`
    : configAction.preview || automationAction.preview || provenanceAction.preview
      ? "preview ready"
      : automation.exists || provenance.exists
        ? "saved"
        : "defaults";
  const sectionItems = [
    { id: "account", label: "Overview", meta: settingsState },
    { id: "intelligence", label: "Intelligence", meta: `${profileEntries.length} profiles` },
    { id: "automation", label: "Automation", meta: `${enabledCount} enabled` },
    { id: "provenance", label: "Provenance", meta: `${enabledSourceCount} sources` },
    { id: "safety", label: "Safety", meta: dirtySections.length ? `${dirtySections.length} staged` : "clear" },
    { id: "validation", label: "Validation", meta: latestValidation?.status || provenanceStatus }
  ];
  const updateStage = (stageId, changes) => {
    setAutomationAction({ status: "idle", message: "", preview: null });
    setAutomationDraft((current) => ({
      ...current,
      stages: {
        ...(current.stages || {}),
        [stageId]: {
          ...(current.stages?.[stageId] || {}),
          ...changes
        }
      }
    }));
  };
  const selectSection = (sectionId) => {
    setActiveSection(sectionId);
    setActiveSettingsSection?.(sectionId);
  };
  useEffect(() => {
    if (activeSettingsSection && activeSettingsSection !== activeSection) {
      setActiveSection(activeSettingsSection);
    }
  }, [activeSettingsSection]);
  const updateTaskDraft = (changes) => {
    setConfigAction({ status: "idle", message: "", preview: null });
    setTaskDraft((draft) => ({ ...draft, ...changes }));
  };
  const updateProfileDraft = (changes) => {
    setConfigAction({ status: "idle", message: "", preview: null });
    setProfileDraft((draft) => ({ ...draft, ...changes }));
  };
  const updateSourceEnabled = (sourceId, checked) => {
    setProvenanceAction({ status: "idle", message: "", preview: null });
    setProvenanceDraft((current) => ({
      ...current,
      sourceEnabled: {
        ...(current.sourceEnabled || {}),
        [sourceId]: checked
      }
    }));
  };
  const updateProvenanceDraft = (changes) => {
    setProvenanceAction({ status: "idle", message: "", preview: null });
    setProvenanceDraft((current) => ({ ...current, ...changes }));
  };
  const discardDrafts = () => {
    setAutomationDraft(automationDraftFromConfig(automation));
    setTaskDraft(taskDraftFromConfig(selectedTaskConfig));
    setProfileDraft(profileDraftFromConfig(selectedProfileConfig));
    setProvenanceDraft(resetProvenanceDraftFromConfig(provenance));
    setAutomationAction({ status: "idle", message: "", preview: null });
    setConfigAction({ status: "idle", message: "", preview: null });
    setProvenanceAction({ status: "idle", message: "", preview: null });
  };
  const showDraftBar = Boolean(
    dirtySections.length
    || configAction.preview
    || automationAction.preview
    || provenanceAction.preview
  );
  return (
    <div className="settings-workbench">
      {showDraftBar && (
        <section className={dirtySections.length ? "settings-dirty-bar dirty" : "settings-dirty-bar"} aria-label="Staged config changes">
          <div>
            <strong>{dirtySections.length ? `${dirtySections.length} staged section${dirtySections.length === 1 ? "" : "s"}` : "Preview ready"}</strong>
            <span>{dirtySections.length ? dirtySections.join(", ") : "Review or apply the prepared config update."}</span>
          </div>
          <div className="settings-dirty-actions">
            {(intelligenceDirty || profileDirty) && <button onClick={onPreviewIntelligence} disabled={configAction.status === "running"} type="button">Preview intelligence</button>}
            {automationDirty && <button onClick={onPreviewAutomation} disabled={automationAction.status === "running"} type="button">Preview automation</button>}
            {provenanceDirty && <button onClick={onPreviewProvenance} disabled={provenanceAction.status === "running"} type="button">Preview provenance</button>}
            {configAction.preview && <button onClick={onApplyIntelligence} disabled={configAction.status === "applying"} type="button">Apply intelligence</button>}
            {automationAction.preview && <button onClick={onApplyAutomation} disabled={automationAction.status === "applying"} type="button">Apply automation</button>}
            {provenanceAction.preview && <button onClick={onApplyProvenance} disabled={provenanceAction.status === "applying"} type="button">Apply provenance</button>}
            <button onClick={discardDrafts} disabled={!dirtySections.length} type="button">Discard draft</button>
          </div>
        </section>
      )}

      <div className="settings-workbench-body">
        <nav className="settings-section-rail" aria-label="Settings sections">
          {sectionItems.map((item) => (
            <button
              aria-pressed={activeSection === item.id}
              className={activeSection === item.id ? "active" : ""}
              key={item.id}
              onClick={() => selectSection(item.id)}
              type="button"
            >
              <span>{item.label}</span>
              <strong>{item.meta}</strong>
            </button>
          ))}
        </nav>

        <section className="settings-section-surface">
          {activeSection === "account" && (
            <div className="settings-account-page">
              <section className="settings-compact-section">
                <div className="settings-section-title">
                  <div>
                    <p className="eyebrow">Settings</p>
                    <h2>Configuration overview</h2>
                  </div>
                  <span className={dirtySections.length ? "settings-state-pill dirty" : "settings-state-pill"}>{settingsState}</span>
                </div>

                <div className="settings-overview-grid">
                  <article className="settings-overview-card">
                    <div>
                      <p className="eyebrow">Intelligence</p>
                      <h3>{profileEntries.length} profiles</h3>
                      <span>{taskEntries.length} task routes · {selectedProfile || "default"} selected</span>
                    </div>
                    <button onClick={() => selectSection("intelligence")} type="button">Configure</button>
                  </article>
                  <article className="settings-overview-card">
                    <div>
                      <p className="eyebrow">Automation</p>
                      <h3>{enabledCount} / {stages.length} enabled</h3>
                      <span>{automation.exists ? "saved" : "defaults"} · {activeProfileName}</span>
                    </div>
                    <button onClick={() => selectSection("automation")} type="button">Configure</button>
                  </article>
                  <article className="settings-overview-card">
                    <div>
                      <p className="eyebrow">Provenance</p>
                      <h3>{enabledSourceCount} / {sources.length} sources</h3>
                      <span>{provenanceStatus}</span>
                    </div>
                    <button onClick={() => selectSection("provenance")} type="button">Configure</button>
                  </article>
                  <article className="settings-overview-card">
                    <div>
                      <p className="eyebrow">Validation</p>
                      <h3>{latestValidation?.status || provenanceStatus}</h3>
                      <span>{latestValidation ? "latest evidence available" : "no browser evidence"}</span>
                    </div>
                    <button onClick={() => selectSection("validation")} type="button">Open</button>
                  </article>
                </div>
              </section>

              <section className="settings-compact-section settings-profile-strip">
                <div className="settings-section-title compact">
                  <div>
                    <p className="eyebrow">Profiles</p>
                    <h2>{activeProfileName}</h2>
                  </div>
                  <button onClick={() => selectSection("safety")} type="button">Safety</button>
                </div>
                <div className="settings-two-column-form">
                  <label>
                    <span>Runtime profile</span>
                    <input value={automationDraft.profile || "default"} onChange={(event) => {
                      setAutomationAction({ status: "idle", message: "", preview: null });
                      setAutomationDraft((current) => ({ ...current, profile: event.target.value }));
                    }} />
                  </label>
                  <label>
                    <span>Provenance profile</span>
                    <input value={provenanceDraft.activeProfile || "default"} onChange={(event) => updateProvenanceDraft({ activeProfile: event.target.value })} />
                  </label>
                </div>
              </section>
              <details className="settings-discrete-details">
                <summary>Runtime paths and facts</summary>
                <dl className="settings-detail-list compact">
                  <dt>Automation config</dt>
                  <dd>{automation.exists ? "saved" : "defaults"}</dd>
                  <dt>Provenance doctor</dt>
                  <dd>{provenanceStatus}</dd>
                  <dt>Task routes</dt>
                  <dd>{taskEntries.length}</dd>
                </dl>
                <div className="settings-path-list compact">
                  <code>{health.store_dir || "store unavailable"}</code>
                  <code>{intelligenceConfig?.config_path || "intelligence config defaults"}</code>
                  <code>{automation.config_path || "automation config defaults"}</code>
                  <code>{provenance?.config_path || "provenance config defaults"}</code>
                </div>
              </details>
            </div>
          )}

          {activeSection === "intelligence" && (
            <div className="settings-intelligence-page">
              <section className="settings-compact-section">
                <div className="settings-section-title">
                  <div>
                    <p className="eyebrow">Profiles</p>
                    <h2>{profileDraft.label || selectedProfile}</h2>
                  </div>
                  <div className="settings-profile-actions">
                    <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value)}>
                      {isNewProfile ? <option value={selectedProfile}>{profileDraft.label || selectedProfile} (new)</option> : null}
                      {profileEntries.map(([profileId, profile]) => (
                        <option key={profileId} value={profileId}>{profile.label || profileId}</option>
                      ))}
                    </select>
                    <button onClick={() => onCreateProfile("openai-compatible")} type="button">New</button>
                    <button onClick={onDuplicateProfile} type="button">Duplicate</button>
                    <button
                      disabled={!canDeleteProfile}
                      onClick={onDeleteProfile}
                      title={
                        defaultProfileIds.has(selectedProfile)
                          ? "Default profiles are protected."
                          : selectedProfileInUse
                            ? "Profiles assigned to components cannot be deleted."
                            : "Delete this custom profile."
                      }
                      type="button"
                    >
                      Delete
                    </button>
                  </div>
                </div>
                <div className="settings-profile-meta">
                  <span>ID: <code>{selectedProfile}</code>{isNewProfile ? " · new, not saved" : ""}</span>
                  <span>{providerProfileNote}</span>
                </div>
                <div className="settings-two-column-form">
                  <label>
                    <span>Label</span>
                    <input value={profileDraft.label} onChange={(event) => updateProfileDraft({ label: event.target.value })} />
                  </label>
                  <label>
                    <span>Provider</span>
                    <select
                      value={profileDraft.provider}
                      onChange={(event) => {
                        const provider = event.target.value;
                        const defaults = defaultProfileDraft(provider);
                        updateProfileDraft({
                          provider,
                          model: defaults.model,
                          base_url: defaults.base_url,
                          timeout: defaults.timeout,
                          temperature: defaults.temperature
                        });
                      }}
                    >
                      {[...new Set([profileDraft.provider, "openai-compatible", "auracall", "codex-exec", "codex-app-server", ...providerList.map((provider) => provider.id)])].filter(Boolean).map((providerId) => (
                        <option key={providerId} value={providerId}>{providerId}</option>
                      ))}
                    </select>
                  </label>
                  {showAuraCallAgentSelector ? (
                    <label>
                      <span>AuraCall agent</span>
                      <select
                        value={selectedAuraCallAgentId}
                        onChange={(event) => updateProfileDraft({ model: agentModelFromId(event.target.value) })}
                        disabled={!auraCallAgentChoices.length}
                      >
                        <option value="">{auraCallAgentChoices.length ? "Select agent" : "No agents returned"}</option>
                        {auraCallAgentChoices.map((agent) => (
                          <option key={agent.id} value={agent.id}>
                            {agent.label || agent.id}{agent.ready ? "" : " (not ready)"}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : showModelField ? (
                    <label>
                      <span>Model</span>
                      <input value={profileDraft.model} onChange={(event) => updateProfileDraft({ model: event.target.value })} placeholder="provider default" />
                    </label>
                  ) : null}
                  {showBaseUrlField ? (
                    <label>
                      <span>Base URL</span>
                      <input value={profileDraft.base_url} onChange={(event) => updateProfileDraft({ base_url: event.target.value })} placeholder="env/provider default" />
                    </label>
                  ) : null}
                  <label>
                    <span>Timeout</span>
                    <input type="number" value={profileDraft.timeout} onChange={(event) => updateProfileDraft({ timeout: event.target.value })} />
                  </label>
                  {showTemperatureField ? (
                    <label>
                      <span>Temperature</span>
                      <input type="number" step="0.1" value={profileDraft.temperature} onChange={(event) => updateProfileDraft({ temperature: event.target.value })} />
                    </label>
                  ) : null}
                  <label className="wide-field">
                    <span>Description</span>
                    <input value={profileDraft.description} onChange={(event) => updateProfileDraft({ description: event.target.value })} />
                  </label>
                </div>
                {showAuraCallAgentSelector && (
                  <div className="settings-agent-summary">
                    <div>
                      <strong>{selectedAuraCallAgentDetail?.label || selectedAuraCallAgentId || "No AuraCall agent selected"}</strong>
                      <span>{selectedAuraCallAgentDetail ? auraCallAgentDescription(selectedAuraCallAgentDetail) : "Select one of the runtime-advertised agents for this profile."}</span>
                    </div>
                    <dl className="settings-detail-list compact">
                      <dt>Choices API</dt>
                      <dd>{auraCallReadiness.source?.fetched ? "available" : "unavailable"}</dd>
                      <dt>Agents</dt>
                      <dd>{auraCallReadiness.counts?.agents ?? auraCallAgents.length}</dd>
                      <dt>Dispatch team</dt>
                      <dd>{auraCallReadiness.dispatch_team || "none"}</dd>
                      <dt>Selected model</dt>
                      <dd>{profileDraft.model || "none"}</dd>
                    </dl>
                  </div>
                )}
              </section>

              <section className="settings-compact-section">
                <div className="settings-section-title">
                  <div>
                    <p className="eyebrow">Component profile selections</p>
                    <h2>{statusLabel(selectedTask || "task")}</h2>
                  </div>
                  <button onClick={() => selectSection("validation")} type="button">Validation</button>
                </div>
                <div className="settings-route-matrix">
                  <div className="settings-route-matrix-heading">
                    <span>Component</span>
                    <span>Profile</span>
                    <span>Policy</span>
                  </div>
                  {taskEntries.map(([task, route]) => (
                    <button className={task === selectedTask ? "settings-route-row active" : "settings-route-row"} key={task} onClick={() => setSelectedTask(task)} type="button">
                      <strong>{statusLabel(task)}</strong>
                      <span>{route.profile || intelligenceConfig.task_profiles?.[task] || "unassigned"}</span>
                      <small>{route.requires_ledger ? "ledger" : "direct"} · {route.human_review || "review policy"}</small>
                    </button>
                  ))}
                </div>
                <div className="settings-two-column-form compact">
                  <label>
                    <span>Selected component</span>
                    <select value={selectedTask} onChange={(event) => setSelectedTask(event.target.value)}>
                      {taskEntries.map(([task]) => <option key={task} value={task}>{statusLabel(task)}</option>)}
                    </select>
                  </label>
                  <label>
                    <span>Uses profile</span>
                    <select value={taskDraft.profile || selectedTaskConfig?.profile || ""} onChange={(event) => updateTaskDraft({ profile: event.target.value, provider: "", model: "", timeout: "", temperature: "" })}>
                      {profileEntries.map(([profileId, profile]) => (
                        <option key={profileId} value={profileId}>{profile.label || profileId}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Fallbacks</span>
                    <input value={taskDraft.fallbacks} onChange={(event) => updateTaskDraft({ fallbacks: event.target.value })} />
                  </label>
                  <label>
                    <span>Human review</span>
                    <input value={taskDraft.human_review} onChange={(event) => updateTaskDraft({ human_review: event.target.value })} />
                  </label>
                  <label className="checkbox-line">
                    <input type="checkbox" checked={taskDraft.requires_ledger} onChange={(event) => updateTaskDraft({ requires_ledger: event.target.checked })} />
                    <span>Requires run ledger</span>
                  </label>
                </div>
              </section>

              <details className="settings-discrete-details">
                <summary>Config facts and resolved route</summary>
                <dl className="settings-detail-list">
                  <dt>Config path</dt>
                  <dd>{intelligenceConfig?.config_path || "defaults"}</dd>
                  <dt>Selected provider</dt>
                  <dd>{selectedProfileProvider?.label || profileDraft.provider || "unknown"}</dd>
                  <dt>Resolved task provider</dt>
                  <dd>{selectedTaskConfig?.provider || "unknown"}</dd>
                  <dt>Resolved task model</dt>
                  <dd>{selectedTaskConfig?.model || "provider default"}</dd>
                  <dt>Resolved source</dt>
                  <dd>{selectedTaskConfig?.source || "defaults"}</dd>
                </dl>
              </details>

              <div className="notice-actions">
                <button onClick={onPreviewIntelligence} disabled={(!intelligenceDirty && !profileDirty) || configAction.status === "running"} type="button">Preview intelligence update</button>
                <button onClick={onApplyIntelligence} disabled={!configAction.preview || configAction.status === "applying"} type="button">Apply with approval</button>
              </div>
              {configAction.message && <div className={`action-notice ${configAction.status}`}><strong>{configAction.message}</strong></div>}
            </div>
          )}

          {activeSection === "automation" && (
            <div className="settings-panel-block">
              <p className="eyebrow">Automation</p>
              <h2>{enabledCount} enabled stages</h2>
              <div className="automation-stage-table settings-automation-table">
                {stages.map(([stageId, stage]) => {
                  const draft = automationDraft.stages?.[stageId] || {};
                  const capabilities = stage.capabilities || {};
                  return (
                    <article className={draft.enabled ? "automation-stage-row enabled" : "automation-stage-row"} key={stageId}>
                      <label className="checkbox-line">
                        <input
                          checked={Boolean(draft.enabled)}
                          onChange={(event) => updateStage(stageId, { enabled: event.target.checked })}
                          type="checkbox"
                        />
                        <span>{stage.label || statusLabel(stageId)}</span>
                      </label>
                      <select value={draft.mode || "manual"} onChange={(event) => updateStage(stageId, { mode: event.target.value })}>
                        {modeChoices.map((mode) => (
                          <option key={`${stageId}-${mode}`} value={mode}>{statusLabel(mode)}</option>
                        ))}
                      </select>
                      <label className="checkbox-line compact">
                        <input
                          checked={draft.requires_review !== false}
                          onChange={(event) => updateStage(stageId, { requires_review: event.target.checked })}
                          type="checkbox"
                        />
                        <span>Review</span>
                      </label>
                      <span className={capabilities.automatic_available ? "risk-badge read-only" : "risk-badge write-bearing"}>
                        {capabilities.automatic_available ? "auto-ready" : capabilities.one_click_available ? "one-click" : "manual"}
                      </span>
                    </article>
                  );
                })}
              </div>
              <div className="notice-actions">
                <button onClick={onPreviewAutomation} disabled={!automationDirty || automationAction.status === "running"} type="button">Preview automation update</button>
                <button onClick={onApplyAutomation} disabled={!automationAction.preview || automationAction.status === "applying"} type="button">Apply with approval</button>
              </div>
              {automationAction.message && (
                <div className={`action-notice ${automationAction.status}`}>
                  <strong>{automationAction.message}</strong>
                  {automationAction.preview && (
                    <code>{JSON.stringify({
                      will_write: automationAction.preview.will_write,
                      will_execute_workflow_stage: automationAction.preview.will_execute_workflow_stage,
                      requires_approval_token: automationAction.preview.requires_approval_token
                    }, null, 2)}</code>
                  )}
                </div>
              )}
            </div>
          )}

          {activeSection === "provenance" && (
            <div className="settings-section-grid">
              <div className="settings-panel-block">
                <p className="eyebrow">Provenance</p>
                <h2>{enabledSourceCount} enabled sources</h2>
                <div className="source-toggle-list">
                  {sources.map(([sourceId, source]) => {
                    const checked = provenanceDraft.sourceEnabled?.[sourceId] ?? source.enabled !== false;
                    return (
                      <label className="source-toggle-row" key={sourceId}>
                        <input checked={checked} onChange={(event) => updateSourceEnabled(sourceId, event.target.checked)} type="checkbox" />
                        <span>
                          <strong>{source.label || sourceId}</strong>
                          <small>{sourceId}</small>
                        </span>
                        <em>{statusLabel(source.kind)}</em>
                      </label>
                    );
                  })}
                  {!sources.length && <p className="muted">No provenance sources are configured.</p>}
                </div>
                <div className="notice-actions">
                  <button onClick={onPreviewProvenance} disabled={!provenanceDirty || provenanceAction.status === "running"} type="button">Preview provenance update</button>
                  <button onClick={onApplyProvenance} disabled={!provenanceAction.preview || provenanceAction.status === "applying"} type="button">Apply with approval</button>
                  <button onClick={onRefreshProvenance} disabled={provenanceAction.status === "refreshing"} type="button">Refresh config</button>
                  <button onClick={onOpenProvenance} type="button">Open Provenance tab</button>
                </div>
                {provenanceAction.message && <div className={`action-notice ${provenanceAction.status}`}><strong>{provenanceAction.message}</strong></div>}
              </div>
              <div className="settings-panel-block">
                <p className="eyebrow">iCal draft</p>
                <h2>{provenanceDraft.newIcalLabel || "Add feed"}</h2>
                <div className="editor-grid">
                  <label>
                    <span>Source id</span>
                    <input value={provenanceDraft.newIcalId} onChange={(event) => updateProvenanceDraft({ newIcalId: event.target.value })} placeholder="ical-saber-zoho" />
                  </label>
                  <label>
                    <span>Label</span>
                    <input value={provenanceDraft.newIcalLabel} onChange={(event) => updateProvenanceDraft({ newIcalLabel: event.target.value })} placeholder="SABER Zoho" />
                  </label>
                  <label className="wide-field">
                    <span>URL or env ref</span>
                    <input value={provenanceDraft.newIcalUrl} onChange={(event) => updateProvenanceDraft({ newIcalUrl: event.target.value })} placeholder="env:SABER_ICAL_URL" />
                  </label>
                </div>
                <div className="settings-chip-row">
                  {Object.entries(provenanceCounts).map(([kind, count]) => <span className="status-chip" key={kind}>{statusLabel(kind)} {count}</span>)}
                </div>
              </div>
            </div>
          )}

          {activeSection === "safety" && (
            <div className="settings-section-grid">
              <div className="settings-panel-block">
                <p className="eyebrow">Safety gates</p>
                <h2>{dirtySections.length ? `${dirtySections.length} staged changes` : "Clear"}</h2>
                <div className="settings-safety-list">
                  <article>
                    <strong>Intelligence apply</strong>
                    <span>APPLY_INTELLIGENCE_CONFIG_UPDATE</span>
                    <small>Writes user-scoped config only; no model turn is sent.</small>
                  </article>
                  <article>
                    <strong>Automation apply</strong>
                    <span>APPLY_AUTOMATION_CONFIG_UPDATE</span>
                    <small>Must return will_execute_workflow_stage=false.</small>
                  </article>
                  <article>
                    <strong>Provenance apply</strong>
                    <span>APPLY_PROVENANCE_CONFIG_UPDATE</span>
                    <small>Writes config only; no contact/calendar refresh starts.</small>
                  </article>
                </div>
              </div>
              <div className="settings-panel-block">
                <p className="eyebrow">Preview flags</p>
                <h2>Latest safety evidence</h2>
                <div className="settings-preview-stack">
                  <code>{JSON.stringify({
                    automation_preview: automationAction.preview ? {
                      will_write: automationAction.preview.will_write,
                      will_execute_workflow_stage: automationAction.preview.will_execute_workflow_stage,
                      requires_approval_token: automationAction.preview.requires_approval_token
                    } : "not previewed",
                    intelligence_preview: configAction.preview ? {
                      will_write: configAction.preview.will_write,
                      requires_approval_token: configAction.preview.requires_approval_token
                    } : "not previewed",
                    provenance_preview: provenanceAction.preview ? {
                      will_write: provenanceAction.preview.will_write,
                      requires_approval_token: provenanceAction.preview.requires_approval_token
                    } : "not previewed"
                  }, null, 2)}</code>
                </div>
              </div>
            </div>
          )}

          {activeSection === "validation" && (
            <div className="settings-section-grid">
              <div className="settings-panel-block">
                <p className="eyebrow">Validation</p>
                <h2>{latestValidation ? `${latestValidation.status || "unknown"} latest validation` : "No validation evidence"}</h2>
                <dl className="settings-detail-list">
                  <dt>Provenance doctor</dt>
                  <dd>{provenanceStatus}</dd>
                  <dt>Validation report</dt>
                  <dd>{latestValidation ? "available" : "not yet run"}</dd>
                  <dt>Browser check screenshot</dt>
                  <dd>{latestValidation?.screenshot_exists ? "available" : "not yet run"}</dd>
                </dl>
                <details className="settings-discrete-details inline-details">
                  <summary>Artifact paths</summary>
                  <dl className="settings-detail-list">
                    <dt>Validation report path</dt>
                    <dd>{latestValidation?.path || "not yet run"}</dd>
                    <dt>Browser check screenshot path</dt>
                    <dd>{latestValidation?.screenshot_exists ? latestValidation.screenshot_path : "not yet run"}</dd>
                    <dt>Baseline desktop</dt>
                    <dd>~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-desktop.png</dd>
                    <dt>Baseline mobile</dt>
                    <dd>~/.local/state/transcribe-audio/browser-smokes/plan-0016-config-panel-baseline-mobile.png</dd>
                  </dl>
                </details>
              </div>
              <div className="settings-panel-block">
                <p className="eyebrow">Readiness</p>
                <h2>Config surface</h2>
                <div className="settings-chip-row">
                  <span className="status-chip">{taskEntries.length} intelligence routes</span>
                  <span className="status-chip">{stages.length} automation stages</span>
                  <span className="status-chip">{sources.length} provenance sources</span>
                  <span className="status-chip">{provenanceStatus}</span>
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function ProvenancePanel({
  provenance,
  doctor,
  draft,
  setDraft,
  action,
  onPreview,
  onApply,
  onRefresh
}) {
  const sources = provenanceSourceEntries(provenance);
  const profiles = Object.keys(provenance?.config?.profiles || {});
  const profileOptions = [...new Set([draft.activeProfile, provenance?.profile, provenance?.config?.active_profile, ...profiles, "default"].filter(Boolean))];
  const calendar = provenance?.calendar_metadata || {};
  const contactConfig = provenance?.contact_source_config || {};
  return (
    <div className="provenance-grid">
      <section className="intelligence-card provenance-overview">
        <p className="eyebrow">Profile</p>
        <h2>{draft.activeProfile || "default"}</h2>
        <div className="editor-grid">
          <label>
            <span>Active profile</span>
            <select
              value={draft.activeProfile}
              onChange={(event) => setDraft((current) => ({ ...current, activeProfile: event.target.value }))}
            >
              {profileOptions.map((profile) => <option key={profile} value={profile}>{profile}</option>)}
            </select>
          </label>
          <label>
            <span>Doctor status</span>
            <input readOnly value={doctor?.status || "unknown"} />
          </label>
        </div>
        <div className="source-toggle-list">
          {sources.map(([sourceId, source]) => {
            const checked = draft.sourceEnabled?.[sourceId] ?? source.enabled !== false;
            return (
              <label className="source-toggle-row" key={sourceId}>
                <input
                  checked={checked}
                  onChange={(event) => setDraft((current) => ({
                    ...current,
                    sourceEnabled: {
                      ...current.sourceEnabled,
                      [sourceId]: event.target.checked
                    }
                  }))}
                  type="checkbox"
                />
                <span>
                  <strong>{source.label || sourceId}</strong>
                  <small>{sourceId}</small>
                </span>
                <em>{statusLabel(source.kind)}</em>
              </label>
            );
          })}
          {!sources.length && <p className="muted">No provenance sources are configured.</p>}
        </div>
        <div className="notice-actions">
          <button onClick={onPreview} disabled={action.status === "running"} type="button">Preview update</button>
          <button onClick={onApply} disabled={!action.preview || action.status === "applying"} type="button">Apply with approval</button>
          <button onClick={onRefresh} disabled={action.status === "refreshing"} type="button">Refresh</button>
        </div>
        {action.message && <div className={`action-notice ${action.status}`}><strong>{action.message}</strong></div>}
      </section>

      <section className="intelligence-card provenance-ical-editor">
        <p className="eyebrow">iCal Calendar</p>
        <h2>Add feed</h2>
        <div className="editor-grid">
          <label>
            <span>Source id</span>
            <input
              value={draft.newIcalId}
              onChange={(event) => setDraft((current) => ({ ...current, newIcalId: event.target.value }))}
              placeholder="ical-saber-zoho"
            />
          </label>
          <label>
            <span>Label</span>
            <input
              value={draft.newIcalLabel}
              onChange={(event) => setDraft((current) => ({ ...current, newIcalLabel: event.target.value }))}
              placeholder="SABER Zoho"
            />
          </label>
          <label className="wide-field">
            <span>URL or env ref</span>
            <input
              value={draft.newIcalUrl}
              onChange={(event) => setDraft((current) => ({ ...current, newIcalUrl: event.target.value }))}
              placeholder="env:SABER_ICAL_URL"
            />
          </label>
        </div>
        <div className="provenance-feed-list">
          {(calendar.provenance_ical_urls || []).map((feed) => (
            <article key={feed}>
              <strong>{feed}</strong>
              <span>configured</span>
            </article>
          ))}
          {!(calendar.provenance_ical_urls || []).length && <p className="muted">No iCal feeds are active in this profile.</p>}
        </div>
      </section>

      <section className="intelligence-card provenance-calendar-map">
        <p className="eyebrow">Calendar Metadata</p>
        <h2>{calendar.primary_calendar_id || "primary"}</h2>
        <div className="task-table">
          {(calendar.provider_configs || []).map((provider, index) => (
            <article className="task-row" key={`${provider.name}-${index}`}>
              <strong>{provider.name}</strong>
              <span>{provider.account || provider.config_dir || "default"}</span>
              <small>{provider.client || "calendar provider"}</small>
            </article>
          ))}
          {(calendar.provenance_calendar_ids || []).map((calendarId) => (
            <article className="task-row" key={calendarId}>
              <strong>{calendarId}</strong>
              <span>shared calendar</span>
              <small>resolved from provenance config</small>
            </article>
          ))}
        </div>
        {calendar.warnings?.length ? (
          <div className="warning-list">
            {calendar.warnings.map((warning) => <span key={warning}>{warning}</span>)}
          </div>
        ) : null}
      </section>

      <section className="intelligence-card provenance-contact-map">
        <p className="eyebrow">Contact Sources</p>
        <h2>Participant identity</h2>
        <div className="task-table">
          {Object.entries(contactConfig).flatMap(([kind, config]) => (
            (config?.profiles || []).map((profile) => (
              <article className="task-row" key={`${kind}-${profile.label || profile.display_label}`}>
                <strong>{profile.display_label || profile.label}</strong>
                <span>{kind}</span>
                <small>{profile.surfaces?.join(", ") || profile.repo_root || profile.config_dir || "configured"}</small>
              </article>
            ))
          ))}
          {!Object.keys(contactConfig).length && <p className="muted">No contact sources are active in this profile.</p>}
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
  const tableMinWidth = Object.values(columnWidths).reduce((sum, width) => sum + width, 0);
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
      <table className="conversation-table" style={{ minWidth: `${tableMinWidth}px` }}>
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

function ReviewQueue({ queue, reviewAction, batchManifests, onPrepareFirstPass, onSubmitFirstPass, onRefreshFirstPass, onSelectFirstPassManifest, humanReviewAction, onRecordHumanReview, onOpenQueueConversation }) {
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
              <span>
                {item.type === "app_intelligence_human_review"
                  ? statusLabel(item.decision_status || item.status)
                  : item.type === "route_review"
                    ? item.route_decision_exists ? "route available" : "stale route reference"
                    : statusLabel(item.workflow_stage || item.status || "review")}
              </span>
              <code>{item.artifact_path || item.route_decision_path || item.review_path}</code>
              {item.document_id ? (
                <div className="notice-actions">
                  <button onClick={() => onOpenQueueConversation(item)} type="button">Open conversation</button>
                </div>
              ) : null}
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
  intelligence,
  provenance,
  provenanceDoctor,
  provenanceAction
}) {
  if (activeNav === "Provenance") {
    const preview = provenanceAction.preview;
    const sources = provenanceSourceEntries(provenance);
    return (
      <div className="inspector-content">
        <p className="eyebrow">Provenance Inspector</p>
        <h2>{provenance.profile || provenance.config?.active_profile || "default"}</h2>
        <dl>
          <dt>Status</dt>
          <dd>{provenanceDoctor?.status || "unknown"}</dd>
          <dt>Config path</dt>
          <dd>{provenance.config_path || "Unavailable"}</dd>
          <dt>Sources</dt>
          <dd>{sources.length} total · {sources.filter(([, source]) => source.enabled !== false).length} enabled</dd>
          <dt>Calendar IDs</dt>
          <dd>{provenance.calendar_metadata?.provenance_calendar_ids?.length || 0}</dd>
          <dt>iCal feeds</dt>
          <dd>{provenance.calendar_metadata?.provenance_ical_urls?.length || 0}</dd>
          <dt>Contact profiles</dt>
          <dd>
            {Object.entries(provenance.contact_source_config || {})
              .map(([kind, config]) => `${kind}: ${config?.profiles?.length || 0}`)
              .join(" · ") || "None"}
          </dd>
        </dl>
        {provenanceDoctor?.errors?.length ? (
          <div className="action-notice error">
            <strong>{provenanceDoctor.errors.join(" · ")}</strong>
          </div>
        ) : null}
        {preview ? (
          <div className="preview-card">
            <span>Preview</span>
            <strong>{preview.will_write ? "Apply response" : "No write preview"}</strong>
            <code>{JSON.stringify(preview.after || preview, null, 2)}</code>
          </div>
        ) : (
          <p className="muted">Preview a provenance edit to inspect the redacted config diff.</p>
        )}
      </div>
    );
  }

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
        <button disabled title="Scoped artifact sharing is planned but not wired yet." type="button">Prepare scoped artifact link (planned)</button>
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
  activeWorkflowView,
  onClose,
  onSelectDocument,
  onWorkflowViewChange
}) {
  const [retranscriptionBackend, setRetranscriptionBackend] = useState("faster_whisper");
  const [retranscriptionPreflight, setRetranscriptionPreflight] = useState({ status: "idle", message: "", payload: null });
  const [retranscriptionQueue, setRetranscriptionQueue] = useState({ status: "idle", message: "", payload: null });
  const [sourceDetail, setSourceDetail] = useState(null);
  const [sourceDetailAction, setSourceDetailAction] = useState({ status: "idle", message: "" });
  const [identityReview, setIdentityReview] = useState(conversationDetail?.identity_review || null);
  const [firstPassAction, setFirstPassAction] = useState({ status: "idle", message: "", payload: null });
  const [speakerReviewAction, setSpeakerReviewAction] = useState({ status: "idle", message: "", payload: null });
  const [speakerPreprocessing, setSpeakerPreprocessing] = useState(null);
  const [speakerPreprocessingAction, setSpeakerPreprocessingAction] = useState({ status: "idle", message: "", payload: null });
  const [speakerLocalAssignments, setSpeakerLocalAssignments] = useState({});
  const [speakerManualLabels, setSpeakerManualLabels] = useState({});
  const [joinedShadowDecisions, setJoinedShadowDecisions] = useState({});
  const [contextAction, setContextAction] = useState({ status: "idle", message: "", payload: null });
  const [contextContactAction, setContextContactAction] = useState({ status: "idle", message: "", payload: null });
  const [contextContactQuery, setContextContactQuery] = useState("");
  const [contextSearchAction, setContextSearchAction] = useState({ status: "idle", message: "", payload: null });
  const [contextAffinityAction, setContextAffinityAction] = useState({ status: "idle", message: "", payload: null });
  const [contextMergeAction, setContextMergeAction] = useState({ status: "idle", message: "", payload: null });
  const [contextManualContact, setContextManualContact] = useState({ label: "", email: "" });
  const [contextLocalSelection, setContextLocalSelection] = useState({
    selectedIds: [],
    excludedIds: [],
    pendingActions: {},
    candidatesById: {},
    dirty: false
  });
  const [contextInstructionDraft, setContextInstructionDraft] = useState("");
  const [contextInstructionAction, setContextInstructionAction] = useState({ status: "idle", message: "", payload: null });
  const [finalPreviewAction, setFinalPreviewAction] = useState({ status: "idle", message: "", payload: null });
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
  const activeIdentityReview = identityReview || conversationDetail?.identity_review || {};
  const acousticShadowEvidence = activeIdentityReview.acoustic_shadow_evidence || {};
  const joinedShadowEvidence = activeIdentityReview.joined_shadow_evidence || {};
  const stagedSpeakerCount = Object.keys(speakerLocalAssignments || {}).length;
  const speakersForDisplay = useMemo(
    () => (activeIdentityReview.speakers || []).map((speaker) => {
      const staged = speakerLocalAssignments[speaker.speaker_label];
      if (!staged) return speaker;
      return {
        ...speaker,
        status: staged.status,
        assignment: staged.assignment,
        review_required: false,
        staged: true
      };
    }),
    [activeIdentityReview.speakers, speakerLocalAssignments]
  );
  const speakerPendingCount = speakersForDisplay.filter((speaker) => speaker.review_required).length;
  const currentSpeakerEvaluation = speakerPreprocessing?.current_evaluation || null;
  const speakerIdentityProposals = currentSpeakerEvaluation?.proposals || [];
  const latestSpeakerProposalDecisions = useMemo(() => {
    const byProposal = {};
    (speakerPreprocessing?.review_decisions || []).forEach((decision) => {
      if (decision?.evaluation_id === currentSpeakerEvaluation?.evaluation_id && decision?.proposal_id) {
        byProposal[decision.proposal_id] = decision;
      }
    });
    return byProposal;
  }, [speakerPreprocessing?.review_decisions, currentSpeakerEvaluation?.evaluation_id]);
  const firstPassSummaryState = firstPassAction.payload?.first_pass_summary || conversationDetail?.first_pass_summary || {};
  const selectedFirstPassManifest = firstPassAction.payload?.manifest || "";
  const firstPassBusy = ["running", "submitting", "checking"].includes(firstPassAction.status);
  const firstPassPreparedOnly = Boolean(selectedFirstPassManifest && firstPassAction.payload?.dry_run);
  const firstPassPrimaryLabel = summaryText
    ? "Summary ready"
    : firstPassPreparedOnly
      ? "Submit prepared summary"
      : selectedFirstPassManifest
        ? "Check summary status"
        : "Run initial summary";
  const firstPassPrimaryDetail = summaryText
    ? "A stored first-pass summary is linked to this conversation."
    : firstPassPreparedOnly
      ? "Submit the already prepared scoped request to the configured provider."
      : selectedFirstPassManifest
        ? "Poll the submitted batch and materialize the readout if the provider has completed it."
        : "Prepare one scoped request and submit it with the selected first-pass provider route.";
  const firstPassExternalActionText = summaryText
    ? "none"
    : selectedFirstPassManifest && !firstPassPreparedOnly
      ? "status poll"
      : "provider submit";
  const contextWorkbench = contextInstructionAction.payload?.context_workbench || contextContactAction.payload?.context_workbench || contextAction.payload?.context_workbench || conversationDetail?.context_workbench || {};
  const identityBundle = activeIdentityReview.identity_bundle || contextWorkbench.participant_identity_bundle || {};
  const contactSelection = contextWorkbench.contact_selection || {};
  const operatorContext = contextWorkbench.operator_context || contextWorkbench.context_instructions || {};
  const proposedContextContacts = contextWorkbench.proposed_contact_candidates?.length
    ? contextWorkbench.proposed_contact_candidates
    : identityBundle.contact_candidates || [];
  const searchableContextContacts = contactSelection.searchable_candidates?.length
    ? contactSelection.searchable_candidates
    : proposedContextContacts;
  const backendSearchContacts = contextSearchAction.payload?.items || contextAffinityAction.payload?.items || [];
  const selectedIdSet = useMemo(() => new Set(contextLocalSelection.selectedIds), [contextLocalSelection.selectedIds]);
  const excludedIdSet = useMemo(() => new Set(contextLocalSelection.excludedIds), [contextLocalSelection.excludedIds]);
  const contextContactsForDisplay = useMemo(
    () => uniqueContactCandidates([
      ...searchableContextContacts,
      ...(contextSearchAction.status === "loaded" ? backendSearchContacts : []),
      ...Object.values(contextLocalSelection.candidatesById || {}),
      ...(contactSelection.selected_candidates || []),
      ...(contactSelection.excluded_candidates || [])
    ]),
    [
      contextLocalSelection.candidatesById,
      contactSelection.selected_candidates,
      contactSelection.excluded_candidates,
      searchableContextContacts,
      contextSearchAction.status,
      contextAffinityAction.status,
      backendSearchContacts
    ]
  );
  const selectedContextContacts = contextContactsForDisplay.filter((candidate) => contactIdSetHasCandidate(selectedIdSet, candidate));
  const visibleContextContacts = contextContactsForDisplay
    .filter((candidate) => contactMatchesQuery(candidate, contextContactQuery))
    .sort((left, right) => {
      const leftSelected = contactIdSetHasCandidate(selectedIdSet, left) ? 1 : 0;
      const rightSelected = contactIdSetHasCandidate(selectedIdSet, right) ? 1 : 0;
      if (leftSelected !== rightSelected) return rightSelected - leftSelected;
      const rankDelta = contactRankScore(right) - contactRankScore(left);
      if (rankDelta !== 0) return rankDelta;
      const confidenceDelta = contactConfidence(right) - contactConfidence(left);
      if (confidenceDelta !== 0) return confidenceDelta;
      return String(left.label || left.email || "").localeCompare(String(right.label || right.email || ""));
    })
    .slice(0, 30);
  const dedupeClusters = contactSelection.dedupe_clusters || [];
  const contactMergeState = contactSelection.merge_state || {};
  const finalPreview = finalPreviewAction.payload?.final_preview || conversationDetail?.final_preview || {};
  const finalPreviewBlocked = finalPreview.status === "blocked_identity_or_context_review" || Boolean(finalPreview.identity_context_warnings?.length);
  useEffect(() => {
    setIdentityReview(conversationDetail?.identity_review || null);
    setSpeakerLocalAssignments({});
    setSpeakerManualLabels({});
    setFirstPassAction({ status: "idle", message: "", payload: null });
    setSpeakerReviewAction({ status: "idle", message: "", payload: null });
    setSpeakerPreprocessing(null);
    setSpeakerPreprocessingAction({ status: "idle", message: "", payload: null });
    setContextAction({ status: "idle", message: "", payload: null });
    setContextContactAction({ status: "idle", message: "", payload: null });
    setContextContactQuery("");
    setContextSearchAction({ status: "idle", message: "", payload: null });
    setContextAffinityAction({ status: "idle", message: "", payload: null });
    setContextMergeAction({ status: "idle", message: "", payload: null });
    setContextManualContact({ label: "", email: "" });
    setContextLocalSelection({ selectedIds: [], excludedIds: [], pendingActions: {}, candidatesById: {}, dirty: false });
    setContextInstructionDraft("");
    setContextInstructionAction({ status: "idle", message: "", payload: null });
    setFinalPreviewAction({ status: "idle", message: "", payload: null });
  }, [conversationDetail?.conversation?.key, item.id]);
  useEffect(() => {
    let cancelled = false;
    async function loadSpeakerPreprocessing() {
      if (activeWorkflowView !== "speakers") return;
      setSpeakerPreprocessingAction((current) => ({ ...current, status: "loading", message: "Loading speaker preprocessing..." }));
      try {
        const state = await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing`);
        if (cancelled) return;
        setSpeakerPreprocessing(state);
        setSpeakerPreprocessingAction({ status: "loaded", message: "", payload: state });
      } catch (error) {
        if (cancelled) return;
        setSpeakerPreprocessingAction({ status: "error", message: `Speaker preprocessing failed: ${error.message}`, payload: null });
      }
    }
    loadSpeakerPreprocessing();
    return () => {
      cancelled = true;
    };
  }, [activeWorkflowView, item.id]);
  useEffect(() => {
    const candidatesById = {};
    [
      ...(contactSelection.selected_candidates || []),
      ...(contactSelection.excluded_candidates || []),
      ...searchableContextContacts
    ].forEach((candidate) => {
      contactCandidateIds(candidate).forEach((id) => {
        if (!candidatesById[id]) candidatesById[id] = candidate;
      });
    });
    setContextLocalSelection((current) => {
      if (current.dirty) return current;
      return {
        selectedIds: [...(contactSelection.selected_candidate_ids || [])],
        excludedIds: [...(contactSelection.excluded_candidate_ids || [])],
        pendingActions: {},
        candidatesById,
        dirty: false
      };
    });
  }, [
    conversationDetail?.conversation?.key,
    item.id,
    contactSelection.selection_path,
    JSON.stringify(contactSelection.selected_candidate_ids || []),
    JSON.stringify(contactSelection.excluded_candidate_ids || []),
    searchableContextContacts
  ]);
  useEffect(() => {
    setContextInstructionDraft(operatorContext.instruction_text || "");
  }, [conversationDetail?.conversation?.key, operatorContext.instruction_text]);
  useEffect(() => {
    let cancelled = false;
    const query = contextContactQuery.trim();
    if (activeWorkflowView !== "context" || query.length < 2) {
      setContextSearchAction({ status: "idle", message: "", payload: null });
      return () => {
        cancelled = true;
      };
    }
    setContextSearchAction((current) => ({ ...current, status: "loading", message: "Searching contacts..." }));
    const timer = window.setTimeout(async () => {
      try {
        const payload = await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/contact-search?q=${encodeURIComponent(query)}&limit=30`);
        if (cancelled) return;
        setContextSearchAction({ status: "loaded", message: "", payload });
      } catch (error) {
        if (cancelled) return;
        setContextSearchAction({ status: "error", message: `Contact search failed: ${error.message}`, payload: null });
      }
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeWorkflowView, contextContactQuery, item.id]);
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

  async function prepareSelectedFirstPassSummary() {
    setFirstPassAction({ status: "running", message: "Preparing selected summary batch...", payload: null });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/first-pass-summary/prepare`, { store: true });
      setFirstPassAction({
        status: payload.status || "prepared",
        message: `Prepared ${payload.request_count || 0} selected request; no provider work was submitted.`,
        payload
      });
    } catch (error) {
      setFirstPassAction({ status: "error", message: `First-pass prepare failed: ${error.message}`, payload: null });
    }
  }

  async function submitSelectedFirstPassSummary() {
    if (!selectedFirstPassManifest) return;
    const approved = window.confirm("Submit this selected first-pass summary batch to the configured provider?");
    if (!approved) return;
    setFirstPassAction((current) => ({ ...current, status: "submitting", message: "Submitting selected summary batch..." }));
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/first-pass-summary/submit`, {
        manifest: selectedFirstPassManifest,
        approval_token: firstPassSummaryState.future_required_approval_token_for_submit || "SUBMIT_FIRST_PASS_SUMMARY_BATCH"
      });
      setFirstPassAction({
        status: payload.status || "submitted",
        message: `Submitted selected request; batch ${payload.batch_id || "pending id"}.`,
        payload
      });
    } catch (error) {
      setFirstPassAction((current) => ({ ...current, status: "error", message: `First-pass submit failed: ${error.message}` }));
    }
  }

  async function runSelectedFirstPassSummary() {
    setFirstPassAction({ status: "running", message: "Running initial summary...", payload: firstPassAction.payload });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/first-pass-summary/run`, {
        store: true,
        approval_token: firstPassSummaryState.future_required_approval_token_for_submit || "SUBMIT_FIRST_PASS_SUMMARY_BATCH"
      });
      setFirstPassAction({
        status: payload.status || "submitted",
        message: `Initial summary submitted; batch ${payload.batch_id || "pending id"}.`,
        payload
      });
    } catch (error) {
      setFirstPassAction((current) => ({ ...current, status: "error", message: `Initial summary failed: ${error.message}` }));
    }
  }

  function runFirstPassPrimaryAction() {
    if (summaryText || firstPassBusy) return;
    if (firstPassPreparedOnly) {
      submitSelectedFirstPassSummary();
      return;
    }
    if (selectedFirstPassManifest) {
      refreshSelectedFirstPassSummary();
      return;
    }
    runSelectedFirstPassSummary();
  }

  async function refreshSelectedFirstPassSummary() {
    if (!selectedFirstPassManifest) return;
    setFirstPassAction((current) => ({ ...current, status: "checking", message: "Checking selected summary batch..." }));
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/first-pass-summary/status`, {
        manifest: selectedFirstPassManifest,
        materialize: true
      });
      const counts = payload.batch_counts || {};
      const countText = Object.entries(counts).map(([key, value]) => `${key}: ${value}`).join(", ");
      setFirstPassAction({
        status: payload.status || "checked",
        message: countText
          ? `Batch status ${payload.status}; ${countText}. Materialized ${payload.materialized?.length || 0}.`
          : `Batch status ${payload.status}. Materialized ${payload.materialized?.length || 0}.`,
        payload
      });
    } catch (error) {
      setFirstPassAction((current) => ({ ...current, status: "error", message: `First-pass status failed: ${error.message}` }));
    }
  }

  function stageSpeakerReview(speaker, action, candidate = {}) {
    const speakerLabel = speaker.speaker_label;
    const contactId = contactCandidateId(candidate);
    const contactLabel = candidate.label || candidate.email || speakerLabel;
    const email = candidate.email || "";
    setSpeakerLocalAssignments((current) => ({
      ...current,
      [speakerLabel]: {
        action,
        speaker_label: speakerLabel,
        contact_id: contactId,
        contact_label: contactLabel,
        email,
        status: action === "confirm" ? "confirmed" : action === "llm_readout" ? "llm_readout" : "deferred",
        assignment: action === "confirm"
          ? {
              contact_id: contactId,
              contact_label: contactLabel,
              email
            }
          : null
      }
    }));
    setSpeakerReviewAction({
      status: "staged",
      message: `${speakerLabel} staged. Save speaker choices when finished.`,
      payload: null
    });
  }

  async function saveSpeakerReviews() {
    const stagedAssignments = Object.values(speakerLocalAssignments || {});
    if (!stagedAssignments.length) return;
    setSpeakerReviewAction({ status: "running", message: `Saving ${stagedAssignments.length} speaker choice(s)...`, payload: null });
    try {
      let latestPayload = null;
      for (const staged of stagedAssignments) {
        const note = staged.action === "confirm"
          ? "Confirmed in the conversation workspace."
          : staged.action === "llm_readout"
            ? "LLM should assign this speaker during the final readout using selected context contacts."
            : "Deferred from the conversation workspace.";
        latestPayload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/identity-review`, {
          action: staged.action,
          speaker_label: staged.speaker_label,
          contact_id: staged.contact_id || "",
          contact_label: staged.contact_label || staged.speaker_label,
          email: staged.email || "",
          reviewer: "operator",
          note
        });
      }
      if (latestPayload?.identity_review) setIdentityReview(latestPayload.identity_review);
      setSpeakerLocalAssignments({});
      setSpeakerReviewAction({
        status: latestPayload?.status || "saved",
        message: `${stagedAssignments.length} speaker choice(s) saved.`,
        payload: latestPayload
      });
    } catch (error) {
      setSpeakerReviewAction({ status: "error", message: `Speaker review save failed: ${error.message}`, payload: null });
    }
  }

  async function prepareSpeakerClueDiscovery() {
    setSpeakerPreprocessingAction({ status: "running", message: "Preparing reviewed Clue Discovery...", payload: null });
    try {
      const prepared = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing/prepare-discovery`,
        {}
      );
      setSpeakerPreprocessingAction({
        status: "prepared",
        message: "Clue Discovery packet prepared in App Intelligence; no prompt was sent.",
        payload: prepared
      });
    } catch (error) {
      setSpeakerPreprocessingAction({ status: "error", message: `Clue Discovery prepare failed: ${error.message}`, payload: null });
    }
  }

  async function prepareSpeakerIdentityEvaluation() {
    const clueRunId = speakerPreprocessingAction.payload?.phase === "clue_discovery"
      ? speakerPreprocessingAction.payload.run_id
      : "";
    if (!clueRunId) return;
    setSpeakerPreprocessingAction({ status: "running", message: "Validating captured clues and retrieving bounded provenance...", payload: speakerPreprocessingAction.payload });
    try {
      const prepared = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing/prepare-evaluation`,
        { clue_discovery_run_id: clueRunId }
      );
      setSpeakerPreprocessingAction({
        status: "prepared",
        message: "Identity Evaluation packet prepared from captured Clue Discovery; no prompt was sent.",
        payload: { ...prepared, clue_discovery_run_id: clueRunId }
      });
    } catch (error) {
      setSpeakerPreprocessingAction((current) => ({ ...current, status: "error", message: `Identity Evaluation prepare failed: ${error.message}` }));
    }
  }

  async function captureSpeakerIdentityEvaluation() {
    const evaluationRunId = speakerPreprocessingAction.payload?.phase === "identity_evaluation"
      ? speakerPreprocessingAction.payload.run_id
      : "";
    if (!evaluationRunId) return;
    setSpeakerPreprocessingAction((current) => ({ ...current, status: "running", message: "Validating and persisting captured Identity Evaluation..." }));
    try {
      const result = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing/capture-evaluation`,
        {
          identity_evaluation_run_id: evaluationRunId,
          clue_discovery_run_id: speakerPreprocessingAction.payload?.clue_discovery_run_id || ""
        }
      );
      const state = await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing`);
      setSpeakerPreprocessing(state);
      setSpeakerPreprocessingAction({ status: "recorded", message: "Identity Evaluation validated and stored for review.", payload: result });
    } catch (error) {
      setSpeakerPreprocessingAction((current) => ({ ...current, status: "error", message: `Identity Evaluation capture failed: ${error.message}` }));
    }
  }

  async function recordSpeakerProposalDecision(proposal, action) {
    if (!currentSpeakerEvaluation) return;
    setSpeakerPreprocessingAction({ status: "running", message: `Recording ${action} decision...`, payload: null });
    try {
      const result = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing/decisions`,
        {
          evaluation_id: currentSpeakerEvaluation.evaluation_id,
          proposal_id: proposal.proposal_id,
          action,
          reviewer: "operator",
          method: "individual"
        }
      );
      const state = await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing`);
      setSpeakerPreprocessing(state);
      setSpeakerPreprocessingAction({ status: "recorded", message: `Proposal ${action} recorded.`, payload: result });
    } catch (error) {
      setSpeakerPreprocessingAction({ status: "error", message: `Speaker decision failed: ${error.message}`, payload: null });
    }
  }

  async function confirmReadySpeakerProposals() {
    if (!currentSpeakerEvaluation) return;
    setSpeakerPreprocessingAction({ status: "running", message: "Confirming ready proposals only...", payload: null });
    try {
      const result = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing/confirm-ready`,
        {
          evaluation_id: currentSpeakerEvaluation.evaluation_id,
          reviewer: "operator"
        }
      );
      const state = await fetchJson(`/api/conversations/${encodeURIComponent(item.id)}/speaker-preprocessing`);
      setSpeakerPreprocessing(state);
      setSpeakerPreprocessingAction({
        status: "recorded",
        message: `Confirmed ${result.confirmed_proposal_ids?.length || 0} ready proposal(s).`,
        payload: result
      });
    } catch (error) {
      setSpeakerPreprocessingAction({ status: "error", message: `Ready-confirm failed: ${error.message}`, payload: null });
    }
  }

  async function previewContextWorkbench(queue = false) {
    setContextAction({ status: "running", message: queue ? "Queueing context workbench review..." : "Preparing context workbench preview...", payload: null });
    try {
      const payload = await postJson(
        `/api/conversations/${encodeURIComponent(item.id)}/context-workbench/${queue ? "queue" : "preview"}`,
        queue ? { approval_token: "QUEUE_CONTEXT_WORKBENCH_RUN" } : {}
      );
      setContextAction({
        status: payload.status || "previewed",
        message: queue ? "Context workbench run queued locally; no provider was run." : "Context workbench preview recorded locally.",
        payload
      });
    } catch (error) {
      setContextAction({ status: "error", message: `Context workbench action failed: ${error.message}`, payload: null });
    }
  }

  function stageContextContactSelection(candidate, action) {
    const id = contactCandidateId(candidate) || `manual-${Date.now()}`;
    const ids = contactCandidateIds({ ...candidate, contact_id: id });
    setContextLocalSelection((current) => {
      const selected = new Set(current.selectedIds || []);
      const excluded = new Set(current.excludedIds || []);
      ids.forEach((candidateId) => {
        if (action === "select") {
          selected.add(candidateId);
          excluded.delete(candidateId);
        } else if (action === "exclude") {
          excluded.add(candidateId);
          selected.delete(candidateId);
        } else {
          selected.delete(candidateId);
          excluded.delete(candidateId);
        }
      });
      const candidatesById = { ...(current.candidatesById || {}) };
      ids.forEach((candidateId) => {
        candidatesById[candidateId] = { ...candidate, contact_id: contactCandidateId(candidate) || candidateId };
      });
      return {
        selectedIds: [...selected],
        excludedIds: [...excluded],
        pendingActions: {
          ...(current.pendingActions || {}),
          [id]: { action, candidate: { ...candidate, contact_id: contactCandidateId(candidate) || id } }
        },
        candidatesById,
        dirty: true
      };
    });
    setContextContactAction({
      status: "dirty",
      message: action === "select"
        ? "Contact marked for context."
        : action === "exclude"
          ? "Contact marked as excluded."
          : "Contact choice cleared.",
      payload: null
    });
  }

  async function persistContextContactSelection({ silent = false } = {}) {
    const pendingActions = Object.values(contextLocalSelection.pendingActions || {});
    if (!pendingActions.length) return null;
    if (!silent) {
      setContextContactAction({ status: "running", message: "Saving contact choices...", payload: null });
    }
    try {
      const actions = pendingActions.map((pending) => {
        const candidate = pending.candidate || {};
        const action = {
          action: pending.action,
          candidate_id: candidate.contact_id || "",
          actor_type: "operator",
          reviewer: "operator",
          note: pending.action === "select"
            ? "Selected in the context workbench."
            : pending.action === "exclude"
              ? "Excluded in the context workbench."
              : "Cleared in the context workbench."
        };
        if (candidate.manual_candidate) action.manual_candidate = candidate.manual_candidate;
        return action;
      });
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/contact-selection-batch`, { actions });
      setContextContactAction({
        status: "saved",
        message: silent ? "" : "Contact choices saved.",
        payload
      });
      setContextLocalSelection((current) => ({
        ...current,
        pendingActions: {},
        dirty: false
      }));
      return payload;
    } catch (error) {
      setContextContactAction({ status: "error", message: `Contact selection failed: ${error.message}`, payload: null });
      throw error;
    }
  }

  function addManualContextContact() {
    const label = contextManualContact.label.trim();
    const email = contextManualContact.email.trim();
    if (!label && !email) {
      setContextContactAction({ status: "error", message: "Enter a contact name or email before adding.", payload: null });
      return;
    }
    const contact_id = `manual-${Date.now()}`;
    stageContextContactSelection(
      {
        contact_id,
        label: label || email,
        email,
        source_type: "manual_context_contact",
        source_profile: "operator",
        confidence: 1,
        manual_candidate: { label, email }
      },
      "select"
    );
    setContextManualContact({ label: "", email: "" });
  }

  async function searchConfiguredContextContacts() {
    const query = contextContactQuery.trim();
    if (query.length < 2) {
      setContextSearchAction({ status: "error", message: "Enter at least two characters before searching sources.", payload: null });
      return;
    }
    setContextSearchAction({ status: "refreshing", message: "Searching configured contact sources...", payload: null });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/contact-refresh`, {
        query,
        limit: 30
      });
      setContextSearchAction({
        status: "loaded",
        message: payload.total
          ? `Found ${payload.total} contact candidate(s) from configured sources.`
          : "No configured source contacts matched.",
        payload
      });
    } catch (error) {
      setContextSearchAction({ status: "error", message: `Configured source search failed: ${error.message}`, payload: null });
    }
  }

  async function refreshContextContactAffinity() {
    const query = contextContactQuery.trim();
    setContextAffinityAction({ status: "refreshing", message: "Refreshing relationship ranking...", payload: null });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/contact-affinity/refresh`, {
        query,
        limit: 50
      });
      setContextAffinityAction({
        status: "loaded",
        message: payload.item_count
          ? `Ranked ${payload.item_count} contact candidate(s) with local relationship signals.`
          : "No contact candidates were available for affinity ranking.",
        payload
      });
      setContextSearchAction((current) => current.status === "loaded" ? { ...current, payload } : current);
    } catch (error) {
      setContextAffinityAction({ status: "error", message: `Relationship ranking failed: ${error.message}`, payload: null });
    }
  }

  async function recordContextContactMerge(cluster, action) {
    const contactIds = (cluster.contact_ids || []).filter(Boolean);
    if (contactIds.length < 2) {
      setContextMergeAction({ status: "error", message: "Merge review requires at least two contact ids.", payload: null });
      return;
    }
    setContextMergeAction({
      status: "running",
      message: action === "split" ? "Recording reviewed split..." : "Recording reviewed merge...",
      payload: null
    });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/contact-merge-batch`, {
        actions: [
          {
            action,
            contact_ids: contactIds,
            dedupe_key: cluster.dedupe_key || "",
            canonical_candidate: {
              contact_id: contactIds[0],
              label: cluster.label || cluster.email || "",
              email: cluster.email || ""
            },
            actor_type: "operator",
            reviewer: "operator",
            note: action === "split"
              ? "Split from the context workbench merge review."
              : "Approved from the context workbench merge review."
          }
        ]
      });
      setContextMergeAction({
        status: "saved",
        message: action === "split" ? "Reviewed split saved." : "Reviewed merge saved.",
        payload
      });
      setContextContactAction({
        status: "saved",
        message: action === "split" ? "Reviewed split saved." : "Reviewed merge saved.",
        payload
      });
    } catch (error) {
      setContextMergeAction({ status: "error", message: `Merge review failed: ${error.message}`, payload: null });
    }
  }

  async function saveContextInstructions() {
    setContextInstructionAction({ status: "running", message: "Saving context instructions...", payload: null });
    try {
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/context-workbench/instructions`, {
        instruction_text: contextInstructionDraft,
        actor_type: "operator",
        reviewer: "operator",
        note: "Saved in the context workbench."
      });
      setContextInstructionAction({
        status: payload.status || "saved",
        message: contextInstructionDraft.trim() ? "Context instructions saved." : "Context instructions cleared.",
        payload
      });
    } catch (error) {
      setContextInstructionAction({ status: "error", message: `Context instructions failed: ${error.message}`, payload: null });
    }
  }

  async function queueFinalPreview() {
    setFinalPreviewAction({ status: "running", message: "Queueing deposition and memory preview...", payload: null });
    try {
      if (Object.keys(contextLocalSelection.pendingActions || {}).length) {
        await persistContextContactSelection({ silent: true });
      }
      const payload = await postJson(`/api/conversations/${encodeURIComponent(item.id)}/final-preview/queue`, {
        approval_token: "QUEUE_DEPOSITION_MEMORY_PREVIEW"
      });
      setFinalPreviewAction({
        status: payload.status || "queued",
        message: payload.status === "blocked_identity_or_context_review"
          ? "Preview is blocked until identity and context warnings are resolved."
          : "Deposition and memory preview queued for local review; no external write was performed.",
        payload
      });
    } catch (error) {
      setFinalPreviewAction({ status: "error", message: `Preview queue failed: ${error.message}`, payload: null });
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
              {WORKFLOW_VIEWS.map((view) => (
                <button
                  aria-pressed={activeWorkflowView === view.id}
                  className={activeWorkflowView === view.id ? "active" : ""}
                  key={view.id}
                  onClick={() => onWorkflowViewChange(view.id)}
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
                <div className="workflow-action-panel">
                  <div className="workflow-prep-card">
                    <div>
                      <span>Initial summary prep</span>
                      <strong>{firstPassPrimaryLabel}</strong>
                      <p>{firstPassPrimaryDetail}</p>
                    </div>
                    <button className="primary-workflow-action" onClick={runFirstPassPrimaryAction} disabled={Boolean(summaryText) || firstPassBusy} type="button">
                      {firstPassPrimaryLabel}
                    </button>
                    <dl className="compact-definition-list">
                      <dt>Status</dt>
                      <dd>{statusLabel(firstPassSummaryState.status || "unknown")}</dd>
                      <dt>Source</dt>
                      <dd>{firstPassSummaryState.source_document_id || "Not linked"}</dd>
                      <dt>Summary</dt>
                      <dd>{firstPassSummaryState.summary_document_id || "Not materialized"}</dd>
                      <dt>External action</dt>
                      <dd>{firstPassExternalActionText}</dd>
                    </dl>
                  </div>
                  <details className="workflow-secondary-actions">
                    <summary>Advanced summary controls</summary>
                    <div className="workflow-action-row">
                      <button onClick={prepareSelectedFirstPassSummary} disabled={firstPassBusy} type="button">
                        Prepare only
                      </button>
                      <button onClick={submitSelectedFirstPassSummary} disabled={!selectedFirstPassManifest || firstPassBusy} type="button">
                        Submit
                      </button>
                      <button onClick={refreshSelectedFirstPassSummary} disabled={!selectedFirstPassManifest || firstPassBusy} type="button">
                        Check
                      </button>
                    </div>
                  </details>
                  {firstPassAction.message ? (
                    <div className={`action-notice ${firstPassAction.status}`}>
                      <strong>{firstPassAction.message}</strong>
                      {selectedFirstPassManifest ? <small>{selectedFirstPassManifest}</small> : null}
                    </div>
                  ) : null}
                </div>
              </section>
            ) : null}

            {activeWorkflowView === "context" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Context workbench</span>
                    <h3>{statusLabel(contextWorkbench.status || "provenance and context gathering")}</h3>
                  </div>
                  <strong>{contextWorkbench.included_source_count || 0} included</strong>
                </div>
                {contextWorkbench.selected_candidate?.label ? (
                  <div className="context-route-card">
                    <span>Selected route</span>
                    <strong>{contextWorkbench.selected_candidate.label}</strong>
                    <small>{contextWorkbench.selected_candidate.target_kind || "target"} · confidence {contextWorkbench.confidence ?? "unknown"}</small>
                  </div>
                ) : (
                  <p className="muted">No selected route candidate is attached yet.</p>
                )}
                {contextWorkbench.warnings?.length ? (
                  <div className="warning-list">
                    {contextWorkbench.warnings.map((warning) => <span key={warning}>{warning}</span>)}
                  </div>
                ) : null}
                <div className="context-route-card">
                  <span>Participant identity</span>
                  <strong>{statusLabel(identityBundle.review_status || contextWorkbench.identity_status || "unknown")}</strong>
                  <small>
                    {(identityBundle.contact_candidates?.length || 0)} contact candidate(s) · {(identityBundle.unresolved_ambiguities?.length || 0)} unresolved
                  </small>
                </div>
                {identityBundle.source_profiles?.length ? (
                  <div className="chip-cloud">
                    {identityBundle.source_profiles.map((profile) => (
                      <span key={`${profile.source}-${profile.profile}`}>
                        {profile.source}: {profile.profile}
                      </span>
                    ))}
                  </div>
                ) : null}
                <div className="context-contact-panel">
                  <div className="workflow-view-heading compact">
                    <div>
                      <span>Contacts for context</span>
                      <h3>{selectedContextContacts.length || 0} selected for context</h3>
                    </div>
                    <strong>{contextContactsForDisplay.length || 0} searchable</strong>
                  </div>
                  <div className="context-contact-tools">
                    <label className="workflow-field">
                      Search contacts
                      <input
                        onChange={(event) => setContextContactQuery(event.target.value)}
                        placeholder="Name, email, source, or tenant"
                        type="search"
                        value={contextContactQuery}
                      />
                    </label>
                    <div className="manual-contact-row">
                      <input
                        aria-label="New contact name"
                        onChange={(event) => setContextManualContact((current) => ({ ...current, label: event.target.value }))}
                        placeholder="Contact name"
                        type="text"
                        value={contextManualContact.label}
                      />
                      <input
                        aria-label="New contact email"
                        onChange={(event) => setContextManualContact((current) => ({ ...current, email: event.target.value }))}
                        placeholder="Email"
                        type="email"
                        value={contextManualContact.email}
                      />
                      <button disabled={contextContactAction.status === "running"} onClick={addManualContextContact} type="button">
                        Add
                      </button>
                    </div>
                    <button
                      className="save-contact-selection"
                      disabled={!Object.keys(contextLocalSelection.pendingActions || {}).length || contextContactAction.status === "running"}
                      onClick={() => persistContextContactSelection()}
                      type="button"
                    >
                      Save choices
                    </button>
                    <button
                      className="source-contact-search"
                      disabled={contextContactQuery.trim().length < 2 || contextSearchAction.status === "refreshing"}
                      onClick={searchConfiguredContextContacts}
                      type="button"
                    >
                      Search sources
                    </button>
                    <button
                      className="source-contact-search affinity-refresh"
                      disabled={contextAffinityAction.status === "refreshing"}
                      onClick={refreshContextContactAffinity}
                      type="button"
                    >
                      Refresh ranking
                    </button>
                  </div>
                  <div className="contact-workbench-status">
                    <span>Search cache: {statusLabel(contactSelection.search_cache_status || "empty")}</span>
                    <span>Affinity: {statusLabel(contactSelection.affinity_cache_status || contextSearchAction.payload?.affinity_cache_status || "empty")}</span>
                    <span>Merge review: {statusLabel(contactMergeState.status || "empty")}</span>
                  </div>
                  {selectedContextContacts.length ? (
                    <div className="selected-contact-strip">
                      {selectedContextContacts.map((candidate) => (
                        <button
                          key={`selected-${contactCandidateId(candidate)}`}
                          onClick={() => stageContextContactSelection(candidate, "clear")}
                          type="button"
                        >
                          {candidate.label || candidate.email}
                        </button>
                      ))}
                    </div>
                  ) : null}
                  {dedupeClusters.length ? (
                    <div className="dedupe-cluster-list">
                      {dedupeClusters.slice(0, 4).map((cluster) => (
                        <article key={cluster.dedupe_key}>
                          <div>
                            <strong>{cluster.label || cluster.email}</strong>
                            <small>{cluster.source_count} merged sources · {(cluster.contact_ids || []).length} contact ids</small>
                          </div>
                          <div className="context-contact-actions">
                            <button
                              disabled={contextMergeAction.status === "running"}
                              onClick={() => recordContextContactMerge(cluster, "merge")}
                              type="button"
                            >
                              Merge
                            </button>
                            <button
                              disabled={contextMergeAction.status === "running"}
                              onClick={() => recordContextContactMerge(cluster, "split")}
                              type="button"
                            >
                              Split
                            </button>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : null}
                  {visibleContextContacts.length ? (
                    <div className="context-contact-grid">
                      {visibleContextContacts.map((candidate) => {
                        const selected = contactIdSetHasCandidate(selectedIdSet, candidate);
                        const excluded = contactIdSetHasCandidate(excludedIdSet, candidate);
                        return (
                          <article className={selected ? "selected" : excluded ? "excluded" : ""} key={`${candidate.contact_id}-${candidate.source_type}`}>
                            <div>
                              <strong>{candidate.label || candidate.email || "Contact candidate"}</strong>
                              <small>
                                {[candidate.email, candidate.source_type || candidate.source, candidate.source_profile, candidate.source_count > 1 ? `${candidate.source_count} merged sources` : "", candidate.confidence ? `confidence ${candidate.confidence}` : "", candidate.rank_score ? `rank ${candidate.rank_score}` : ""]
                                  .filter(Boolean)
                                  .join(" · ")}
                              </small>
                              {contactRankingReasons(candidate).length ? (
                                <div className="ranking-reason-list">
                                  {contactRankingReasons(candidate).map((reason) => (
                                    <span key={`${candidate.contact_id}-${reason}`}>{reason}</span>
                                  ))}
                                </div>
                              ) : null}
                              {candidate.merged_sources?.length ? (
                                <details className="contact-evidence-details">
                                  <summary>Sources</summary>
                                  <ul>
                                    {candidate.merged_sources.slice(0, 5).map((source) => (
                                      <li key={`${candidate.contact_id}-${source.contact_id}-${source.source_type}`}>
                                        {[source.label || source.email || "source", source.source_type, source.source_profile]
                                          .filter(Boolean)
                                          .join(" · ")}
                                      </li>
                                    ))}
                                  </ul>
                                </details>
                              ) : null}
                            </div>
                            <div className="context-contact-actions">
                              <button
                                disabled={selected}
                                onClick={() => stageContextContactSelection(candidate, "select")}
                                type="button"
                              >
                                Use
                              </button>
                              <button
                                disabled={excluded}
                                onClick={() => stageContextContactSelection(candidate, "exclude")}
                                type="button"
                              >
                                Exclude
                              </button>
                              {(selected || excluded) ? (
                                <button
                                  onClick={() => stageContextContactSelection(candidate, "clear")}
                                  type="button"
                                >
                                  Clear
                                </button>
                              ) : null}
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="muted">{contextContactQuery ? "No cached contacts match the search." : "No proposed contacts are available yet."}</p>
                  )}
                  {contextSearchAction.message ? (
                    <div className={`action-notice ${contextSearchAction.status}`}>
                      <strong>{contextSearchAction.message}</strong>
                      {contextSearchAction.payload?.job_id ? <small>{contextSearchAction.payload.job_id}</small> : null}
                    </div>
                  ) : null}
                  {contextAffinityAction.message ? (
                    <div className={`action-notice ${contextAffinityAction.status}`}>
                      <strong>{contextAffinityAction.message}</strong>
                      {contextAffinityAction.payload?.cache_path ? <small>{contextAffinityAction.payload.cache_path}</small> : null}
                    </div>
                  ) : null}
                  {contextMergeAction.message ? (
                    <div className={`action-notice ${contextMergeAction.status}`}>
                      <strong>{contextMergeAction.message}</strong>
                      {contextMergeAction.payload?.merge_path ? <small>{contextMergeAction.payload.merge_path}</small> : null}
                    </div>
                  ) : null}
                  {contextContactAction.message ? (
                    <div className={`action-notice ${contextContactAction.status}`}>
                      <strong>{contextContactAction.message}</strong>
                      {contextContactAction.payload?.selection_path ? <small>{contextContactAction.payload.selection_path}</small> : null}
                    </div>
                  ) : null}
                </div>
                <div className="context-contact-panel">
                  <div className="workflow-view-heading compact">
                    <div>
                      <span>Operator instructions</span>
                      <h3>{operatorContext.status === "provided" ? "Saved context is attached" : "No saved context yet"}</h3>
                    </div>
                  </div>
                  <label className="workflow-field">
                    Natural language context
                    <textarea
                      onChange={(event) => setContextInstructionDraft(event.target.value)}
                      placeholder="Add participant notes, identity hints, customer context, disambiguation rules, or readout instructions."
                      rows={5}
                      value={contextInstructionDraft}
                    />
                  </label>
                  <div className="workflow-action-row">
                    <button disabled={contextInstructionAction.status === "running"} onClick={saveContextInstructions} type="button">
                      Save instructions
                    </button>
                  </div>
                  {contextInstructionAction.message ? (
                    <div className={`action-notice ${contextInstructionAction.status}`}>
                      <strong>{contextInstructionAction.message}</strong>
                      {contextInstructionAction.payload?.instruction_path ? <small>{contextInstructionAction.payload.instruction_path}</small> : null}
                    </div>
                  ) : null}
                </div>
                <div className="readout-columns">
                  <div>
                    <strong>Included provenance</strong>
                    {contextWorkbench.included_sources?.length ? (
                      <ul>
                        {contextWorkbench.included_sources.slice(0, 8).map((source) => (
                          <li key={`${source.source_id}-${source.label}`}>
                            {source.label || source.source_type}
                            <small>{source.source_type}{source.snippet ? ` · ${source.snippet}` : ""}</small>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="muted">No included provenance sources are recorded yet.</p>
                    )}
                  </div>
                  <div>
                    <strong>Excluded provenance</strong>
                    {contextWorkbench.excluded_sources?.length ? (
                      <ul>
                        {contextWorkbench.excluded_sources.slice(0, 8).map((source) => (
                          <li key={`${source.source_id}-${source.label}`}>
                            {source.label || source.source_type}
                            <small>{source.quality_status || source.quality_reason || source.source_type}</small>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="muted">{contextWorkbench.excluded_source_count ? `${contextWorkbench.excluded_source_count} excluded source(s) summarized in warnings.` : "No excluded provenance sources are recorded yet."}</p>
                    )}
                  </div>
                </div>
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
                  <button onClick={() => previewContextWorkbench(false)} disabled={contextAction.status === "running"} type="button">Preview context run</button>
                  <button onClick={() => previewContextWorkbench(true)} disabled={contextAction.status === "running"} type="button">Queue context review</button>
                </div>
                {contextAction.message ? (
                  <div className={`action-notice ${contextAction.status}`}>
                    <strong>{contextAction.message}</strong>
                    {contextAction.payload?.manifest ? <small>{contextAction.payload.manifest}</small> : null}
                  </div>
                ) : null}
              </section>
            ) : null}

            {activeWorkflowView === "speakers" ? (
              <section className="workflow-view">
                <div className="workflow-view-heading">
                  <div>
                    <span>Speakers and contacts</span>
                    <h3>{speakerPendingCount} pending assignments</h3>
                  </div>
                  <div className="workflow-action-row">
                    <small>{selectedContextContacts.length} selected contacts available · {stagedSpeakerCount} staged</small>
                    <button
                      disabled={!stagedSpeakerCount || speakerReviewAction.status === "running"}
                      onClick={saveSpeakerReviews}
                      type="button"
                    >
                      Save speaker choices
                    </button>
                    {stagedSpeakerCount ? (
                      <button
                        disabled={speakerReviewAction.status === "running"}
                        onClick={() => {
                          setSpeakerLocalAssignments({});
                          setSpeakerReviewAction({ status: "idle", message: "Staged speaker choices discarded.", payload: null });
                        }}
                        type="button"
                      >
                        Discard
                      </button>
                    ) : null}
                  </div>
                </div>
                <div className="workflow-action-panel speaker-preprocessing-panel">
                  <div className="workflow-prep-card acoustic-shadow-card">
                    <div>
                      <span>Acoustic shadow evidence</span>
                      <strong>{statusLabel(acousticShadowEvidence.status || "absent")}</strong>
                      <p>
                        Enrolled subject-ID proposals are read-only evidence. They do not
                        create contacts, apply speaker assignments, or update voice profiles.
                      </p>
                    </div>
                    {acousticShadowEvidence.status === "available" ? (
                      <div className="identity-list preprocessing-proposals">
                        {(acousticShadowEvidence.rows || []).map((row) => (
                          <article key={`${acousticShadowEvidence.content_sha256}-${row.speaker_ref}`}>
                            <strong>{row.speaker_ref}</strong>
                            <small>
                              {statusLabel(row.disposition)} · {statusLabel(row.confidence_band)}
                              {row.subject_id ? ` · ${row.subject_id}` : " · no enrolled subject"}
                            </small>
                            <p>{row.rationale}</p>
                            <small>
                              {row.supporting_unit_count} supporting units across {row.supporting_candidate_family_count} model families · {row.opposing_unit_count} opposing units
                            </small>
                          </article>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">
                        {acousticShadowEvidence.status === "rejected"
                          ? "Private acoustic evidence was rejected because its binding or integrity could not be verified."
                          : "No validated acoustic shadow evidence is bound to this transcript."}
                      </p>
                    )}
                  </div>
                  <div className="workflow-prep-card joined-shadow-card">
                    <div>
                      <span>Joined identity shadow</span>
                      <strong>{statusLabel(joinedShadowEvidence.status || "absent")}</strong>
                      <p>
                        Three blinded conditions preserve acoustic and context factors separately.
                        Choices stay local to this browser view; no identity is preselected and no
                        assignment or knowledge write is available.
                      </p>
                    </div>
                    {joinedShadowEvidence.status === "sealed_pending_human_review" ? (
                      <>
                        <small>
                          {(joinedShadowEvidence.speaker_slots || []).length} sealed speaker slot(s) · {joinedShadowEvidence.preselected_decision_count || 0} preselected · apply disabled
                        </small>
                        <div className="identity-list preprocessing-proposals">
                          {(joinedShadowEvidence.speaker_slots || []).map((slot) => (
                            <article key={`${joinedShadowEvidence.packet_content_sha256}-${slot.speaker_ref}`}>
                              <strong>{slot.speaker_ref}</strong>
                              <small>
                                Acoustic {statusLabel(slot.acoustic?.disposition || "unknown")} · {statusLabel(slot.acoustic?.confidence_band || "none")}
                              </small>
                              <label className="workflow-field">
                                Human decision (not yet recorded)
                                <select
                                  aria-label={`Joined shadow decision for ${slot.speaker_ref}`}
                                  onChange={(event) => setJoinedShadowDecisions((current) => ({ ...current, [slot.speaker_ref]: event.target.value }))}
                                  value={joinedShadowDecisions[slot.speaker_ref] || ""}
                                >
                                  <option value="">Select after review</option>
                                  {(joinedShadowEvidence.candidate_options || []).map((candidate) => (
                                    <option key={candidate.person_id} value={candidate.person_id}>
                                      {candidate.label || candidate.email || candidate.person_id}
                                    </option>
                                  ))}
                                  <option value="not_listed">Person not listed</option>
                                  <option value="unresolved">Unresolved</option>
                                </select>
                              </label>
                              <details className="contact-evidence-details">
                                <summary>Blinded condition evidence</summary>
                                <div className="identity-list">
                                  {(slot.conditions || []).map((condition) => (
                                    <article key={condition.evaluation_id}>
                                      <strong>{statusLabel(condition.condition)}</strong>
                                      <small>{statusLabel(condition.outcome)} · confidence {condition.capped_confidence}</small>
                                      <p>{statusLabel(condition.abstention_reason || "no abstention")}</p>
                                    </article>
                                  ))}
                                </div>
                              </details>
                            </article>
                          ))}
                        </div>
                        <div className="workflow-action-row">
                          <button disabled title="Plan 0060 exposes no live apply path." type="button">Apply identity (disabled)</button>
                        </div>
                      </>
                    ) : (
                      <p className="muted">
                        {joinedShadowEvidence.status === "not_in_frozen_cohort"
                          ? "This conversation is outside the exact Plan 0060 cohort."
                          : "No validated joined shadow packet is bound to this transcript."}
                      </p>
                    )}
                  </div>
                  <div className="workflow-prep-card">
                    <div>
                      <span>App Intelligence preprocessing</span>
                      <strong>{statusLabel(speakerPreprocessing?.status || "not started")}</strong>
                      <p>Clue Discovery runs before host-owned provenance retrieval and Identity Evaluation. Prepared prompts remain unsent until reviewed through App Intelligence.</p>
                    </div>
                    <div className="workflow-action-row">
                      <button
                        className="primary-workflow-action"
                        disabled={speakerPreprocessingAction.status === "running"}
                        onClick={prepareSpeakerClueDiscovery}
                        type="button"
                      >
                        Prepare Clue Discovery
                      </button>
                      <button
                        disabled={!currentSpeakerEvaluation?.safe_bulk_confirm_ready || speakerPreprocessingAction.status === "running"}
                        onClick={confirmReadySpeakerProposals}
                        type="button"
                      >
                        Confirm ready only
                      </button>
                      <button
                        disabled={speakerPreprocessingAction.payload?.phase !== "clue_discovery" || speakerPreprocessingAction.status === "running"}
                        onClick={prepareSpeakerIdentityEvaluation}
                        type="button"
                      >
                        Prepare Identity Evaluation
                      </button>
                      <button
                        disabled={speakerPreprocessingAction.payload?.phase !== "identity_evaluation" || speakerPreprocessingAction.status === "running"}
                        onClick={captureSpeakerIdentityEvaluation}
                        type="button"
                      >
                        Capture scored proposals
                      </button>
                    </div>
                    {speakerPreprocessingAction.payload?.run_id ? (
                      <small>App Intelligence run: {speakerPreprocessingAction.payload.run_id} · prompt not sent</small>
                    ) : null}
                  </div>
                  {currentSpeakerEvaluation?.warnings?.length ? (
                    <div className="warning-list">
                      {currentSpeakerEvaluation.warnings.map((warning) => <span key={warning}>{warning}</span>)}
                    </div>
                  ) : null}
                  {speakerIdentityProposals.length ? (
                    <div className="identity-list preprocessing-proposals">
                      {speakerIdentityProposals.map((proposal) => {
                        const decision = latestSpeakerProposalDecisions[proposal.proposal_id];
                        const label = proposal.person_id
                          ? currentSpeakerEvaluation.people?.find((person) => person.person_id === proposal.person_id)?.display_name
                          : proposal.suggested_person?.name || proposal.suggested_person?.email;
                        return (
                          <article key={proposal.proposal_id}>
                            <strong>{(proposal.speaker_labels || []).join(" + ") || "Speaker proposal"}</strong>
                            <small>
                              {statusLabel(proposal.status)}{label ? ` · ${label}` : ""}
                              {proposal.confidence ? ` · ${proposal.confidence.numeric} / 100 (${proposal.confidence.band_label || statusLabel(proposal.confidence.band)})` : ""}
                            </small>
                            {proposal.rationale ? <p>{proposal.rationale}</p> : null}
                            {proposal.review_flags?.length ? (
                              <div className="warning-list">
                                {proposal.review_flags.map((flag) => <span key={`${proposal.proposal_id}-${flag}`}>{statusLabel(flag)}</span>)}
                              </div>
                            ) : null}
                            {proposal.factors?.length ? (
                              <details className="contact-evidence-details">
                                <summary>Evidence factors</summary>
                                <ul>
                                  {proposal.factors.map((factor, index) => (
                                    <li key={`${proposal.proposal_id}-${index}-${factor.factor}`}>
                                      {statusLabel(factor.factor)} · {factor.direction} · {factor.strength}
                                      <small>{(factor.evidence_ids || []).join(", ")}</small>
                                    </li>
                                  ))}
                                </ul>
                              </details>
                            ) : null}
                            <div className="workflow-action-row">
                              <button disabled={speakerPreprocessingAction.status === "running"} onClick={() => recordSpeakerProposalDecision(proposal, "confirm")} type="button">Confirm</button>
                              <button disabled={speakerPreprocessingAction.status === "running"} onClick={() => recordSpeakerProposalDecision(proposal, "reject")} type="button">Reject</button>
                              <button disabled={speakerPreprocessingAction.status === "running"} onClick={() => recordSpeakerProposalDecision(proposal, "defer")} type="button">Defer</button>
                              {decision ? <small>{statusLabel(decision.action)} by {decision.reviewer}</small> : null}
                            </div>
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="muted">No scored App Intelligence speaker proposals are persisted yet.</p>
                  )}
                  {speakerPreprocessingAction.message ? (
                    <div className={`action-notice ${speakerPreprocessingAction.status}`}>
                      <strong>{speakerPreprocessingAction.message}</strong>
                    </div>
                  ) : null}
                </div>
                {speakersForDisplay.length ? (
                  <div className="identity-list">
                    {speakersForDisplay.map((speaker) => (
                      <article key={speaker.speaker_label}>
                        <strong>{speaker.speaker_label}</strong>
                        <small>
                          {speaker.assignment?.contact_label
                            ? `${speaker.staged ? "staged " : ""}${statusLabel(speaker.status)} as ${speaker.assignment.contact_label}`
                            : `${speaker.staged ? "staged " : ""}${statusLabel(speaker.status || "pending")}`}
                        </small>
                        {selectedContextContacts.length ? (
                          <div className="candidate-list">
                            {selectedContextContacts.map((candidate) => {
                              const selected = speaker.status === "confirmed" && speakerAssignmentMatchesCandidate(speaker, candidate);
                              return (
                                <button
                                  className={selected ? "speaker-assignment-option selected" : "speaker-assignment-option"}
                                  disabled={speakerReviewAction.status === "running"}
                                  key={`${speaker.speaker_label}-${contactCandidateId(candidate)}`}
                                  onClick={() => stageSpeakerReview(speaker, "confirm", candidate)}
                                  type="button"
                                >
                                  <span className="assignment-option-main">
                                    <span>{candidate.label}</span>
                                    {selected ? <span className="selection-check" aria-label="Selected" /> : null}
                                  </span>
                                  <small>
                                    {candidate.email || candidate.source_type || candidate.source || "selected contact"}
                                  </small>
                                </button>
                              );
                            })}
                            <button
                              className={speaker.status === "llm_readout" ? "speaker-assignment-option selected readout-option" : "speaker-assignment-option readout-option"}
                              disabled={speakerReviewAction.status === "running"}
                              onClick={() => stageSpeakerReview(speaker, "llm_readout")}
                              type="button"
                            >
                              <span className="assignment-option-main">
                                <span>Assign at readout</span>
                                {speaker.status === "llm_readout" ? <span className="selection-check" aria-label="Selected" /> : null}
                              </span>
                              <small>Let the final readout LLM choose from selected contacts.</small>
                            </button>
                          </div>
                        ) : (
                          <div className="candidate-list">
                            <p className="muted">Select contacts in the Context workbench before matching speakers.</p>
                            <button
                              className={speaker.status === "llm_readout" ? "speaker-assignment-option selected readout-option" : "speaker-assignment-option readout-option"}
                              disabled={speakerReviewAction.status === "running"}
                              onClick={() => stageSpeakerReview(speaker, "llm_readout")}
                              type="button"
                            >
                              <span className="assignment-option-main">
                                <span>Assign at readout</span>
                                {speaker.status === "llm_readout" ? <span className="selection-check" aria-label="Selected" /> : null}
                              </span>
                              <small>Let the final readout LLM choose after context is assembled.</small>
                            </button>
                          </div>
                        )}
                        <div className="manual-contact-row">
                          <input
                            aria-label={`Manual contact for ${speaker.speaker_label}`}
                            disabled={speakerReviewAction.status === "running"}
                            onChange={(event) => setSpeakerManualLabels((current) => ({
                              ...current,
                              [speaker.speaker_label]: event.target.value
                            }))}
                            placeholder="Contact name or email"
                            type="text"
                            value={speakerManualLabels[speaker.speaker_label] || ""}
                          />
                          <button
                            disabled={
                              speakerReviewAction.status === "running" ||
                              !String(speakerManualLabels[speaker.speaker_label] || "").trim()
                            }
                            onClick={() => {
                              const value = String(speakerManualLabels[speaker.speaker_label] || "").trim();
                              const email = value.includes("@") ? value : "";
                              stageSpeakerReview(speaker, "confirm", { label: value, email });
                            }}
                            type="button"
                          >
                            Confirm typed
                          </button>
                        </div>
                        <button
                          disabled={speakerReviewAction.status === "running"}
                          onClick={() => stageSpeakerReview(speaker, "defer")}
                          title="Create a Review Queue item so an operator can resolve this speaker later."
                          type="button"
                        >
                          Needs manual review
                        </button>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No speaker turns were extracted for this conversation yet.</p>
                )}
                {speakerReviewAction.message ? (
                  <div className={`action-notice ${speakerReviewAction.status}`}>
                    <strong>{speakerReviewAction.message}</strong>
                  </div>
                ) : null}
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
                <div className="context-route-card">
                  <span>Deposition and memory preview</span>
                  <strong>{statusLabel(finalPreview.status || "not prepared")}</strong>
                  <small>{finalPreview.action_count || 0} deposition action(s) · {finalPreview.memory_candidate_count || 0} memory candidate(s)</small>
                </div>
                {finalPreview.warnings?.length ? (
                  <div className="warning-list">
                    {finalPreview.warnings.map((warning) => <span key={warning}>{warning}</span>)}
                  </div>
                ) : null}
                {finalPreviewBlocked ? (
                  <div className="action-notice blocked">
                    <strong>Identity or context review is still required.</strong>
                    {finalPreview.identity_context_warnings?.slice(0, 3).map((warning) => <small key={warning}>{warning}</small>)}
                  </div>
                ) : null}
                {finalPreview.actions?.length || finalPreview.memory_candidates?.length ? (
                  <div className="readout-columns">
                    <div>
                      <strong>Preview actions</strong>
                      {finalPreview.actions?.length ? (
                        <ul>
                          {finalPreview.actions.map((action, index) => (
                            <li key={`${index}-${action.target_kind}`}>
                              {statusLabel(action.action_type || "action")}
                              <small>{action.target_kind} · writes {action.writes_enabled ? "enabled" : "disabled"}</small>
                            </li>
                          ))}
                        </ul>
                      ) : <p className="muted">No deposition actions are previewed yet.</p>}
                    </div>
                    <div>
                      <strong>Memory candidates</strong>
                      {finalPreview.memory_candidates?.length ? (
                        <ul>
                          {finalPreview.memory_candidates.slice(0, 6).map((candidate) => (
                            <li key={candidate.candidate_id}>
                              {statusLabel(candidate.kind || "memory")}
                              <small>{candidate.target_group_id} · {candidate.evidence || candidate.status}</small>
                            </li>
                          ))}
                        </ul>
                      ) : <p className="muted">No memory candidates are previewed yet.</p>}
                    </div>
                  </div>
                ) : null}
                <div className="workflow-action-row">
                  <button disabled title="Final readout generation needs context-run output and a reviewed provider action." type="button">Generate final readout (planned)</button>
                  <button
                    disabled={!contextualDetail || finalPreviewBlocked || finalPreviewAction.status === "running"}
                    onClick={queueFinalPreview}
                    type="button"
                  >
                    Queue preview review
                  </button>
                </div>
                {finalPreviewAction.message ? (
                  <div className={`action-notice ${finalPreviewAction.status}`}>
                    <strong>{finalPreviewAction.message}</strong>
                    {finalPreviewAction.payload?.review_item_path ? <small>{finalPreviewAction.payload.review_item_path}</small> : null}
                  </div>
                ) : null}
              </section>
            ) : null}
          </main>
        </div>
      </section>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
