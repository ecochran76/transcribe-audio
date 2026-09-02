export function flattenDirectoryReviewRows(rows, reviewState = "unreviewed") {
  const source = Array.isArray(rows) ? rows : [];
  return source.flatMap((item) => {
    const leads = Array.isArray(item.review_leads) ? item.review_leads : [];
    const itemId = item.entity_id || item.organization_id || item.person_id || item.primary_name || "directory-row";
    return leads
      .filter((lead) => !reviewState || lead.review_state === reviewState)
      .map((lead) => ({
        item,
        lead,
        row_id: `${itemId}:${lead.hypothesis_id}:${lead.projection_version || ""}`
      }));
  });
}

export function createInFlightGate() {
  let active = false;
  return {
    begin() {
      if (active) return false;
      active = true;
      return true;
    },
    end() {
      active = false;
    }
  };
}

export function personTargetDisplayLabel(target) {
  const displayName = String(
    target?.display_name || target?.label || target?.primary_name || "Unnamed person"
  ).trim();
  return target?.name_completeness && target.name_completeness !== "complete"
    ? `${displayName} — incomplete name`
    : displayName;
}

function normalizedNames(values) {
  return new Set((values || []).map((value) => String(value || "").trim().toLocaleLowerCase()).filter(Boolean));
}

export function findUniqueAcceptedPersonTarget(item, targets) {
  const acceptedId = String(item?.accepted_person_id || "");
  if (acceptedId) return (targets || []).find((target) => target.id === acceptedId) || null;
  const itemNames = normalizedNames([item?.primary_name, ...(item?.aliases || [])]);
  const matches = (targets || []).filter((target) => {
    const targetNames = normalizedNames([
      target.display_name,
      target.label,
      target.primary_name,
      ...(target.aliases || [])
    ]);
    return [...itemNames].some((name) => targetNames.has(name));
  });
  return matches.length === 1 ? matches[0] : null;
}

export function hasDirectoryReviewDecision(rows, hypothesisId, idempotencyKey) {
  return (rows || []).some((item) => (item.review_leads || []).some((lead) => (
    lead.hypothesis_id === hypothesisId
    && (lead.decision_history || []).some((decision) => decision.idempotency_key === idempotencyKey)
  )));
}
