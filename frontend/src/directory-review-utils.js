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
