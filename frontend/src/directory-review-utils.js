export function filterDirectoryRows(rows, scope) {
  const source = Array.isArray(rows) ? rows : [];
  if (scope === "all") return source;
  return source.filter((row) => {
    const leads = Array.isArray(row.review_leads) ? row.review_leads : [];
    const hasUnreviewed = leads.some((lead) => lead.review_state === "unreviewed");
    if (scope === "actionable") return hasUnreviewed;
    if (scope === "decided") return leads.length > 0 && !hasUnreviewed;
    return true;
  });
}

export function directoryRowCounts(rows) {
  return {
    actionable: filterDirectoryRows(rows, "actionable").length,
    decided: filterDirectoryRows(rows, "decided").length,
    all: Array.isArray(rows) ? rows.length : 0
  };
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
