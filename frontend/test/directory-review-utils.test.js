import assert from "node:assert/strict";
import test from "node:test";

import {
  createInFlightGate,
  directoryRowCounts,
  filterDirectoryRows
} from "../src/directory-review-utils.js";

const rows = [
  {
    primary_name: "Needs review",
    review_leads: [{ review_state: "unreviewed" }, { review_state: "accepted" }]
  },
  {
    primary_name: "Already decided",
    review_leads: [{ review_state: "accepted" }, { review_state: "deferred" }]
  },
  { primary_name: "Evidence only", review_leads: [] }
];

test("directory scopes distinguish actionable, decided, and all rows", () => {
  assert.deepEqual(
    filterDirectoryRows(rows, "actionable").map((row) => row.primary_name),
    ["Needs review"]
  );
  assert.deepEqual(
    filterDirectoryRows(rows, "decided").map((row) => row.primary_name),
    ["Already decided"]
  );
  assert.equal(filterDirectoryRows(rows, "all").length, 3);
  assert.deepEqual(directoryRowCounts(rows), { actionable: 1, decided: 1, all: 3 });
});

test("an in-flight review gate suppresses a duplicate until release", () => {
  const gate = createInFlightGate();
  assert.equal(gate.begin(), true);
  assert.equal(gate.begin(), false);
  gate.end();
  assert.equal(gate.begin(), true);
});
