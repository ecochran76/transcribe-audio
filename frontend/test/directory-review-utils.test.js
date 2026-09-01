import assert from "node:assert/strict";
import test from "node:test";

import {
  createInFlightGate,
  flattenDirectoryReviewRows
} from "../src/directory-review-utils.js";

test("approval rows flatten every unreviewed hypothesis without collapsing contacts", () => {
  const reviewRows = flattenDirectoryReviewRows([
    {
      person_id: "person-1",
      primary_name: "One Contact",
      review_leads: [
        { hypothesis_id: "affiliation-1", projection_version: "1", review_state: "unreviewed" },
        { hypothesis_id: "role-1", projection_version: "2", review_state: "unreviewed" },
        { hypothesis_id: "accepted-1", projection_version: "1", review_state: "accepted" }
      ]
    },
    {
      person_id: "person-2",
      primary_name: "No Pending Work",
      review_leads: [{ hypothesis_id: "accepted-2", projection_version: "1", review_state: "accepted" }]
    }
  ]);

  assert.deepEqual(reviewRows.map(({ item, lead }) => [item.primary_name, lead.hypothesis_id]), [
    ["One Contact", "affiliation-1"],
    ["One Contact", "role-1"]
  ]);
  assert.deepEqual(reviewRows.map(({ row_id }) => row_id), [
    "person-1:affiliation-1:1",
    "person-1:role-1:2"
  ]);
});

test("an in-flight review gate suppresses a duplicate until release", () => {
  const gate = createInFlightGate();
  assert.equal(gate.begin(), true);
  assert.equal(gate.begin(), false);
  gate.end();
  assert.equal(gate.begin(), true);
});
