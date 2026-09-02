import assert from "node:assert/strict";
import test from "node:test";

import {
  createInFlightGate,
  findUniqueAcceptedPersonTarget,
  flattenDirectoryReviewRows,
  hasDirectoryReviewDecision,
  isCompletePersonName,
  personTargetDisplayLabel
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

test("a unique accepted name or alias match is suggested without guessing ambiguity", () => {
  const targets = [
    { id: "person-eric", label: "Eric Cochran", aliases: ["Eric W Cochran"] },
    { id: "person-other", label: "Other Person", aliases: ["Ecochran"] }
  ];
  assert.equal(findUniqueAcceptedPersonTarget(
    { primary_name: "Ecochran", aliases: ["Eric Cochran"] },
    targets
  ), null);
  assert.equal(findUniqueAcceptedPersonTarget(
    { primary_name: "Ecochran", aliases: ["Eric Cochran"] },
    targets.slice(0, 1)
  )?.id, "person-eric");
  assert.equal(findUniqueAcceptedPersonTarget(
    { accepted_person_id: "person-eric", primary_name: "Anything" },
    targets
  )?.id, "person-eric");
});

test("person targets show normalized names and disclose incomplete identities", () => {
  assert.equal(personTargetDisplayLabel({
    display_name: "Basia Cienkosz",
    label: "Cienkosz, Basia",
    name_completeness: "complete"
  }), "Basia Cienkosz");
  assert.equal(personTargetDisplayLabel({
    display_name: "Dr. Stefl",
    label: "Dr. Stefl",
    name_completeness: "incomplete"
  }), "Dr. Stefl — incomplete name");
  assert.equal(personTargetDisplayLabel({ label: "Legacy Person" }), "Legacy Person");
});

test("target suggestions still match preserved source names after display normalization", () => {
  const target = {
    id: "person-zachary",
    display_name: "Zachary Gates",
    label: "Zachary Gates",
    primary_name: "Gates, Zachary",
    aliases: ["zgates@example.com"]
  };
  assert.equal(findUniqueAcceptedPersonTarget(
    { primary_name: "Gates, Zachary", aliases: [] },
    [target]
  )?.id, "person-zachary");
});

test("person-name candidates keep organization labels out of target matching", () => {
  const targets = [{ id: "person-robert", label: "Robert McElmurry", aliases: ["Precision Land Solutions"] }];
  assert.equal(findUniqueAcceptedPersonTarget({
    primary_name: "Robert",
    aliases: ["Precision Land Solutions"],
    person_name_candidates: []
  }, targets), null);
  assert.equal(findUniqueAcceptedPersonTarget({
    primary_name: "Robert",
    aliases: ["Precision Land Solutions", "Robert McElmurry"],
    person_name_candidates: ["Robert McElmurry"]
  }, targets)?.id, "person-robert");
});

test("new canonical people require a complete non-organization name", () => {
  assert.equal(isCompletePersonName("Jason Colbourne", "Duraflex Solutions"), true);
  assert.equal(isCompletePersonName("Jason", "Duraflex Solutions"), false);
  assert.equal(isCompletePersonName("jason@example.test", "Duraflex Solutions"), false);
  assert.equal(isCompletePersonName("Duraflex Solutions", "Duraflex Solutions"), false);
});

test("ambiguous response reconciliation requires the exact review idempotency key", () => {
  const rows = [{
    review_leads: [{
      hypothesis_id: "hypothesis-1",
      decision_history: [{ idempotency_key: "review-key-1" }]
    }]
  }];
  assert.equal(hasDirectoryReviewDecision(rows, "hypothesis-1", "review-key-1"), true);
  assert.equal(hasDirectoryReviewDecision(rows, "hypothesis-1", "review-key-2"), false);
  assert.equal(hasDirectoryReviewDecision(rows, "hypothesis-2", "review-key-1"), false);
});
