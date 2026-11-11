# Managed recording work-item routing

This repository uses repo-native planning artifacts under docs/dev/plans. The
cross-repository managed-recording implementation is centrally tracked in the LocalAI
repository qualification/docs/specs/0005-managed-diarized-recordings.md and one
artifact per ticket under qualification/docs/tickets. Its INDEX.md and execution
ledger own dependencies, owner, source custody and acceptance. Plan0095 is this
consumer milestone and links the accepted design; do not create competing issues.

State mapping: ready-for-agent, in-progress, blocked, complete. Source closure is
not native or deployment acceptance; those have separate tickets. Primary owns
shared tracker writes and one substantive work item per agent. Final code review
and one post-review TDD pass follow closure of the requested ticket set.
