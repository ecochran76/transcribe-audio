# Plan 0073 P0 contract fixtures

This directory freezes the deterministic, privacy-preserving contracts for
Mail Receipts relationship and contextual-role evidence.

- `contract-catalog.json` mirrors `mail_relationship_contracts.contract()`.
- `portable-artifacts.json` supplies one valid synthetic example for each
  versioned portable artifact.
- `schemas/` contains one closed top-level JSON Schema per artifact.
- `discovery-scenarios.json` freezes the expected behavior for one-way mail,
  bidirectional correspondence, recurring thread coparticipation, duplicate
  source copies, conflicting structured roles, shared addresses, and historical
  `as_of` exclusion.

Every address uses the reserved `.test` domain. The fixtures contain no message
body, subject, attachment, provider identifier, private namespace, filesystem
path, or live corpus selector. They authorize no Mail Receipts or provider
access and every effect count remains zero.

Validate the executable contract with:

```bash
.venv/bin/python -m pytest -q tests/test_mail_relationship_contracts.py
```
