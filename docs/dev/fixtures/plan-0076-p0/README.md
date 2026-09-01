# Plan 0076 P0 Redacted Fixtures

`adversarial-directory.json` is a provider-free contract fixture. It contains
two distinct same-name records, one shared address, repeated organization
strings, and a duplicate calendar copy. Expected effects are all zero; the
fixture may create reconciliation and organization proposals only.

`directory-index.schema.json` freezes the bounded public response shape. The
runtime has stricter event-specific validation in
`identity_learning_ledger.py`; this schema describes the stable API envelope.
