# Mail Receipts archive-plus-live search recovery handoff

Date: 2026-08-31

Consumer: transcribe-audio Plan 0073 P5

Status: RESOLVED

## Resolution

Mail Receipts Plan 235 fixed the defect at installed checkpoint
`3fca8b7e430fcb45c231656f40a163821929650f` by following the source-owned
`attached_corpus_storage_root` with namespace validation and fail-closed
ambiguity handling. Final documentation commit
`4f270a47ce5f2c168aebf48567cf91eeabcde8c7` is upstream-even. Fresh installed
runtime readback proved `archive_plus_live`, two DuckDB backends, and real exact
participant results while the backend remained healthy at PID `64879` with
zero restarts.

Transcribe Audio commit `88786d5` removed its server-side `before:` directive,
which had selected a different backend path, and retained the historical
`as_of` check over body-free returned metadata. The unchanged frozen cohort
then completed 57 of 57 queries with zero unavailable queries and replay-equal
aggregate SHA-256
`f758d82123e0882ac489b60f9ed1e93214cceb3f5f31060315d7350f4e32a568`.
No record qualified before the historical conversation cutoffs, so Plan 0073
closed with a zero-coverage, shadow-only decision. This is no longer an open
Mail Receipts defect.

## Outcome required

Make the existing authenticated `operator-lite` public read surface return a
complete archive-plus-live result for exact participant-address searches. Do
not broaden to mailbox mutation, provider fetch, body retrieval, or a different
tenant/account.

## Authority order

1. Current Mail Receipts repository policy and accepted plans.
2. Current installed runtime and public MCP readback.
3. This handoff as a locator for the consumer failure.
4. Historical Plan 0073 receipts, which do not prove current runtime state.

The transcribe-audio user granted session-wide Mail Receipts read access. That
does not authorize Mail Receipts corpus mutation, backfill, service deployment,
or mailbox/provider writes. Obtain the needed authority before any repair with
those effects.

## Reproduce with public reads only

1. Re-anchor the installed Mail Receipts identity, branch/commit, backend
   service, authenticated `operator-lite` profile, namespace, and public source
   selector bundle.
2. Resolve the selected live corpus through `search_mail` or public corpus
   inspection and confirm its `merge_target` has
   `merge_kind=archive_plus_live` and two target corpus IDs.
3. Run a metadata-only exact participant-address query with
   `result_mode=occurrence`, lexical ranking, no rerank, no body, no summary,
   and no persisted workflow snapshot.
4. Inspect only aggregate execution metadata. The observed failure reports
   `duckdb-message-search-direct-participant-address`, meaning the exact-address
   fast path searched only the requested live registry record.
5. Repeat with `result_mode=logical_message`. The observed fallback reaches
   merged-target loading but returns zero because public inspection/search of
   the advertised archive anchor fails.

Do not place account addresses, archive IDs, query addresses, subjects, bodies,
or private hits in repository evidence. The archive selector is safely tracked
by SHA-256
`aa85a8a5cf8a5ef325b2529a4447250de67cd99b15f1989f15ea409bfeb51e21`.

## Current source diagnosis

In Mail Receipts
`/home/ecochran76/workspace.local/mail-receipts/src/unified_mail/api/service.py`:

- `UnifiedMailService.search_corpus` resolves the multi-corpus merge target,
  then constructs `exact_email_message_search_table` from only
  `registry_record` when logical-message projection is not requested.
- That direct table branch runs before
  `_registry_default_merge_target_search_service`, so an exact participant
  query can advertise two merge targets while searching only the live member.
- `_registry_default_merge_target_search_service` needs a resolvable archive
  registry record and preferred retrieval index. The current public archive
  anchor cannot be inspected or searched, so the method falls back.
- `_load_registered_messages_for_merge_target` loops both advertised target
  IDs, but the unresolved archive contributes no messages; the registered live
  incremental corpus is retrieval-pending and returned no historical hits.

The consumer now fails closed when an archive-plus-live response reports the
single-corpus direct participant-address effect. This guard is in
`/home/ecochran76/workspace.local/transcribe-audio/mail_receipts_mcp_reader.py`.

## Acceptance evidence

- The archive anchor resolves through the public corpus registry in the same
  tenant and namespace as the live source.
- An exact participant-address search reports a merged execution source and can
  return archive and live occurrences without body retrieval.
- A regression test covers exact-address search against a two-member source
  family; it must fail before the repair and pass afterward.
- Existing single-corpus exact-address search remains fast and unchanged.
- Public selector discovery remains complete and provider-read/mutation flags
  remain false.
- Fresh installed-runtime smoke reproduces the repaired behavior; source tests
  alone are not acceptance.

## Hard stops

- No direct storage enumeration or private-storage bypass.
- No mailbox/provider call, backfill, or corpus registration mutation without
  separate authority.
- No message bodies, subjects, attachments, private addresses, or raw hits in
  logs, tests, screenshots, commits, or handoff updates.
- Do not treat a zero-hit response as success unless execution metadata proves
  every advertised source-family member was searched.

## Historical bounded repair packet

Implement and test one Mail Receipts change that makes the exact-address path
merge-aware and restores/resolves the existing archive anchor through the
public registry. Install only after review and explicit write/deployment
authority, then repeat the two public aggregate smokes above. Return the new
commit, installed identity, test counts, execution-source metadata, and zero-
effect readback to Plan 0073; do not rerun the 57-query private pilot from the
Mail Receipts repository.
