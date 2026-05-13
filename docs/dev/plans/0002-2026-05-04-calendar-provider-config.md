# Plan 0002 | Calendar Provider Configuration

State: CLOSED

Lane: P02

## Scope

Make calendar lookup provider order and tenant/profile selection explicit.

## Non-Goals

- No write operations to calendars.
- No routing/deposition based on calendar data yet.

## Current State

Calendar lookup has an explicit provider config path. CLI flags and watcher `calendar` config can select ordered providers, `gog` account/client selectors, and a `gws` config directory. The default order is `gog,gws,google-api`, so local tenant-aware tools are tried before the built-in Google API fallback. Event metadata now includes `matching_calendars` for overlapping events found on accessible calendars. A temp-location watcher `--run-once` smoke validated structured config expansion and `gog` calendar lookup.

## Work Items

- Done: define `CalendarProviderConfig`.
- Done: support ordered provider lists in watcher config and CLI overrides.
- Done: map `gog` provider fields to `--account` and `--client`.
- Done: map `gws` provider fields to `GOOGLE_WORKSPACE_CLI_CONFIG_DIR`.
- Done: preserve built-in Google API fallback without eagerly triggering OAuth.
- Done: improve logs for provider selection and failure reasons.
- Done: include matching accessible-calendar context in event metadata for downstream readouts.
- Done: manual run-once watcher test with calendar lookup enabled.

## Acceptance Criteria

- Config can select default, `gog`, `gws`, or Google API fallback.
- Tenant/profile fields are visible in config and not hardcoded.
- Existing `--use-calendar` usage still works.

## Validation

- Unit tests for command construction.
- Manual run-once watcher test with calendar lookup enabled.
