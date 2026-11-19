"""Versioned, effect-free recording retention and admission policy.

Time inputs are qualified StableTime observations at their measurement instants,
never extrapolated guest clocks. Decisions grant no filesystem mutation authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

POLICY_VERSION = 'recording-storage-v1'
GIB = 1024 ** 3


class StorageClass(str, Enum):
    LIVE_METADATA = 'live_metadata'
    RUNTIME = 'runtime'
    ACTIVE_TRANSIENT = 'active_transient'
    CANONICAL_SOURCE = 'canonical_source'
    HISTORICAL_DERIVATIVE = 'historical_derivative'
    COMPLETED_SCRATCH = 'completed_scratch'
    UNRESOLVED = 'unresolved'
    UNCLASSIFIED = 'unclassified'
    CLOSED_EVIDENCE = 'closed_evidence'
    PRIVATE_CACHE = 'private_cache'
    QUALIFICATION_PAYLOAD = 'qualification_payload'
    RESTORE_POINT = 'restore_point'


@dataclass(frozen=True)
class TimeObservation:
    utc_seconds: float
    uncertainty_seconds: float = 0
    qualified: bool = True

    @property
    def valid(self) -> bool:
        return (self.qualified and isfinite(self.utc_seconds)
                and isfinite(self.uncertainty_seconds) and self.uncertainty_seconds >= 0)


@dataclass(frozen=True)
class ObjectState:
    classification: StorageClass | str
    ownership_known: bool = True
    active: bool = False
    pinned: bool = False
    consumers_qualified: bool = True
    completed_at: TimeObservation | None = None
    inactive_since: TimeObservation | None = None
    publication_verified: bool = False
    recovery_verified: bool = False


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    policy_version: str = POLICY_VERSION


def decide(item: ObjectState, now: TimeObservation | None = None) -> Decision:
    """Classify eligibility only; cold_move never means source erasure/retry."""
    try:
        kind = StorageClass(item.classification)
    except (ValueError, TypeError):
        return Decision('quarantine', 'unknown_classification')
    if not item.ownership_known or kind == StorageClass.UNCLASSIFIED:
        return Decision('quarantine', 'unreconciled_ownership_or_classification')
    if item.active or item.pinned:
        return Decision('hold', 'active_or_pinned')
    if not item.consumers_qualified:
        return Decision('hold', 'consumer_compatibility_unqualified')
    if kind in (StorageClass.COMPLETED_SCRATCH, StorageClass.UNRESOLVED):
        since = item.completed_at if kind == StorageClass.COMPLETED_SCRATCH else item.inactive_since
        if now is None or since is None or not now.valid or not since.valid:
            return Decision('hold', 'trusted_time_unavailable')
        age = now.utc_seconds - now.uncertainty_seconds - since.utc_seconds - since.uncertainty_seconds
        threshold = 48 * 3600 if kind == StorageClass.COMPLETED_SCRATCH else 7 * 86400
        if age < threshold:
            return Decision('hold', 'grace_or_inactivity_incomplete')
        if kind == StorageClass.UNRESOLVED:
            return Decision('cold_move', 'preserve_unresolved_payload_and_checkpoint')
        if not item.publication_verified or not item.recovery_verified:
            return Decision('hold', 'publication_or_recovery_unverified')
        return Decision('evict_scratch', 'completed_grace_satisfied')
    if kind == StorageClass.PRIVATE_CACHE:
        return Decision('evict_cache', 'inactive_owned_regenerable_cache')
    if kind in (StorageClass.CANONICAL_SOURCE, StorageClass.HISTORICAL_DERIVATIVE,
                StorageClass.CLOSED_EVIDENCE, StorageClass.QUALIFICATION_PAYLOAD):
        return Decision('cold_move', 'retain_exact_bytes_and_identity')
    return Decision('retain', 'durable_or_separately_governed_retention')


@dataclass(frozen=True)
class BudgetState:
    transient_bytes: int
    nvme_free_bytes: int
    pass_written_bytes: int = 0
    pass_started_at: TimeObservation | None = None


@dataclass(frozen=True)
class ReservationDecision:
    admitted: bool
    reason: str
    transient_bytes: int = 0
    nvme_write_bytes: int = 0
    payload_write_bytes: int = 0
    policy_version: str = POLICY_VERSION


def reserve(state: BudgetState, *, transient_bytes: int, nvme_write_bytes: int,
            payload_write_bytes: int, duration_seconds: float,
            now: TimeObservation | None, background: bool = True) -> ReservationDecision:
    """Evaluate an atomic reservation against totals including outstanding leases.

    Caller serializes accounting, debits every retry's payload writes, and enforces
    the admitted duration at safe checkpoints using fresh external observations.
    """
    values = (state.transient_bytes, state.nvme_free_bytes, state.pass_written_bytes,
              transient_bytes, nvme_write_bytes, payload_write_bytes)
    if any(type(v) is not int or v < 0 for v in values) or not isfinite(duration_seconds) or duration_seconds < 0:
        return ReservationDecision(False, 'invalid_budget_input')
    if state.transient_bytes + transient_bytes > 20 * GIB:
        return ReservationDecision(False, 'transient_budget')
    if state.nvme_free_bytes - nvme_write_bytes < (100 * GIB if background else 0):
        return ReservationDecision(False, 'nvme_reserve')
    if state.pass_written_bytes + payload_write_bytes > 16 * GIB:
        return ReservationDecision(False, 'pass_write_budget')
    start = state.pass_started_at
    if now is None or start is None or not now.valid or not start.valid:
        return ReservationDecision(False, 'trusted_time_unavailable')
    if now.utc_seconds - now.uncertainty_seconds < start.utc_seconds + start.uncertainty_seconds:
        return ReservationDecision(False, 'time_order_uncertain')
    elapsed = now.utc_seconds + now.uncertainty_seconds - start.utc_seconds + start.uncertainty_seconds
    if elapsed + duration_seconds > 1800:
        return ReservationDecision(False, 'pass_time_budget')
    return ReservationDecision(True, 'reserved', transient_bytes, nvme_write_bytes, payload_write_bytes)
