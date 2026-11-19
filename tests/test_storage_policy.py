from storage_policy import ObjectState, StorageClass, TimeObservation, decide


def test_completed_grace_uses_conservative_external_time():
    item = ObjectState(StorageClass.COMPLETED_SCRATCH, completed_at=TimeObservation(0),
                       publication_verified=True, recovery_verified=True)
    assert decide(item, TimeObservation(172800, 1)).action == 'hold'
    assert decide(item, TimeObservation(172801, 1)).action == 'evict_scratch'

import pytest
from dataclasses import replace
from storage_policy import BudgetState, GIB, POLICY_VERSION, reserve


@pytest.mark.parametrize('change,action', [({'pinned': True}, 'hold'),
    ({'active': True}, 'hold'), ({'ownership_known': False}, 'quarantine'),
    ({'classification': 'surprise'}, 'quarantine'),
    ({'consumers_qualified': False}, 'hold')])
def test_ownership_and_usage_exclude_retirement(change, action):
    assert decide(replace(ObjectState(StorageClass.PRIVATE_CACHE), **change)).action == action


def test_unresolved_is_cold_only_and_never_retried():
    item = ObjectState(StorageClass.UNRESOLVED, inactive_since=TimeObservation(0))
    assert decide(item, TimeObservation(604799)).action == 'hold'
    assert decide(item, TimeObservation(604800)).action == 'cold_move'
    assert decide(item, TimeObservation(999999, qualified=False)).action == 'hold'
    assert decide(item).reason == 'trusted_time_unavailable'


@pytest.mark.parametrize('kind', list(StorageClass))
def test_every_accepted_class_has_versioned_decision(kind):
    assert decide(ObjectState(kind)).policy_version == POLICY_VERSION


def test_completed_requires_recovery_and_qualified_historical_observation():
    item = ObjectState(StorageClass.COMPLETED_SCRATCH, completed_at=TimeObservation(0))
    assert decide(item, TimeObservation(999999)).reason == 'publication_or_recovery_unverified'
    item = replace(item, completed_at=TimeObservation(0, qualified=False))
    assert decide(item, TimeObservation(999999)).reason == 'trusted_time_unavailable'


@pytest.mark.parametrize('changes,reason', [
    ({'transient_bytes': 20 * GIB + 1}, 'transient_budget'),
    ({'nvme_write_bytes': 1}, 'nvme_reserve'),
    ({'payload_write_bytes': 16 * GIB + 1}, 'pass_write_budget'),
    ({'duration_seconds': 1801}, 'pass_time_budget'),
    ({'now': None}, 'trusted_time_unavailable'),
    ({'transient_bytes': -1}, 'invalid_budget_input')])
def test_reservation_limits(changes, reason):
    args = dict(transient_bytes=0, nvme_write_bytes=0, payload_write_bytes=0,
                duration_seconds=0, now=TimeObservation(0))
    args.update(changes)
    result = reserve(BudgetState(0, 100 * GIB, pass_started_at=TimeObservation(0)), **args)
    assert not result.admitted
    assert result.reason == reason


def test_exact_limits_and_outstanding_reservations_are_accounted():
    state = BudgetState(19 * GIB, 101 * GIB, 15 * GIB, TimeObservation(0))
    args = dict(transient_bytes=GIB, nvme_write_bytes=GIB, payload_write_bytes=GIB,
                duration_seconds=1790, now=TimeObservation(10))
    assert reserve(state, **args).admitted
    assert not reserve(replace(state, pass_written_bytes=16 * GIB), **args).admitted
    assert not reserve(state, **{**args, 'now': TimeObservation(10, 1)}).admitted
