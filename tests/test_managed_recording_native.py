"""Native harness capacity selection must fail closed without model effects."""
import importlib.util
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest


def harness():
    spec = importlib.util.spec_from_file_location('native_harness', Path(__file__).resolve().parents[1]/'scripts/managed_recording_native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_capacity_provider_keeps_source_and_fresh_observation():
    module = harness()
    with tempfile.TemporaryDirectory() as folder:
        token = Path(folder)/'token'
        token.write_text('private-token')
        token.chmod(0o600)
        config = dict(endpoint='http://127.0.0.1:18157', host_id='COOPER',
            gpu_uuid='GPU-6e6ec07d-41b2-f2a4-6ce5-4be996267477',
            token_path=str(token), expected_source='windows-dxcore')
        with patch('inference_manager.windows_telemetry.WindowsGpuTelemetry') as factory:
            factory.return_value.sample.return_value = dict(observed_at=123,
                source='windows-dxcore', age_seconds=.25, gpu=dict(free_bytes=100, used_bytes=200))
            result = module.gpu(config)
            assert result['observed_at'] == 123
            assert result['age_seconds'] == .25
            assert result['free_bytes'] == 100
            assert factory.call_args.kwargs['expected_source'] == 'windows-dxcore'
            factory.return_value.sample.side_effect = RuntimeError('stale')
            with pytest.raises(RuntimeError, match='stale'):
                module.gpu(config)
        token.chmod(0o644)
        with pytest.raises(ValueError, match='private'):
            module.gpu(config)


def test_successor_limits_preserve_old_packets_and_bound_two_new_loads():
    module = harness()
    assert module.native_limits({'max_physical_stage_load_attempts': 4,
                                 'limits': {'case_seconds': 600, 'cleanup_seconds': 15}}) == (4, 600, 15)
    assert module.native_limits({'max_physical_stage_load_attempts': 6,
                                 'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}) == (6, 160, 15)
    for attempts, seconds, cleanup in [(16, 160, 15), (6, 161, 15), (6, 160, 16), (6, 0, 15)]:
        with pytest.raises(ValueError, match='bounded native'):
            module.native_limits({'max_physical_stage_load_attempts': attempts,
                                  'limits': {'case_seconds': seconds, 'cleanup_seconds': cleanup}})


def test_prepared_controller_telemetry_file_is_private_and_explicit(tmp_path):
    import json
    module = harness()
    path = tmp_path/'telemetry.json'
    config = dict(endpoint='http://127.0.0.1:18157', host_id='COOPER',
                  gpu_uuid='GPU-6e6ec07d-41b2-f2a4-6ce5-4be996267477',
                  expected_source='windows-dxcore', token_path=str(tmp_path/'bearer.token'))
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    assert module.packet_telemetry({'telemetry_config_path': str(path)}) == config
    path.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        module.packet_telemetry({'telemetry_config_path': str(path)})


def test_contract_corrected_case_uses_qualified_five_second_age(tmp_path):
    module = harness()
    assert module.native_limits({'max_physical_stage_load_attempts': 7,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (7, 140, 15)
    token = tmp_path/'token'
    token.write_text('private-token')
    token.chmod(0o600)
    config = dict(endpoint='http://127.0.0.1:18157', host_id='COOPER',
        gpu_uuid='GPU-6e6ec07d-41b2-f2a4-6ce5-4be996267477',
        token_path=str(token), expected_source='windows-dxcore')
    with patch('inference_manager.windows_telemetry.WindowsGpuTelemetry') as factory:
        factory.return_value.sample.return_value = dict(source='windows-dxcore',
            age_seconds=3, observed_at=123, gpu=dict(free_bytes=100, used_bytes=200))
        assert module.gpu(config, max_age_seconds=5)['age_seconds'] == 3
        assert factory.call_args.kwargs['max_age_seconds'] == 5
        for age in (0, 5.01, float('nan'), True):
            with pytest.raises(ValueError, match='age'):
                module.gpu(config, max_age_seconds=age)


def test_rejection_descriptor_excludes_token_and_does_not_supply_capacity(tmp_path):
    module = harness()
    token = tmp_path/'token'
    token.write_text('private-token')
    token.chmod(0o600)
    config = dict(endpoint='http://127.0.0.1:18157', host_id='COOPER',
        gpu_uuid='GPU-6e6ec07d-41b2-f2a4-6ce5-4be996267477',
        token_path=str(token), expected_source='windows-dxcore')
    with patch('inference_manager.windows_telemetry.WindowsGpuTelemetry') as factory:
        factory.return_value.sample.side_effect = RuntimeError('stale')
        with pytest.raises(RuntimeError) as caught:
            module.gpu(config, max_age_seconds=5)
    assert caught.value.capacity_rejection == dict(error_type='RuntimeError',
        expected_source='windows-dxcore', age_limit_seconds=5,
        rejected_sample_available=False)


def test_explicit_release_successor_preserves_history_and_two_stage_proof():
    module = harness()
    assert module.native_limits({'max_physical_stage_load_attempts': 7,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (7, 140, 15)
    assert module.native_limits({'max_physical_stage_load_attempts': 8,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (8, 140, 15)
    assert module.native_limits({'max_physical_stage_load_attempts': 9,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (9, 140, 15)
    assert module.native_limits({'max_physical_stage_load_attempts': 11,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (11, 140, 15)
    assert module.native_limits({'max_physical_stage_load_attempts': 12,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (12, 140, 15)
    with pytest.raises(ValueError, match='bounded native'):
        module.native_limits({'max_physical_stage_load_attempts': 16,
            'limits': {'case_seconds': 140, 'cleanup_seconds': 15}})


def test_new_asr_packet_does_not_replenish_consumed_recording_budget(tmp_path):
    """A new bounded packet may continue; the original twelve-load packet cannot."""
    import json
    module = harness()
    assert module.native_limits({'max_physical_stage_load_attempts': 13,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}) == (13, 140, 15)
    ledger = tmp_path/'attempts.json'
    history = json.dumps([{'stage': 'diarization', 'attempt': n} for n in range(12)])
    ledger.write_text(history)
    packet = tmp_path/'original.json'
    packet.write_text(json.dumps({'localai_qualification': str(tmp_path),
        'max_physical_stage_load_attempts': 12,
        'limits': {'case_seconds': 140, 'cleanup_seconds': 15}}))
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        module.worker_exec(packet, ['/bin/false', '--stage', 'asr'])
    assert ledger.read_text() == history


def test_diarization_recovery_keeps_consumed_pilot_budget_and_forbids_asr(tmp_path):
    """A scoped sixteenth start never replenishes the original pilot or ASR."""
    import json
    module = harness()
    config = {'localai_qualification': str(tmp_path),
              'max_physical_stage_load_attempts': 16,
              'allowed_stages': ['diarization'],
              'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}
    assert module.native_limits(config) == (16, 160, 15)
    ledger = tmp_path/'attempts.json'
    history = json.dumps([{'stage': 'diarization', 'attempt': n} for n in range(15)])
    ledger.write_text(history)
    successor = tmp_path/'successor.json'
    successor.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='stage not allowed'):
        module.worker_exec(successor, ['/bin/false', '--stage', 'asr'])
    original = tmp_path/'original-pilot.json'
    original.write_text(json.dumps({**config, 'max_physical_stage_load_attempts': 15}))
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        module.worker_exec(original, ['/bin/false', '--stage', 'diarization'])
    assert ledger.read_text() == history


def test_sixteenth_start_requires_retained_fifteen_attempt_history(tmp_path):
    import json
    module = harness()
    ledger = tmp_path/'attempts.json'
    history = json.dumps([{'stage': 'diarization', 'attempt': n} for n in range(14)])
    ledger.write_text(history)
    packet = tmp_path/'recovery.json'
    packet.write_text(json.dumps({'localai_qualification': str(tmp_path),
        'max_physical_stage_load_attempts': 16, 'allowed_stages': ['diarization'],
        'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}))
    with patch('subprocess.check_output', side_effect=AssertionError('external capacity query reached')):
        with pytest.raises(RuntimeError, match='retained fifteen-load history'):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
    assert ledger.read_text() == history


def test_mixed_dia_cohort_retains_sixteen_starts_and_rejects_asr(tmp_path):
    import hashlib
    import json
    module = harness()
    history = [{'stage': 'diarization', 'attempt': n} for n in range(16)]
    pin = hashlib.sha256(json.dumps(history, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()
    config = {'localai_qualification': str(tmp_path),
              'max_physical_stage_load_attempts': 19,
              'allowed_stages': ['diarization'],
              'budget_contract': 'staged-mixed-dia-v1',
              'retained_physical_starts': 16, 'retained_history_sha256': pin,
              'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}
    assert module.native_limits(config) == (19, 160, 15)
    for field in ('budget_contract', 'retained_physical_starts', 'retained_history_sha256'):
        with pytest.raises(ValueError, match='bounded native'):
            module.native_limits({k: v for k, v in config.items() if k != field})
    ledger = tmp_path/'attempts.json'
    original = json.dumps(history)
    ledger.write_text(original)
    packet = tmp_path/'mixed.json'
    packet.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='stage not allowed'):
        module.worker_exec(packet, ['/bin/false', '--stage', 'asr'])
    # A changed retained prefix must fail before any external capacity query.
    history[0]['attempt'] = 'changed'
    ledger.write_text(json.dumps(history))
    changed = ledger.read_text()
    with patch('subprocess.check_output', side_effect=AssertionError('external capacity query reached')):
        with pytest.raises(RuntimeError, match='retained sixteen-load history'):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
    assert ledger.read_text() == changed
    ledger.write_text(original)
    old = tmp_path/'recovery16.json'
    old.write_text(json.dumps({**config, 'max_physical_stage_load_attempts': 16}))
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        module.worker_exec(old, ['/bin/false', '--stage', 'diarization'])
    assert ledger.read_text() == original


def test_prepared_mixed_digest_roundtrips_guard_before_capacity_or_effects(tmp_path):
    import hashlib
    import json
    module = harness()
    history = [{'stage': 'diarization', 'attempt': n} for n in range(16)]
    ledger = tmp_path/'attempts.json'
    ledger.write_text(json.dumps(history, indent=2))
    original = ledger.read_bytes()
    pin = module.mixed_retained_history_sha256(history)
    raw_pin = hashlib.sha256(original).hexdigest()
    assert pin != raw_pin
    config = {'localai_qualification': str(tmp_path),
              'max_physical_stage_load_attempts': 19,
              'allowed_stages': ['diarization'],
              'budget_contract': 'staged-mixed-dia-v1',
              'retained_physical_starts': 16, 'retained_history_sha256': pin,
              'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}
    packet = tmp_path/'mixed.json'
    class CapacityReached(Exception):
        pass
    packet.write_text(json.dumps({**config, 'retained_history_sha256': raw_pin}))
    with patch.object(module, 'gpu', side_effect=CapacityReached) as capacity:
        with pytest.raises(RuntimeError, match='retained sixteen-load history'):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
        capacity.assert_not_called()
        packet.write_text(json.dumps(config))
        with pytest.raises(CapacityReached):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
        assert capacity.call_count == 1
    assert ledger.read_bytes() == original
    assert module.mixed_retained_history_sha256(history + [{'stage': 'diarization'}]) == pin
    with pytest.raises(ValueError, match='sixteen'):
        module.mixed_retained_history_sha256(history[:15])


def test_v3_cohort_retains_nineteen_starts_without_renewing_old_budget(tmp_path):
    import json
    module = harness()
    history = [{'stage': 'diarization', 'attempt': n} for n in range(19)]
    pin = module.independent_receivers_history_sha256(history)
    config = {'localai_qualification': str(tmp_path),
              'max_physical_stage_load_attempts': 22,
              'allowed_stages': ['diarization'],
              'budget_contract': 'independent-clients-v3-mixed-v1',
              'measurement_layout': 'INDEPENDENT_CLIENT_RECEIVERS_V3',
              'retained_physical_starts': 19, 'retained_history_sha256': pin,
              'limits': {'case_seconds': 160, 'cleanup_seconds': 15}}
    assert module.native_limits(config) == (22, 160, 15)
    for invalid in ({'retained_physical_starts': 19.0},
                    {'measurement_layout': 'QUEUED_OFFERS_V2'}):
        with pytest.raises(ValueError, match='bounded native'):
            module.native_limits({**config, **invalid})
    assert module.independent_receivers_history_sha256(history + [{'stage': 'diarization'}]) == pin
    with pytest.raises(ValueError, match='nineteen'):
        module.independent_receivers_history_sha256(history[:18])
    for field in ('budget_contract', 'measurement_layout', 'retained_physical_starts',
                  'retained_history_sha256'):
        with pytest.raises(ValueError, match='bounded native'):
            module.native_limits({k: v for k, v in config.items() if k != field})
    for attempts in (20, 21):
        with pytest.raises(ValueError, match='bounded native'):
            module.native_limits({**config, 'max_physical_stage_load_attempts': attempts})
    ledger = tmp_path/'attempts.json'
    original = json.dumps(history, indent=2)
    ledger.write_text(original)
    packet = tmp_path/'v3.json'
    packet.write_text(json.dumps(config))
    class CapacityReached(Exception):
        pass
    with patch.object(module, 'gpu', side_effect=CapacityReached) as capacity:
        with pytest.raises(ValueError, match='stage not allowed'):
            module.worker_exec(packet, ['/bin/false', '--stage', 'asr'])
        capacity.assert_not_called()
        with pytest.raises(CapacityReached):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
        assert capacity.call_count == 1
        history[18]['attempt'] = 'changed'
        ledger.write_text(json.dumps(history))
        capacity.reset_mock()
        with pytest.raises(RuntimeError, match='retained nineteen-load history'):
            module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
        capacity.assert_not_called()
    ledger.write_text(original)
    old = {**config, 'max_physical_stage_load_attempts': 19,
           'budget_contract': 'staged-mixed-dia-v1', 'retained_physical_starts': 16,
           'retained_history_sha256': module.mixed_retained_history_sha256(history)}
    packet.write_text(json.dumps(old))
    with pytest.raises(RuntimeError, match='allowance exhausted'):
        module.worker_exec(packet, ['/bin/false', '--stage', 'diarization'])
    assert ledger.read_text() == original
