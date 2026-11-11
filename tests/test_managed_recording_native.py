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
    for attempts, seconds, cleanup in [(7, 160, 15), (6, 161, 15), (6, 160, 16), (6, 0, 15)]:
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
