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
                source='windows-dxcore', gpu=dict(free_bytes=100, used_bytes=200))
            result = module.gpu(config)
            assert result['observed_at'] == 123
            assert result['free_bytes'] == 100
            assert factory.call_args.kwargs['expected_source'] == 'windows-dxcore'
            factory.return_value.sample.side_effect = RuntimeError('stale')
            with pytest.raises(RuntimeError, match='stale'):
                module.gpu(config)
        token.chmod(0o644)
        with pytest.raises(ValueError, match='private'):
            module.gpu(config)
