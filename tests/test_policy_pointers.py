"""Policy routing must fail visibly when a local authority is missing."""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_policy_pointers.py'


def test_missing_policy_is_reported_and_corrected_pointer_passes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy = root / 'docs/dev/policies/current.md'
        policy.parent.mkdir(parents=True)
        (root / 'AGENTS.md').write_text('Read `docs/dev/policies/current.md`.\n')
        command = [sys.executable, str(SCRIPT), '--repo', str(root)]
        red = subprocess.run(command, capture_output=True, text=True)
        assert red.returncode == 1
        assert 'docs/dev/policies/current.md' in red.stdout
        policy.write_text('Current authority\n')
        assert subprocess.run(command, capture_output=True).returncode == 0
