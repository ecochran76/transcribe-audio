"""Local check failures must propagate through the same hook entry point."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_local.py'


class LocalCheckTests(unittest.TestCase):
    def test_failures_and_hook_preservation(self):
        spec = importlib.util.spec_from_file_location('local_checks', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertNotEqual(module.run_checks([[sys.executable, '-c', 'raise SystemExit(7)']], Path.cwd())['exit_code'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True, env={k: v for k, v in os.environ.items() if not k.startswith('GIT_')})
            hook = module.install_hook(root)
            self.assertIn('check_local.py', hook.read_text())
            (root / 'scripts').mkdir()
            (root / 'scripts/check_local.py').write_text('raise SystemExit(7)\n')
            self.assertEqual(subprocess.run([str(hook)], cwd=root, env=dict(__import__('os').environ, LOCAL_CHECK_PYTHON=sys.executable)).returncode, 7)
            (root / 'local-check-baseline.json').write_text(json.dumps({'findings': []}))
            (root / 'bad.py').write_text('print(undefined_name)\n')
            self.assertEqual(module.lint(root), 1)
            hook.write_text('existing\n')
            with self.assertRaises(FileExistsError):
                module.install_hook(root)
            self.assertEqual(hook.read_text(), 'existing\n')
