"""Local qualification presubmit. Full checks are explicit; no hosted CI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time

RUFF_VERSION = '0.17.3'
ROOT = Path(__file__).resolve().parents[1]


def ruff_command():
    candidate = shutil.which('ruff')
    if candidate:
        version = subprocess.check_output([candidate, '--version'], text=True).strip()
        if version != 'ruff ' + RUFF_VERSION:
            raise RuntimeError('Use Ruff ' + RUFF_VERSION + ' or cached uvx --offline ruff==' + RUFF_VERSION)
        return [candidate]
    uvx = shutil.which('uvx')
    if not uvx:
        cached = Path('/home/linuxbrew/.linuxbrew/bin/uvx')
        uvx = str(cached) if cached.is_file() else None
    if uvx:
        return [uvx, '--offline', 'ruff==' + RUFF_VERSION]
    raise RuntimeError('Ruff unavailable: provide pinned Ruff ' + RUFF_VERSION + ' or cached uvx on PATH')


def lint(root):
    command = ruff_command() + ['check', '--select', 'F', '--output-format', 'json', '.']
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError('Ruff failed: ' + result.stderr[:500])
    found = json.loads(result.stdout)
    baseline = json.loads((root / 'local-check-baseline.json').read_text())
    allowed = {tuple(x) for x in baseline['findings']}
    current = {(str(Path(x['filename']).relative_to(root)), x['code'], x['message'],
                x['location']['row'], x['location']['column']) for x in found}
    new = sorted(current - allowed)
    print(json.dumps({'lint_new': new, 'accepted_legacy_findings': len(current & allowed),
                      'unused_baseline_entries': len(allowed - current)}))
    return 1 if new else 0


def run_checks(commands, cwd):
    rows = []
    for command in commands:
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=cwd, timeout=900)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        rows.append({'command': command, 'returncode': code,
                     'seconds': round(time.monotonic() - started, 3)})
        if code:
            break
    return {'exit_code': rows[-1]['returncode'] if rows else 0, 'checks': rows}


def install_hook(root):
    """Opt in without replacing another hook or altering hooksPath."""
    root = Path(root)
    raw = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--git-path', 'hooks/pre-commit'], text=True).strip()
    hook = Path(raw)
    if not hook.is_absolute():
        hook = root / hook
    hook.parent.mkdir(parents=True, exist_ok=True)
    with hook.open('x') as stream:
        stream.write('#!/bin/sh\nset -eu\nroot=$(git rev-parse --show-toplevel)\n'
                     'if [ -f "$root/qualification/scripts/check_local.py" ]; then base="$root/qualification"; else base="$root"; fi\n'
                     'python=' + shlex.quote(sys.executable) + '\n'
                     'exec "${LOCAL_CHECK_PYTHON:-$python}" "$base/scripts/check_local.py" --fast\n')
    hook.chmod(0o755)
    return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fast', action='store_true')
    mode.add_argument('--full', action='store_true')
    parser.add_argument('--install-hook', action='store_true')
    parser.add_argument('--lint-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Python >=3.12 required; select the existing environment with LOCAL_CHECK_PYTHON')
    try:
        if args.install_hook:
            print('Installed ' + str(install_hook(ROOT)))
            return 0
        if args.lint_only:
            return lint(ROOT)
        commands = [[sys.executable, str(Path(__file__).resolve()), '--lint-only']]
        if (ROOT / 'inference_manager').is_dir():
            selection = ['tests.test_fixture_preflight', 'tests.test_fixture_expectations',
                         'tests.test_local_checks', 'tests.test_isolated_dxcore_case']
            for name in ('test_protected_admission', 'test_fixture_execution_lock', 'test_receipt_summary', 'test_execution_projection'):
                if (ROOT / 'tests' / (name + '.py')).is_file():
                    selection.append('tests.' + name)
            commands.append([sys.executable, '-m', 'unittest'] + (['discover', '-s', 'tests', '-t', '.', '-q'] if args.full else selection + ['-q']))
        else:
            commands.append([sys.executable, str(ROOT / 'scripts/check_policy_pointers.py')])
            commands.append([sys.executable, '-m', 'pytest', '-q'] + ([] if args.full else ['tests/test_policy_pointers.py', 'tests/test_local_checks.py']))
        result = run_checks(commands, ROOT)
        result.update(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      python=sys.version, tier='full' if args.full else 'fast',
                      hosted_environment_difference='Local Linux/Python; optional live/protocol skips remain skips; no hosted Actions')
        if args.receipt:
            with args.receipt.open('x') as stream:
                os.chmod(args.receipt, 0o600)
                json.dump(result, stream, indent=2)
        print(json.dumps(result, sort_keys=True))
        return result['exit_code']
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('Local checks failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
