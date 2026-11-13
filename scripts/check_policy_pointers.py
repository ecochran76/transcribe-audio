"""Validate repository-local policy references in the agent routing surface."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check(root: Path) -> dict:
    root = root.resolve()
    text = (root / 'AGENTS.md').read_text()
    refs = sorted(set(re.findall(r'docs/dev/policies/[\w./-]+\.md', text)))
    if not refs:
        raise ValueError('AGENTS.md contains no local policy references')
    missing = []
    for ref in refs:
        target = (root / ref).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            missing.append(ref)
    return {'checked': len(refs), 'missing': missing, 'passed': not missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = check(args.repo)
    except (OSError, ValueError) as exc:
        print(json.dumps({'passed': False, 'error': str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
