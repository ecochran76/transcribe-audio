#!/usr/bin/env python3
"""Operator entry point for the bounded storage controller."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage_maintenance import main

if __name__ == '__main__':
    raise SystemExit(main())
