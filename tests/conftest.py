"""Make the modules-under-test importable.

`vga` is a package (import from repo root), but coverage_report.py lives in
train/rl as a standalone script (train/ is not a package), so add that dir too.
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
for p in (_REPO, os.path.join(_REPO, "train", "rl")):
    if p not in sys.path:
        sys.path.insert(0, p)
