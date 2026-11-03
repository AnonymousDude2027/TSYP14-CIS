"""The CLI is the reproducibility surface, so it gets its own checks.

These are deliberately cheap: they exercise argument parsing and the two
commands that need no model cache (``sites`` and ``data-verify``).  The heavy
commands (``gonogo``, ``experiments``, ``report-figures``) are covered by the
Makefile pipeline and by ``tsfdi.verify``.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args, timeout=180):
    return subprocess.run([sys.executable, "-m", "tsfdi", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout,
                          env={"PYTHONPATH": "src", "TSFDI_ROOT": str(ROOT),
                               "PATH": "/usr/local/bin:/usr/bin:/bin"})


def test_help_lists_every_documented_command():
    out = run("--help")
    assert out.returncode == 0, out.stderr
    for cmd in ("fetch", "data-verify", "sites", "gonogo", "calibrate", "stream",
                "experiments", "report-figures", "verify", "bench", "selftest"):
        assert cmd in out.stdout, cmd


def test_data_verify_accepts_the_signed_bundle():
    out = run("data-verify")
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout.upper() or "ok" in out.stdout


def test_sites_command_reports_every_configured_site():
    out = run("sites", timeout=300)
    assert out.returncode == 0, out.stderr
    from tsfdi.config import sites

    for s in sites():
        assert s.name in out.stdout


def test_unknown_command_fails_loudly():
    out = run("no-such-command")
    assert out.returncode != 0
