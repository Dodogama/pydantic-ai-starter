"""Repo I/O: running commands, snapshots and diffs."""

import difflib
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def sh(root: Path, cmd: str, timeout: float) -> tuple[bool, str]:
    # ponytail: model-authored acceptance commands run via the shell; run in a container/VM for untrusted repos.
    cmd = cmd.replace("{python}", shlex.quote(sys.executable))
    # Fresh bytecode cache per command: .pyc staleness is checked by mtime (1s) + size, so a same-size
    # rewrite within a second would otherwise run the old code.
    with tempfile.TemporaryDirectory() as pycache:
        env = {**os.environ, "PYTHONPYCACHEPREFIX": pycache}
        try:
            p = subprocess.run(cmd, shell=True, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout}s"
    return p.returncode == 0, (p.stdout + p.stderr)[-4000:]


def repo_files(root: Path):
    skip = {"__pycache__", ".git", ".venv"}
    return (p for p in root.rglob("*") if p.is_file() and not skip & set(p.relative_to(root).parts))


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(errors="replace") for p in repo_files(root)}


def diff_since(before: dict[str, str], root: Path) -> str:
    after = snapshot(root)
    return "".join(
        "".join(difflib.unified_diff(
            before.get(f, "").splitlines(True), after.get(f, "").splitlines(True), f"a/{f}", f"b/{f}"
        ))
        for f in sorted(before.keys() | after.keys())
    )
