# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gds-diag version metadata."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


TOOL_NAME = "gds-diag"
__version__ = "1.0.0"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    """Return the current Git commit when this checkout has Git metadata."""
    try:
        git_path = shutil.which("git")
        if git_path is None:
            raise FileNotFoundError("git executable not found")
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        result = subprocess.run(
            [git_path, "rev-parse", "--short=12", "HEAD"],
            cwd=str(_repo_root()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return None
    return commit


def tool_metadata() -> dict[str, str | None]:
    return {
        "name": TOOL_NAME,
        "version": __version__,
        "git_commit": git_commit(),
    }


def version_string() -> str:
    commit = git_commit()
    if commit:
        return f"{TOOL_NAME} {__version__} (git {commit})"
    return f"{TOOL_NAME} {__version__} (git unknown)"
