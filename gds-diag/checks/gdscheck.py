# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
gdscheck binary discovery, invocation, and output parsing.

These utilities are shared by any module that needs to run `gdscheck -p`
and extract a named section from its output.
"""
from __future__ import annotations

import glob as _glob
import os
import shutil
import subprocess
from typing import Optional


def _find_gdscheck() -> Optional[str]:
    def cuda_sort_key(candidate: str) -> tuple[int, int]:
        import re
        match = re.search(r"/usr/local/cuda-(\d+)(?:\.(\d+))?/gds/tools/gdscheck$", candidate)
        if not match:
            return (-1, -1)
        major, minor = match.groups(default="0")
        return int(major), int(minor)

    candidates = [
        "/usr/local/cuda/gds/tools/gdscheck",
        "gdscheck",
        *sorted(
            _glob.glob("/usr/local/cuda*/gds/tools/gdscheck"),
            key=cuda_sort_key,
            reverse=True,
        ),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        path = shutil.which(candidate) if os.path.basename(candidate) == candidate else candidate
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _run_gdscheck_raw(
    path: str,
    apply_env: bool = True,
    required_section: str = "CUFILE CONFIGURATION",
) -> tuple[Optional[str], Optional[str]]:
    env = None
    if not apply_env:
        env = {key: value for key, value in os.environ.items() if not key.startswith("CUFILE_")}
    try:
        result = subprocess.run([path, "-p"], capture_output=True, text=True, timeout=30, env=env)
    except FileNotFoundError:
        return None, f"gdscheck not found at {path}"
    except subprocess.TimeoutExpired:
        return None, "gdscheck -p timed out"
    except OSError as exc:
        return None, str(exc)

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        detail = output.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        return None, f"gdscheck -p exited with {result.returncode}{suffix}"
    if required_section.upper() not in output.upper():
        return None, f"gdscheck -p did not include a {required_section} section"
    return output, None


def _gdscheck_section(output: str, section_header: str) -> list[str]:
    """Return content lines belonging to a named section of gdscheck -p output."""
    result: list[str] = []
    in_section = False
    skip_sep = False

    for line in output.splitlines():
        stripped = line.strip()
        is_sep = bool(stripped) and all(c == "=" for c in stripped)

        if not in_section:
            if section_header.upper() in line.upper() and stripped.endswith(":") and not is_sep:
                in_section = True
                skip_sep = True
            continue

        if skip_sep:
            if is_sep:
                skip_sep = False
            continue

        if is_sep:
            break

        result.append(line)

    return result


# ---------------------------------------------------------------------------
# DRIVER CONFIGURATION token helpers
# ---------------------------------------------------------------------------

DIRECT_ROUTE_TOKENS = frozenset({"p2pdma", "c2c"})
NATIVE_ROUTE_TOKENS = frozenset({"nvfs", "dmabuf", "nvidia_peermem"})


def driver_config_tokens(modes: str) -> list[str]:
    return [token.strip().lower() for token in modes.split(",")]


def driver_config_status_supported(modes: str) -> bool:
    return modes.strip().lower() == "supported"


def driver_config_status_unsupported(modes: str) -> bool:
    return modes.strip().lower() in {"unsupported", "not supported"}


def driver_config_has_native(modes: str) -> bool:
    if driver_config_status_supported(modes):
        return True
    if driver_config_status_unsupported(modes):
        return False
    return any(token in NATIVE_ROUTE_TOKENS for token in driver_config_tokens(modes))


def driver_config_has_direct_p2pdma_token(modes: str) -> bool:
    return any(token in DIRECT_ROUTE_TOKENS for token in driver_config_tokens(modes))


def driver_config_has_compat(modes: str) -> bool:
    return "compat" in driver_config_tokens(modes)
