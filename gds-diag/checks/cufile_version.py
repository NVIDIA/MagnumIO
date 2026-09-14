# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cuFile / libcufile version detection.

The cuFile API exposes cuFileGetVersion(), which returns:
    1000 * major + 10 * minor

That API does not include patch/build version, so this module also inspects the
selected libcufile.so filename and gdscheck release banner when available.
"""
from __future__ import annotations

import ctypes.util
import glob
import json
import os
import re
import subprocess
import sys
from typing import Optional


def parse_version_tuple(value: str | None) -> Optional[tuple[int, ...]]:
    if not value:
        return None
    parts = re.findall(r"\d+", value)
    if not parts:
        return None
    return tuple(int(part) for part in parts)


def version_to_string(version: tuple[int, ...] | None) -> Optional[str]:
    if not version:
        return None
    return ".".join(str(part) for part in version)


def version_at_least(version: tuple[int, ...] | None, minimum: tuple[int, ...] | None) -> bool:
    if not minimum:
        return True
    if not version:
        return False
    width = max(len(version), len(minimum))
    padded_version = version + (0,) * (width - len(version))
    padded_minimum = minimum + (0,) * (width - len(minimum))
    return padded_version >= padded_minimum


def decode_cufile_version_int(value: int) -> tuple[int, int]:
    """Decode cuFileGetVersion() integer to (major, minor)."""
    return value // 1000, (value % 1000) // 10


def parse_gds_release_version(output: str | None) -> Optional[str]:
    if not output:
        return None
    match = re.search(r"^\s*GDS release version:\s*([^\s]+)", output, re.MULTILINE)
    return match.group(1) if match else None


def _version_from_lib_path(path: str | None) -> Optional[str]:
    if not path:
        return None
    real = os.path.realpath(path)
    match = re.search(r"libcufile\.so\.(\d+(?:\.\d+){1,3})$", real)
    return match.group(1) if match else None


def _candidate_paths() -> list[str]:
    candidates: list[str] = []

    for root in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if root:
            candidates.extend(glob.glob(os.path.join(root, "libcufile.so*")))

    preferred = [
        "/usr/local/cuda/lib64/libcufile.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
        "/usr/local/cuda/targets/aarch64-linux/lib/libcufile.so",
    ]
    candidates.extend(preferred)

    found = ctypes.util.find_library("cufile")
    if found:
        candidates.append(found)

    candidates.extend(sorted(glob.glob("/usr/local/cuda*/lib64/libcufile.so")))
    candidates.extend(sorted(glob.glob("/usr/local/cuda*/targets/*/lib/libcufile.so")))
    candidates.extend(sorted(glob.glob("/usr/local/cuda*/lib64/libcufile.so.[0-9]*")))
    candidates.extend(sorted(glob.glob("/usr/local/cuda*/targets/*/lib/libcufile.so.[0-9]*")))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        # Keep non-absolute find_library results, but de-dupe exact strings.
        key = os.path.realpath(candidate) if os.path.isabs(candidate) else candidate
        if key in seen:
            continue
        seen.add(key)
        if os.path.exists(candidate) or not os.path.isabs(candidate):
            deduped.append(candidate)
    return deduped


def _probe_script_path() -> str:
    """Return the checkout-local probe path; no package installation is needed."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cufile_version_probe.py")


def _query_cufile_get_version(path: str) -> dict:
    """
    Query cuFileGetVersion in a subprocess, distinguishing "this library
    could not even be loaded" (genuinely missing/incompatible) from "it
    loaded fine but cuFileGetVersion isn't there" (present, but an older
    GDS release that predates that symbol -- e.g. GDS 1.7.x, which shipped
    with CUDA 12.2).

    Doing this in a subprocess (rather than in-process) avoids getting back
    whichever libcufile the dynamic linker already loaded when several CUDA
    versions with the same SONAME are installed.
    """
    try:
        result = subprocess.run(
            [sys.executable, _probe_script_path(), path],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as exc:
        return {"loaded": False, "version_int": None, "error": str(exc)}

    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        error = result.stderr.strip() or result.stdout.strip() or "invalid version probe output"
        return {"loaded": False, "version_int": None, "error": error}

    # Reaching the symbol lookup or the call itself means ctypes.CDLL()
    # already succeeded, so the library file is real and loadable -- only
    # the specific cuFileGetVersion() symbol/call is what's unavailable.
    loaded = payload.get("stage") in ("symbol", "call")

    if result.returncode != 0 or "error" in payload:
        error = payload.get("error") or result.stderr.strip() or "cuFileGetVersion probe failed"
        return {"loaded": loaded, "version_int": None, "error": error}
    if payload.get("rc") != 0:
        return {"loaded": True, "version_int": None, "error": f"cuFileGetVersion returned rc={payload.get('rc')}"}
    try:
        return {"loaded": True, "version_int": int(payload["version"]), "error": None}
    except (KeyError, TypeError, ValueError):
        return {"loaded": True, "version_int": None, "error": "cuFileGetVersion returned no version"}


def detect_libcufile(gdscheck_output: str | None = None) -> dict:
    """
    Detect the preferred libcufile library and version.

    "found" means the library itself was located and successfully loaded --
    it does NOT require a successful cuFileGetVersion() call. Older GDS
    releases (e.g. 1.7.x, shipped with CUDA 12.2) don't export that symbol
    at all, but the library is still genuinely present and in use; treating
    that as "not found" would be wrong. "found" with api_version/
    api_version_tuple/version_source all None means exactly that case:
    library present, API version specifically unknown. file_version (parsed
    from the .so filename) and gds_release_version (parsed from gdscheck's
    banner) are populated whenever available, independent of whether the
    cuFileGetVersion() probe itself succeeded.

    Returns a JSON-friendly dictionary with API version, file/symlink version,
    gdscheck release version if supplied, and any probe error.
    """
    gds_release = parse_gds_release_version(gdscheck_output)
    probe_errors: list[str] = []
    loaded_path: Optional[str] = None  # first candidate that loaded, even if the version probe on it failed
    loaded_path_error: Optional[str] = None  # that candidate's own probe error, kept separate so it isn't buried behind earlier candidates' unrelated failures

    for path in _candidate_paths():
        probe = _query_cufile_get_version(path)
        error_detail = f"{path}: {probe['error']}" if probe.get("error") else None
        if error_detail:
            probe_errors.append(error_detail)

        version_int = probe.get("version_int")
        if version_int is not None:
            api_version = decode_cufile_version_int(version_int)
            return {
                "found": True,
                "path": path,
                "realpath": os.path.realpath(path) if os.path.isabs(path) else path,
                "api_version_int": version_int,
                "api_version": version_to_string(api_version),
                "api_version_tuple": list(api_version) if api_version else None,
                "file_version": _version_from_lib_path(path),
                "gds_release_version": gds_release,
                "version_source": "cuFileGetVersion",
                "probe_errors": probe_errors,
            }

        if probe.get("loaded") and loaded_path is None:
            loaded_path = path
            loaded_path_error = error_detail

    if loaded_path:
        # Callers (e.g. support_matrix._libcufile_summary) read probe_errors[0]
        # to explain why api_version is unknown for the found library -- that
        # must be loaded_path's own failure, not an earlier, unrelated
        # candidate's (e.g. one that didn't even exist on disk).
        ordered_errors = probe_errors
        if loaded_path_error and probe_errors and probe_errors[0] != loaded_path_error:
            ordered_errors = [loaded_path_error] + [e for e in probe_errors if e != loaded_path_error]
        return {
            "found": True,
            "path": loaded_path,
            "realpath": os.path.realpath(loaded_path) if os.path.isabs(loaded_path) else loaded_path,
            "api_version_int": None,
            "api_version": None,
            "api_version_tuple": None,
            "file_version": _version_from_lib_path(loaded_path),
            "gds_release_version": gds_release,
            "version_source": None,
            "probe_errors": ordered_errors,
        }

    return {
        "found": False,
        "path": None,
        "realpath": None,
        "api_version_int": None,
        "api_version": None,
        "api_version_tuple": None,
        "file_version": None,
        "gds_release_version": gds_release,
        "version_source": None,
        "probe_errors": probe_errors,
    }
