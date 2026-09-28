# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kernel log access for GDS checks.

`dmesg` reads the in-memory kernel ring buffer, which is small and can wrap
during long-running boots — relevant messages from early boot may be gone
by the time we look. `journalctl -k -b` reads the systemd journal's
kernel-message stream for the current boot, which is persistent and not
size-bounded the same way.

This module exposes a single source of kernel log lines that uses
noninteractive sudo for `journalctl -k -b` and `dmesg`. Results from the
underlying command are cached per process so multiple callers (e.g. iommu
and nvidia_fs) share one fetch.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import os
import re
import subprocess


@dataclass(frozen=True)
class KernelLogResult:
    lines: tuple[str, ...]
    source: str | None = None
    error: str | None = None
    permission_denied: bool = False

    @property
    def available(self) -> bool:
        return self.source is not None


def _looks_like_permission_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "a password is required",
            "password is required",
            "a terminal is required",
            "not in the sudoers",
            "not allowed to execute",
            "permission denied",
            "operation not permitted",
            "no new privileges",
            "prevents sudo from running as root",
        )
    )


def privileged_validation_mitigation(manual_commands: tuple[str, ...]) -> str:
    """Return consistent guidance for checks skipped by noninteractive sudo failure."""
    lines = [
        (
            "Run this gds-diag.py command with sudo to enable the extra "
            "privileged validation checks, or inspect manually:"
        ),
    ]
    lines.extend(f"  {command}" for command in manual_commands)
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def read_kmsg_result() -> KernelLogResult:
    """Return kernel-log lines plus access status from the current boot.

    Preference order:
      1. sudo -n journalctl -k -b -o cat --no-pager
      2. sudo -n dmesg

    read_kmsg_result returns an unavailable result only when neither command can
    run successfully, for example because access is denied or execution fails. A
    successful journalctl or dmesg command that returns no kernel-log lines is
    still available and is represented as an empty lines tuple. This lets
    callers distinguish no output (available, empty lines) from no access
    (unavailable). The result is cached for the lifetime of the process, so
    callers can grep it repeatedly without re-shelling.
    """
    attempts = (
        (
            "sudo -n journalctl -k -b",
            ["sudo", "-n", "journalctl", "-k", "-b", "-o", "cat", "--no-pager"],
            15,
            True,
        ),
        ("sudo -n dmesg", ["sudo", "-n", "dmesg"], 10, False),
    )
    failures: list[str] = []
    permission_denied = False
    empty_success: str | None = None
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    for label, cmd, timeout, require_output in attempts:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError) as exc:
            detail = f"{label}: {exc}"
            failures.append(detail)
            permission_denied = permission_denied or isinstance(exc, PermissionError)
            continue

        if result.returncode == 0:
            if result.stdout.strip() or not require_output:
                return KernelLogResult(
                    tuple(result.stdout.splitlines()),
                    source=label,
                    error="; ".join(failures) if failures else None,
                    permission_denied=permission_denied,
                )
            empty_success = empty_success or label
            continue

        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        failures.append(f"{label}: {detail}")
        permission_denied = permission_denied or _looks_like_permission_failure(detail)

    if empty_success:
        return KernelLogResult((), source=empty_success)

    return KernelLogResult(
        (),
        error="; ".join(failures) if failures else "no kernel log command succeeded",
        permission_denied=permission_denied,
    )


def read_kmsg() -> tuple[str, ...]:
    """Return all readable kernel-log lines from the current boot."""
    return read_kmsg_result().lines


def grep_kmsg(pattern: str, *, flags: int = re.IGNORECASE) -> list[str]:
    """Return kernel-log lines matching `pattern` (regex, case-insensitive by default)."""
    rx = re.compile(pattern, flags)
    return [line for line in read_kmsg() if rx.search(line)]


def grep_kmsg_result(pattern: str, *, flags: int = re.IGNORECASE) -> KernelLogResult:
    """Return matching kernel-log lines while preserving access status."""
    result = read_kmsg_result()
    rx = re.compile(pattern, flags)
    return KernelLogResult(
        tuple(line for line in result.lines if rx.search(line)),
        source=result.source,
        error=result.error,
        permission_denied=result.permission_denied,
    )
