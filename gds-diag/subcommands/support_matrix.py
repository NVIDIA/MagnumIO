# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
support-matrix subcommand.

Reference table of filesystem GDS support. Default (no flags): auto-detects
gdscheck and shows live data if available, otherwise falls back to the static
reference table. Use --static to force static output or --live to require
gdscheck.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import shutil
import subprocess
from typing import Optional

from ._base import Subcommand
from checks import cufile_version
from checks.iommu import Arch, arch as _detect_arch
from checks.gdscheck import _find_gdscheck, _gdscheck_section, _run_gdscheck_raw
from checks.gdscheck import (
    driver_config_has_compat,
    driver_config_has_direct_p2pdma_token,
    driver_config_has_native,
    driver_config_status_supported,
    driver_config_status_unsupported,
    driver_config_tokens,
)

_DESCRIPTION = """\
Print the GDS filesystem support matrix.

--static (default): reference capability table — Native, P2PDMA, and Compat
  support per filesystem type. Sourced from NVIDIA GDS documentation.
  No system access required.

--live: uses `gdscheck -p` when available and parses DRIVER CONFIGURATION
  output. If gdscheck is unavailable but CUDA Toolkit and libcufile are present,
  falls back to the documentation matrix with libcufile version gating and
  explains that live driver/client mode tokens are unavailable.
"""

CUDA_DOWNLOADS_URL = "https://developer.nvidia.com/cuda-downloads"

# Display order and grouping for the table rows.
# Keep FS_CAPABILITIES keys stable for JSON/API consumers, but render labels
# that describe the storage route users actually think about.
#
# Grouped by whether the entry is a real filesystem namespace (a mounted,
# POSIX-ish directory tree) vs. a device/transport/block path that something
# else (ext4, xfs, ...) mounts on top of — not by physical medium or
# deployment context. nvme-of/raid0 are transports, not namespaces, so they
# sit in "Storage paths" alongside nvmesh/scsi/scaleflux rather than next to
# ext4/xfs.
_FS_GROUPS: list[tuple[str, list[str]]] = [
    ("Local file systems",  ["ext4", "xfs"]),
    ("Remote file systems", ["lustre", "gpfs", "wekafs", "beegfs", "nfs", "virtiofs", "scatefs"]),
    ("Storage paths",       ["nvme-of", "raid0", "nvmesh", "scsi", "scaleflux"]),
    ("Compat-only",         ["squashfs", "tmpfs", "ramfs", "overlayfs", "zfs", "btrfs"]),
]

_FS_DISPLAY_NAMES: dict[str, str] = {
    "ext4": "ext4 on NVMe",
    "xfs": "xfs on NVMe",
    "nvme-of": "NVMe-oF",
    "raid0": "RAID0 over NVMe",
    "lustre": "lustre / DDN EXAScaler",
    "nfs": "nfs / nfs4",
}

_FS_DISPLAY_ALIASES: dict[str, list[str]] = {
    "lustre": ["ddn exascaler"],
}

# Maps lowercase gdscheck DRIVER CONFIGURATION key → our fs_type(s)
_DRIVER_KEY_TO_FS: dict[str, list[str]] = {
    "nvme":               ["ext4", "xfs"],
    "nvmeof":             ["nvme-of"],
    "scsi":               ["scsi"],
    "scaleflux csd":      ["scaleflux"],
    "nvmesh":             ["nvmesh"],
    # DDN EXAScaler is Lustre-based; present it as the same support row.
    "ddn exascaler":      ["lustre"],
    "nfs":                ["nfs"],
    "lustre":             ["lustre"],
    "beegfs":             ["beegfs"],
    "scatefs":            ["scatefs"],
    "wekafs":             ["wekafs"],
    "ibm spectrum scale": ["gpfs"],
    "virtiofs":           ["virtiofs"],
}



def _detect_cuda_toolkit() -> dict:
    """
    Lightweight CUDA Toolkit detection for installer guidance.

    Prefer nvcc, then fall back to common /usr/local/cuda* toolkit layouts.
    This is not a full pre-install check; it only decides which remediation
    message to show when support-matrix --live cannot find gdscheck.
    """
    candidates: list[str] = []
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(nvcc)
    candidates.extend(sorted(_glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True))
    for path in candidates:
        if os.path.isfile(path):
            return {"found": True, "evidence": path}

    cuda_dirs = sorted(_glob.glob("/usr/local/cuda*"), reverse=True)
    for path in cuda_dirs:
        if os.path.isdir(path):
            return {"found": True, "evidence": path}

    return {"found": False, "evidence": None}


def _detect_gds_packages() -> list[str]:
    """Return installed packages whose names mention GDS, if rpm/dpkg are available."""
    packages: list[str] = []
    commands = (
        ["rpm", "-qa"],
        ["dpkg-query", "-W", "-f=${binary:Package} ${Version}\\n"],
    )
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except Exception:
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if "gds" in line.lower() and line not in packages:
                packages.append(line)
    return packages


def _gdscheck_missing_payload() -> dict:
    cuda = _detect_cuda_toolkit()
    gds_packages = _detect_gds_packages()
    if not cuda["found"]:
        summary = (
            "CUDA Toolkit was not detected. Install the CUDA Toolkit first, "
            "then install the matching gds-tools package."
        )
        next_action = "install_cuda_toolkit_then_gds_tools"
    elif not gds_packages:
        summary = (
            "CUDA Toolkit was detected, but no installed GDS tools package was found. "
            "Install the matching gds-tools package."
        )
        next_action = "install_gds_tools"
    else:
        summary = (
            "GDS package(s) appear installed, but gdscheck was not found in the expected "
            "CUDA GDS tool paths."
        )
        next_action = "repair_or_locate_gdscheck"
    return {
        "error": "gdscheck_not_found",
        "summary": summary,
        "next_action": next_action,
        "cuda_toolkit_found": cuda["found"],
        "cuda_toolkit_evidence": cuda["evidence"],
        "gds_packages": gds_packages,
        "install_cuda_url": CUDA_DOWNLOADS_URL,
        "expected_gdscheck_paths": [
            "/usr/local/cuda/gds/tools/gdscheck",
            "/usr/local/cuda-*/gds/tools/gdscheck",
        ],
    }


def _print_gdscheck_missing_text(payload: dict) -> None:
    print("ERROR: --live requested but gdscheck was not found.")
    print("       support-matrix --live needs gdscheck from the GDS tools package.")
    print()
    if payload["cuda_toolkit_found"]:
        print(f"CUDA Toolkit check : found ({payload['cuda_toolkit_evidence']})")
    else:
        print("CUDA Toolkit check : not found")

    packages = payload.get("gds_packages") or []
    if packages:
        print("GDS package check  : found")
        for package in packages:
            print(f"  - {package}")
    else:
        print("GDS package check  : no gds-tools package detected")

    print()
    print("Recommended action:")
    if not payload["cuda_toolkit_found"]:
        print(f"  1. Install CUDA Toolkit first: {CUDA_DOWNLOADS_URL}")
        print("  2. Install the matching GDS tools package for that CUDA version.")
    elif packages:
        print("  1. Verify where the package installed gdscheck:")
        print("     ls /usr/local/cuda*/gds/tools/gdscheck")
        print("  2. If missing, reinstall or repair the matching gds-tools package.")
    else:
        print("  1. Install the matching GDS tools package for the installed CUDA Toolkit.")

    print("     RHEL/RPM example:   sudo dnf install gds-tools-13-3")
    print("     Ubuntu/Deb example: sudo apt-get install gds-tools-13-1")
    print("     Replace the CUDA version suffix so it matches the installed toolkit.")
    print("     Package checks:     rpm -qa | grep gds")
    print("                         dpkg -l | grep gds")
    print()
    print("For the documentation-only table, run without --live:")
    print("  python3 gds-diag.py support-matrix")


def _can_fallback_without_gdscheck(payload: dict, libcufile_info: Optional[dict]) -> bool:
    return bool(payload.get("cuda_toolkit_found") and libcufile_info and libcufile_info.get("found"))


def _fallback_note(payload: dict, reason: str) -> str:
    return (
        f"{reason}; falling back to the documentation matrix with detected "
        "libcufile/GDS version gating. Live driver/client mode tokens are "
        "unavailable until matching gds-tools provides gdscheck."
    )


def _parse_driver_config(output: str) -> dict[str, str]:
    """Parse DRIVER CONFIGURATION section → {driver_key_lower: modes_string}."""
    result: dict[str, str] = {}
    for line in _gdscheck_section(output, "DRIVER CONFIGURATION"):
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip().lower()] = val.strip()
    return result


def _merge_modes(current: Optional[str], new: Optional[str]) -> Optional[str]:
    """Merge mode strings from equivalent gdscheck driver rows."""
    if not current:
        return new
    if not new:
        return current
    tokens: list[str] = []
    for value in (current, new):
        for token in (part.strip() for part in value.split(",")):
            if token and token not in tokens:
                tokens.append(token)
    return ", ".join(tokens)


def _mode_tokens(modes: str) -> list[str]:
    return driver_config_tokens(modes)


def _has_direct_p2pdma_token(modes: str) -> bool:
    return driver_config_has_direct_p2pdma_token(modes)


def _old_status_format(modes: str) -> bool:
    return driver_config_status_supported(modes) or driver_config_status_unsupported(modes)


def _live_rdma_from_modes(modes: str, supports_userspace_rdma: bool) -> bool:
    if not supports_userspace_rdma:
        return False
    if _old_status_format(modes):
        return driver_config_status_supported(modes)
    return any(t in _mode_tokens(modes) for t in ("dmabuf", "nvidia_peermem"))


def _live_native_from_modes(modes: str, supports_userspace_rdma: bool) -> bool:
    if supports_userspace_rdma:
        return _live_rdma_from_modes(modes, supports_userspace_rdma)
    return driver_config_has_native(modes)


def _live_p2pdma_from_modes(fs: str, modes: str, driver_config: dict[str, str]) -> bool:
    if driver_config_has_direct_p2pdma_token(modes):
        return True
    if fs in {"ext4", "xfs"}:
        nvme_p2pdma = driver_config.get("nvme p2pdma")
        return bool(nvme_p2pdma and driver_config_status_supported(nvme_p2pdma))
    return False


def _live_compat_from_modes(caps: dict, modes: str, libcufile_tuple: Optional[tuple[int, ...]] = None):
    if _old_status_format(modes):
        return _compat_ref_value(caps, libcufile_tuple)
    return _compat_ref_value({**caps, "compat": driver_config_has_compat(modes)}, libcufile_tuple)


def _live_status_map(driver_config: dict[str, str]) -> dict[str, Optional[str]]:
    """Build {fs_type: modes_string | None}. None = client not loaded."""
    live: dict[str, Optional[str]] = {}
    for driver_key, fs_types in _DRIVER_KEY_TO_FS.items():
        modes = driver_config.get(driver_key)
        for fs in fs_types:
            live[fs] = _merge_modes(live.get(fs), modes) if fs in live else modes
    return live


RDMA_COL = 16
P2PDMA_COL = 14
COMPAT_COL = 12


def _fmt(v, col: int = 9) -> str:
    """Format a capability value into a padded coloured cell."""
    from checks.output import green, red, yellow, dim
    if v is True:
        return green(f"{'✓ Yes':<{col}}")
    if v is False:
        return red(f"{'✗ No':<{col}}")
    if v == "config":
        return yellow(f"{'Config':<{col}}")
    if v == "nomp-config":
        return yellow(f"{'NoMP Config':<{col}}")
    if isinstance(v, str) and v.startswith("needs-"):
        return yellow(f"{('Need ' + v.split('-', 1)[1]):<{col}}")
    if v is None:
        return dim(f"{'?':<{col}}")
    if isinstance(v, str):
        # Free-form display label (e.g. a specific RDMA mechanism name) —
        # not one of the fixed tokens above, render it directly.
        return yellow(f"{v:<{col}}")
    return dim(f"{'–':<{col}}")


_RDMA_TYPE_DEFAULT_DISPLAY: dict[str, str] = {
    "userspace": "dmabuf/peermem",
    "kernel": "FS/kernel RDMA",
}


def _rdma_ref_value(
    caps: dict,
    live_rdma: Optional[bool] = None,
    live_native: Optional[bool] = None,
):
    """
    Render the RDMA column with the category of mechanism this filesystem's
    GDS route uses, rather than a plain Yes/No.

    A boolean would be misleading: e.g. Lustre uses RDMA just as much as
    GPFS does (dmabuf/peermem), but they're different, non-interchangeable
    mechanisms, and Lustre's is folded into its Native GDS path rather than
    being a standalone route (see the Native column) — collapsing both to
    the same "Yes" would erase the difference, and showing Lustre as "No"
    would wrongly imply it has no RDMA involvement at all.

    Live gating differs by mechanism, because gdscheck's DRIVER CONFIGURATION
    token observes them differently:
      - Userspace cuFile RDMA (gpfs, wekafs) has its own distinct token
        (dmabuf/nvidia_peermem), so it's gated directly on live_rdma.
      - Kernel-level RDMA (Lustre, BeeGFS, NFS/NFSoRDMA) has no live token
        of its own — but it's meaningless without nvidia-fs active (it's a
        prerequisite of Native, not a standalone route), so it's gated on
        this filesystem's own live Native verdict instead: if Native shows
        inactive, kernel RDMA isn't actually delivering anything right now
        either, even though the architectural capability still exists (and
        --static, which has no live verdict to gate on, still shows it).
    """
    rdma_type = caps.get("rdma_type")
    if not rdma_type:
        return False

    display = caps.get("rdma_display") or _RDMA_TYPE_DEFAULT_DISPLAY.get(rdma_type, rdma_type)

    if rdma_type == "userspace" and live_rdma is not None:
        return display if live_rdma else False

    if rdma_type == "kernel" and live_native is not None:
        return display if live_native else False

    return display


def _p2pdma_ref_value(
    fs: str,
    caps: dict,
    live_p2pdma: Optional[bool] = None,
    host_arch: Optional[str] = None,
):
    """
    Render documented P2PDMA availability with route-specific constraints.

    A live p2pdma token is still shown as supported. Otherwise, local NVMe and
    RAID rows should retain their documented config-gated status instead of
    collapsing to plain "No".

    The extra caveat behind "NoMP Config"/"Kernel Config" (NVMe multipath,
    kernel version) is an x86-only concern — on ARM (e.g. NVIDIA Grace) it's
    automatically satisfied, so it collapses to the plain "Config" token
    instead of explaining a non-issue. host_arch=None (e.g. the JSON path,
    which never calls this) keeps today's unconditional, architecture-agnostic
    wording.
    """
    if caps.get("p2pdma", False) is False:
        return False
    if live_p2pdma is True:
        return True

    display = caps.get("p2pdma_display")
    if display == "NoMP Config":
        if host_arch == Arch.ARM:
            return "config"
        return "nomp-config"
    if display == "Arch Config":
        if host_arch == Arch.ARM:
            return "config"
        return "Kernel Config"
    if display == "Config":
        return "config"
    return caps.get("p2pdma", False)


def _libcufile_version_tuple(info: Optional[dict]) -> Optional[tuple[int, ...]]:
    if not info:
        return None
    api_tuple = info.get("api_version_tuple")
    if api_tuple:
        return tuple(int(part) for part in api_tuple)
    return cufile_version.parse_version_tuple(info.get("gds_release_version"))


def _compat_ref_value(caps: dict, libcufile_tuple: Optional[tuple[int, ...]] = None):
    compat = caps.get("compat", True)
    if compat is not True:
        return compat
    minimum = caps.get("compat_min_version")
    if minimum and libcufile_tuple and not cufile_version.version_at_least(libcufile_tuple, tuple(minimum)):
        return f"needs-{cufile_version.version_to_string(tuple(minimum))}"
    return True


def _libcufile_summary(info: Optional[dict]) -> str:
    if not info:
        return "not checked"
    if not info.get("found"):
        release = info.get("gds_release_version")
        summary = "not found" + (f" (gdscheck release {release})" if release else "")
        probe_errors = info.get("probe_errors")
        if probe_errors:
            summary += f" -- probe failed: {probe_errors[0]}"
        return summary
    pieces = []
    if info.get("api_version"):
        pieces.append(f"API {info['api_version']}")
    elif info.get("probe_errors"):
        # Library found and loaded, but cuFileGetVersion() itself didn't work
        # (e.g. an older GDS release that predates that symbol) -- distinct
        # from "not found" above, which means no loadable library at all.
        pieces.append(f"API version unavailable ({info['probe_errors'][0]})")
    if info.get("file_version"):
        pieces.append(f"file {info['file_version']}")
    if info.get("gds_release_version"):
        pieces.append(f"gdscheck {info['gds_release_version']}")
    if info.get("path"):
        pieces.append(info["path"])
    return ", ".join(pieces) if pieces else "detected"


def _run_text(args: argparse.Namespace) -> int:
    from checks.fs_matrix import FS_CAPABILITIES
    from checks.output import bold, dim, green, yellow
    from checks.version import version_string

    # Determine whether to show live column
    want_live = getattr(args, "live", False)
    want_static = getattr(args, "static", False)

    live_available = False
    driver_config: dict[str, str] = {}
    gdscheck_path: Optional[str] = None
    gdscheck_output: Optional[str] = None
    libcufile_info: Optional[dict] = None
    gdscheck_fallback_note: Optional[str] = None

    if want_live:
        gdscheck_path = _find_gdscheck()
        if not gdscheck_path:
            payload = _gdscheck_missing_payload()
            libcufile_info = cufile_version.detect_libcufile(None)
            if not _can_fallback_without_gdscheck(payload, libcufile_info):
                _print_gdscheck_missing_text(payload)
                return 3
            gdscheck_fallback_note = _fallback_note(payload, "gdscheck was not found")
        if gdscheck_path:
            raw, gdscheck_error = _run_gdscheck_raw(gdscheck_path, required_section="DRIVER CONFIGURATION")
            if not raw:
                payload = _gdscheck_missing_payload()
                libcufile_info = libcufile_info or cufile_version.detect_libcufile(None)
                reason = gdscheck_error or "gdscheck did not return DRIVER CONFIGURATION output"
                if not _can_fallback_without_gdscheck(payload, libcufile_info):
                    print(f"ERROR: {reason}")
                    return 3
                gdscheck_fallback_note = _fallback_note(
                    payload,
                    reason,
                )
            else:
                gdscheck_output = raw
                driver_config = _parse_driver_config(raw)
                live_available = True
    elif not want_static:
        # auto-detect: show live if gdscheck is available
        gdscheck_path = _find_gdscheck()
        if gdscheck_path:
            raw, _ = _run_gdscheck_raw(gdscheck_path, required_section="DRIVER CONFIGURATION")
            if raw:
                gdscheck_output = raw
                driver_config = _parse_driver_config(raw)
                live_available = True

    if not want_static:
        libcufile_info = cufile_version.detect_libcufile(gdscheck_output)
    libcufile_tuple = _libcufile_version_tuple(libcufile_info)

    NAME_COL = 24
    SEP_WIDTH = 114

    host_arch = _detect_arch()

    title = "GDS Filesystem Support Matrix"
    if host_arch in (Arch.X86, Arch.ARM):
        title += f" ({host_arch})"
    if live_available:
        title += f"  [live — {gdscheck_path}]"
    elif gdscheck_fallback_note:
        title += "  [fallback — gdscheck unavailable]"
    elif not want_static:
        title += "  [static — gdscheck not found]"
    print()
    print(bold(title))
    if getattr(args, "verbose", False):
        print(f"  Tool: {version_string()}")
    if libcufile_info:
        print(f"  libcufile: {_libcufile_summary(libcufile_info)}")
    if gdscheck_fallback_note:
        print(f"  note: {gdscheck_fallback_note}")
    print("─" * SEP_WIDTH)
    print(
        f"  {'Storage route':<{NAME_COL}} {'Native (nvidia-fs)':<20} "
        f"{'RDMA':<{RDMA_COL}} {'P2PDMA/C2C':<{P2PDMA_COL}} "
        f"{'Compat':<{COMPAT_COL}} {'Compat Since':<13}"
    )
    print("─" * SEP_WIDTH)

    live_map = _live_status_map(driver_config) if live_available else {}

    for group_name, fs_list in _FS_GROUPS:
        print(f"  {dim(group_name)}")
        for fs in fs_list:
            caps = FS_CAPABILITIES.get(fs)
            if caps is None:
                continue

            # dmabuf/peermem applies to filesystems using cuFile userspace RDMA (gpfs, wekafs)
            supports_userspace_rdma = caps.get("rdma_type") == "userspace"

            if live_available and fs in live_map:
                modes_str = live_map[fs]
                if modes_str is None:
                    native_s  = _fmt(None, col=20)
                    rdma_s    = _fmt(None, col=RDMA_COL)
                    p2pdma_s  = _fmt(None, col=P2PDMA_COL)
                    compat_s  = _fmt(None)
                    suffix = dim("  (client not loaded)")
                else:
                    # Gate on static matrix: gdscheck token only counts if architecturally supported
                    static_native = caps.get("native", False)
                    static_p2pdma = caps.get("p2pdma", False)
                    # For userspace RDMA filesystems (gpfs, wekafs), native GDS is via
                    # dmabuf/nvidia_peermem — not nvfs. For others (ext4, xfs, lustre,
                    # beegfs), native is via the nvfs kernel path.
                    live_rdma     = _live_rdma_from_modes(modes_str, supports_userspace_rdma)
                    live_p2pdma   = _live_p2pdma_from_modes(fs, modes_str, driver_config)
                    live_native   = _live_native_from_modes(modes_str, supports_userspace_rdma)
                    native_active = live_native and bool(static_native)
                    native_s  = _fmt(native_active, col=20)
                    rdma_s    = _fmt(_rdma_ref_value(caps, live_rdma, native_active), col=RDMA_COL)
                    p2pdma_s  = _fmt(_p2pdma_ref_value(fs, caps, live_p2pdma, host_arch), col=P2PDMA_COL)
                    compat_s  = _fmt(_live_compat_from_modes(caps, modes_str, libcufile_tuple), col=COMPAT_COL)
                    suffix = ""
            else:
                native_s = _fmt(caps.get("native", False), col=20)
                rdma_s   = _fmt(_rdma_ref_value(caps), col=RDMA_COL)
                p2pdma_s = _fmt(_p2pdma_ref_value(fs, caps, host_arch=host_arch), col=P2PDMA_COL)
                compat_s = _fmt(_compat_ref_value(caps, libcufile_tuple), col=COMPAT_COL)
                suffix = ""

            display_name = _FS_DISPLAY_NAMES.get(fs, fs)
            compat_since = caps.get("compat_since", "")
            print(
                f"  {display_name:<{NAME_COL}} {native_s} {rdma_s} {p2pdma_s} "
                f"{compat_s} {compat_since:<13}{suffix}"
            )
        print()

    print("─" * SEP_WIDTH)
    print()
    print(f"  {'✓ Yes':<12} supported")
    print(f"  {'✗ No':<12} not supported")
    print(f"  {'Config':<12} requires the matching cufile.json key and runtime confirmation")
    if host_arch != Arch.ARM:
        print(f"  {'NoMP Config':<12} NVMe P2PDMA/C2C requires matching cufile.json keys and NVMe multipath disabled")
        print(f"  {'Kernel Config':<12} RAID0 P2PDMA/C2C requires Linux kernel >= 7.1 plus runtime confirmation")
    print(f"  {'Need X.Y':<12} installed libcufile is older than the documented compat-path minimum")
    print(f"  {'?':<12} filesystem client not loaded — live state unknown")
    print(f"  {'>=1.16':<12} requires libcufile/GDS 1.16+ for squashfs/tmpfs/ramfs/overlayfs compat")
    print(f"  {'>=1.17':<12} requires libcufile/GDS 1.17+ for ZFS/BTRFS compat across all I/O APIs")
    print()
    print("  RDMA column: names the category of RDMA mechanism the filesystem uses, if any — each a")
    print("               prerequisite for that filesystem's Native GDS path (not a standalone alternative")
    print("               to it): dmabuf/peermem is cuFile userspace RDMA (GPFS, WekaFS); FS/kernel RDMA")
    print("               covers Lustre, BeeGFS, NFS (NFSoRDMA), NVMe-oF, and ScaTeFS — all still require")
    print("               nvidia-fs (nvfs) to be loaded, just via each filesystem's own kernel-level")
    print("               transport. In --live mode, FS/kernel RDMA only shows when this filesystem's")
    print("               Native column is also live-confirmed active — kernel RDMA isn't meaningful")
    print("               without nvidia-fs running.")
    if live_available:
        print()
        print(f"  Live ✓/No tokens sourced from: {gdscheck_path} -p (DRIVER CONFIGURATION)")
        print(f"  Config labels come from documented route prerequisites and release-note constraints.")
    print()
    return 0


def _run_json(args: argparse.Namespace) -> int:
    from checks.fs_matrix import FS_CAPABILITIES
    from checks.version import tool_metadata

    want_live = getattr(args, "live", False)
    want_static = getattr(args, "static", False)

    live_available = False
    driver_config: dict[str, str] = {}
    gdscheck_path: Optional[str] = None
    gdscheck_output: Optional[str] = None
    libcufile_info: Optional[dict] = None
    gdscheck_fallback_note: Optional[str] = None
    gdscheck_missing_payload: Optional[dict] = None

    if want_live:
        gdscheck_path = _find_gdscheck()
        if not gdscheck_path:
            gdscheck_missing_payload = _gdscheck_missing_payload()
            libcufile_info = cufile_version.detect_libcufile(None)
            if not _can_fallback_without_gdscheck(gdscheck_missing_payload, libcufile_info):
                gdscheck_missing_payload["tool"] = tool_metadata()
                print(json.dumps(gdscheck_missing_payload, indent=2))
                return 3
            gdscheck_fallback_note = _fallback_note(gdscheck_missing_payload, "gdscheck was not found")
        if gdscheck_path:
            raw, gdscheck_error = _run_gdscheck_raw(gdscheck_path, required_section="DRIVER CONFIGURATION")
            if not raw:
                gdscheck_missing_payload = _gdscheck_missing_payload()
                libcufile_info = libcufile_info or cufile_version.detect_libcufile(None)
                if not _can_fallback_without_gdscheck(gdscheck_missing_payload, libcufile_info):
                    print(json.dumps({
                        "tool": tool_metadata(),
                        "error": "gdscheck_no_output",
                        "gdscheck_detail": gdscheck_error,
                    }, indent=2))
                    return 3
                gdscheck_fallback_note = _fallback_note(
                    gdscheck_missing_payload,
                    gdscheck_error or "gdscheck did not return DRIVER CONFIGURATION output",
                )
            else:
                gdscheck_output = raw
                driver_config = _parse_driver_config(raw)
                live_available = True
    elif not want_static:
        gdscheck_path = _find_gdscheck()
        if gdscheck_path:
            raw, _ = _run_gdscheck_raw(gdscheck_path, required_section="DRIVER CONFIGURATION")
            if raw:
                gdscheck_output = raw
                driver_config = _parse_driver_config(raw)
                live_available = True

    if not want_static:
        libcufile_info = cufile_version.detect_libcufile(gdscheck_output)
    libcufile_tuple = _libcufile_version_tuple(libcufile_info)

    live_map = _live_status_map(driver_config) if live_available else {}

    _DISPLAY_FS = {fs for _, group in _FS_GROUPS for fs in group}
    _FS_TO_GROUP = {fs: group_name for group_name, group in _FS_GROUPS for fs in group}
    filesystems = []
    for fs, caps in FS_CAPABILITIES.items():
        if fs not in _DISPLAY_FS:
            continue
        entry: dict = {
            "fs_type": fs,
            "display_name": _FS_DISPLAY_NAMES.get(fs, fs),
            "category": _FS_TO_GROUP.get(fs),
            "alias_fs_types": _FS_DISPLAY_ALIASES.get(fs, []),
            "native":  caps.get("native",  False),
            "p2pdma":  caps.get("p2pdma",  False),
            "p2pdma_display": caps.get("p2pdma_display"),
            "p2pdma_note": caps.get("p2pdma_note"),
            "compat":  caps.get("compat",  True),
            "compat_since": caps.get("compat_since"),
            "compat_min_version": list(caps["compat_min_version"]) if caps.get("compat_min_version") else None,
            "compat_since_note": caps.get("compat_since_note"),
            "notes":   caps.get("notes",   ""),
        }
        compat_effective = _compat_ref_value(caps, libcufile_tuple)
        entry["compat_effective"] = compat_effective
        if isinstance(compat_effective, str) and compat_effective.startswith("needs-"):
            entry["compat_warning"] = (
                f"Requires libcufile/GDS >= {compat_effective.split('-', 1)[1]} "
                f"for this documented compatibility path."
            )
        if live_available:
            if fs in live_map:
                modes_str = live_map[fs]
                if modes_str:
                    supports_userspace_rdma = caps.get("rdma_type") == "userspace"
                    raw_native = _live_native_from_modes(modes_str, supports_userspace_rdma)
                    raw_p2pdma = _live_p2pdma_from_modes(fs, modes_str, driver_config)
                    raw_c2c = "c2c" in _mode_tokens(modes_str)
                    has_native = raw_native and bool(caps.get("native", False))
                    has_p2pdma = raw_p2pdma and bool(caps.get("p2pdma", False))
                    has_rdma = _live_rdma_from_modes(modes_str, supports_userspace_rdma)
                    entry["live_status"] = "supported" if (has_native or has_p2pdma) else "compat_only"
                    entry["live_modes"] = {
                        "native":  has_native,
                        "p2pdma":  has_p2pdma,
                        "c2c": has_p2pdma and raw_c2c,
                        "dmabuf_peermem": has_rdma,
                        "compat":  _live_compat_from_modes(caps, modes_str, libcufile_tuple),
                    }
                    entry["live_raw"] = modes_str
                    if raw_p2pdma and not has_p2pdma:
                        entry["live_warning"] = (
                            "gdscheck reports a p2pdma/c2c token, but this filesystem "
                            "is not a GDS library-supported direct P2P route."
                        )
                else:
                    entry["live_status"] = "client_not_loaded"
                    entry["live_modes"] = None
            else:
                entry["live_status"] = "not_tracked"
                entry["live_modes"] = None
        filesystems.append(entry)

    out = {
        "tool": tool_metadata(),
        "live_status_available": live_available,
        "live_requested": want_live,
        "gdscheck_path": gdscheck_path,
        "gdscheck_fallback": bool(gdscheck_fallback_note),
        "gdscheck_fallback_note": gdscheck_fallback_note,
        "gdscheck_missing": gdscheck_missing_payload,
        "libcufile": libcufile_info,
        "filesystems": filesystems,
            "legend": {
                "true":   "Supported",
                "false":  "Not supported",
                "config": "Requires matching cufile.json key and runtime confirmation",
                "NoMP Config": "NVMe P2PDMA/C2C requires matching cufile.json keys and NVMe multipath disabled on x86 — not required on NVIDIA CPUs, where multipath works fine alongside C2C",
                "Arch Config": "RAID0 P2PDMA/C2C requires NVIDIA Grace or Linux kernel >= 7.1 plus runtime confirmation",
            },
    }
    print(json.dumps(out, indent=2))
    return 0


class SupportMatrixCommand(Subcommand):
    name = "support-matrix"
    help = "print the GDS filesystem support matrix (--static or --live)"
    description = _DESCRIPTION
    order = 10

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument(
            "--static",
            action="store_true",
            default=False,
            help="show static reference table only; never run gdscheck (default if gdscheck absent)",
        )
        mode.add_argument(
            "--live",
            action="store_true",
            default=False,
            help="run gdscheck for live modes; fall back to libcufile-gated static data if possible",
        )

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = SupportMatrixCommand()
