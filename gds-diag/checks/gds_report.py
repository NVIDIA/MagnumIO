# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Filesystem-aware GPUDirect Storage mode compatibility checker.

Determines which GDS modes are available for a given path, explains blockers,
and recommends mitigations based on kernel configuration and system state.

Reference: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import subprocess
import textwrap
from typing import Optional

from .gdscheck import (
    driver_config_has_direct_p2pdma_token,
    driver_config_has_native,
)
from .result import CheckResult, GDSMode, ModeReport, Status
from .fs_matrix import (
    get_fs_type,
    get_fs_capabilities,
    check_odirect,
    check_ext4_data_mode,
    is_nvme_backed,
    get_nvme_transport,
    get_raid_level,
    get_dm_info,
    FS_CAPABILITIES,
    FS_ALIASES,
)
from .output import bold, cyan, dim, green, red, yellow, STATUS_ICON
from .gdscheck import _gdscheck_section

# Supported local block-backed filesystems for GDS (ext4 and xfs only).
# When no path is available, NVMe backing is assumed for these types.
_LOCAL_BLOCK_FS = {"ext4", "xfs"}
from . import kernel, iommu, pcie, nvidia_fs, rdma, cufile_config


# ---------------------------------------------------------------------------

def _raid0_p2pdma_supported_by_arch_or_kernel() -> bool:
    return iommu.is_grace() or kernel._kernel_version()[:2] >= (7, 1)


def _raid0_p2pdma_architecture_check() -> CheckResult:
    if iommu.is_grace():
        return CheckResult(
            check="RAID0 P2PDMA architecture",
            mode=GDSMode.P2PDMA,
            status=Status.PASS,
            why=(
                "NVIDIA Grace platform detected. RAID0 P2PDMA is a Grace-based "
                "supported path when block.raid.use_pci_p2pdma=true and runtime "
                "testing confirms activation."
            ),
        )

    major, minor, patch = kernel._kernel_version()
    ver_str = f"{major}.{minor}.{patch}"
    if (major, minor) >= (7, 1):
        return CheckResult(
            check="RAID0 P2PDMA architecture",
            mode=GDSMode.P2PDMA,
            status=Status.PASS,
            why=(
                f"Linux kernel {ver_str} detected. RAID0 P2PDMA is supported "
                "on non-Grace hosts with Linux kernel 7.1 or newer when "
                "block.raid.use_pci_p2pdma=true and runtime testing confirms "
                "activation."
            ),
        )

    return CheckResult(
        check="RAID0 P2PDMA architecture",
        mode=GDSMode.P2PDMA,
        status=Status.FAIL,
        why=(
            f"RAID0 P2PDMA requires NVIDIA Grace or Linux kernel >= 7.1. "
            f"This host is not detected as an NVIDIA Grace platform and is "
            f"running kernel {ver_str}, so upstream PCI P2PDMA cannot be used "
            "for this RAID0 route even if cufile.json requests "
            "block.raid.use_pci_p2pdma."
        ),
        mitigation=(
            "Use nvidia-fs/nvfs for direct GDS on this RAID0 mount when the NVMe "
            "stack has the required MLNX_OFED/DOCA GDS patches, or use compat mode. "
            "For upstream PCI P2PDMA on x86 RAID0, upgrade to Linux kernel >= 7.1 "
            "or test a non-RAID local NVMe or NVMe-oF route with the matching "
            "cufile.json keys."
        ),
    )


def _unsupported_device_mapper_check(dm_kind: str, mode: GDSMode) -> CheckResult:
    return CheckResult(
        check="Device-mapper backing device",
        mode=mode,
        status=Status.FAIL,
        why=(
            f"This mount is backed by a device-mapper device ({dm_kind}). "
            "Both nvidia-fs/nvfs and P2PDMA/C2C resolve a mount down to a raw "
            "NVMe (or supported RAID0) block device to set up direct DMA; "
            "device-mapper targets such as LVM, dm-crypt, and dm-multipath "
            "remap block addressing in ways GDS cannot see through, so direct "
            "GDS is not supported on this route regardless of what physical "
            "storage backs the device-mapper volume."
        ),
        mitigation=(
            "Remove the device-mapper layer for this mount: use the raw NVMe "
            "partition directly (no LVM, dm-crypt, or multipath), or a "
            "supported mdadm RAID0 route, then remount. If LVM, encryption, or "
            "multipath is required for operational reasons, GDS will not "
            "accelerate this path — use compat mode instead."
        ),
    )


def _unsupported_raid_direct_gds_check(raid_level: str, mode: GDSMode) -> CheckResult:
    return CheckResult(
        check="RAID level GDS support",
        mode=mode,
        status=Status.FAIL,
        why=(
            f"{raid_level.upper()} is not a supported direct GDS RAID route. "
            "NVIDIA documents RAID0 as the supported RAID route; other RAID "
            "levels should not be reported as native GDS or P2PDMA-capable for "
            "this mount."
        ),
        mitigation=(
            "Use a supported RAID0-over-NVMe route, a non-RAID supported NVMe "
            "or NVMe-oF route, or compat mode. Rebuild or remount storage on a "
            "documented GDS-supported route before expecting direct GDS."
        ),
    )


# ---------------------------------------------------------------------------
# gdscheck: authoritative mode verdict
# ---------------------------------------------------------------------------

# Maps filesystem type to keys in gdscheck's DRIVER CONFIGURATION section.
# Local NVMe-backed filesystems use the "NVMe" entry.
# Network filesystems have their own entries.
_FS_DRIVER_KEYS: dict[str, tuple[str, ...]] = {
    "ext4":            ("NVMe",),
    "xfs":             ("NVMe",),
    "ext3":            ("NVMe",),
    "ext2":            ("NVMe",),
    "nvme-of":         ("NVMeOF",),
    "nfs":             ("NFS",),
    "lustre":          ("Lustre", "DDN EXAScaler"),
    "fuse.fsx_lustre": ("Lustre", "DDN EXAScaler"),
    "beegfs":          ("BeeGFS",),
    "fhgfs":           ("BeeGFS",),
    "wekafs":          ("WekaFS",),
    "gpfs":            ("IBM Spectrum Scale",),
    "mmfs":            ("IBM Spectrum Scale",),
    "virtiofs":        ("VIRTIOFS",),
    "raid0":           ("NVMe",),
    "scatefs":         ("ScaTeFS",),
    "scaleflux":       ("ScaleFlux CSD",),
}


def _route_supports_direct_p2pdma(fs_type: str, nvme_backed: bool) -> bool:
    normalized = FS_ALIASES.get(fs_type, fs_type)
    caps = FS_CAPABILITIES.get(normalized, {})
    if caps.get("p2pdma") in (False, None):
        return False
    if normalized in _LOCAL_BLOCK_FS:
        return nvme_backed
    if normalized == "raid0":
        return _raid0_p2pdma_supported_by_arch_or_kernel()
    return True


def _run_gdscheck_raw() -> Optional[str]:
    """
    Run gdscheck -p, return stdout+stderr or None if tool not found.
    """
    candidates = (
        _glob.glob("/usr/local/cuda*/gds/tools/gdscheck")
        + ["/usr/local/cuda/gds/tools/gdscheck", "gdscheck"]
    )
    for c in candidates:
        try:
            r = subprocess.run([c, "-p"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and "DRIVER CONFIGURATION" in r.stdout:
                return r.stdout + r.stderr
        except Exception:
            continue
    return None



def _driver_config_modes(output: str, driver_key: str) -> Optional[str]:
    """
    Parse DRIVER CONFIGURATION section for a device key.
    Returns the modes string (e.g. "nvfs, compat") or None if key not found.
    gdscheck format: "  NVMe               : nvfs, compat"
    """
    for line in _gdscheck_section(output, "DRIVER CONFIGURATION"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        if key.strip().lower() == driver_key.lower():
            return val.strip()
    return None


def _driver_config_modes_any(output: str, driver_keys: tuple[str, ...]) -> Optional[str]:
    for driver_key in driver_keys:
        modes = _driver_config_modes(output, driver_key)
        if modes is not None:
            return modes
    return None


def _cufile_prop(output: str, prop_key: str) -> Optional[str]:
    """
    Parse CUFILE CONFIGURATION section for a dotted property key.
    Returns the value string (e.g. "true" / "false") or None if not found.
    gdscheck format: "  block.nvme.use_pci_p2pdma : true"
    """
    for line in _gdscheck_section(output, "CUFILE CONFIGURATION"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        if key.strip().lower() == prop_key.lower():
            return val.strip().lower()
    return None


def _gdscheck_verdict(fs_type: str, nvme_backed: bool) -> dict[GDSMode, bool]:
    """Convenience wrapper: run gdscheck and parse into mode verdict dict."""
    return _gdscheck_parse(_run_gdscheck_raw(), fs_type, nvme_backed)


def _infer_nvme_backed(path: str, fs_type: str) -> bool:
    """
    Determine whether the path/filesystem is NVMe-backed.
    When a path is given, probe the actual block device.
    When only fs_type is given (no path), infer from the filesystem type:
      local block FSes (ext4, xfs, ext3, ext2) → assume NVMe for P2PDMA purposes.
      network / parallel FSes → not NVMe-backed.
    """
    if path:
        return is_nvme_backed(path)
    return fs_type in _LOCAL_BLOCK_FS


def _gdscheck_rdma_diagnosis(output: str, fs_type: str = "") -> list[tuple[str, str]]:
    """
    Parse gdscheck RDMA sub-items for userspace RDMA filesystems (GPFS, WekaFS).

    Only GPFS and WekaFS use cuFile's userspace RDMA path (Userspace RDMA in gdscheck).
    Lustre uses kernel-level RDMA via LNet (nvidia-fs path) — no userspace RDMA needed.
    BeeGFS has no RDMA.

    gdscheck RDMA lines (only relevant for GPFS/WekaFS):
      Userspace RDMA         : Supported / Unsupported
      --Mellanox PeerDirect  : Enabled / Disabled
      --DmaBuf support       : Enabled / Disabled
      --rdma library         : Loaded (libcufile_rdma.so) / Not loaded
      --rdma devices         : Configured / Not configured
      --rdma_dev_addr_list_status : Up: N Down: M
    """
    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES
    normalized = FS_ALIASES.get(fs_type, fs_type)
    caps = FS_CAPABILITIES.get(normalized, {})
    if caps.get("rdma_type") != "userspace":
        return []  # Lustre/NFS/BeeGFS don't use userspace RDMA — nothing to diagnose here

    issues: list[tuple[str, str]] = []

    driver_keys = _FS_DRIVER_KEYS.get(normalized)
    driver_modes = _driver_config_modes_any(output, driver_keys) if driver_keys else None
    if driver_modes:
        driver_modes_lower = driver_modes.lower()
        if "dmabuf" in driver_modes_lower or "nvidia_peermem" in driver_modes_lower:
            return []

    userspace_rdma = _driver_config_modes(output, "Userspace RDMA")
    if userspace_rdma is None:
        return []  # RDMA section absent from gdscheck output — can't diagnose

    if "unsupported" in userspace_rdma.lower():
        issues.append((
            "Userspace RDMA is Unsupported — MLNX_OFED/DOCA not installed or RDMA library missing",
            (
                "1. Check if MLNX_OFED is installed:  ofed_info -s\n"
                "2. Check if DOCA is installed:       doca_version\n"
                "3. Verify rdma library exists:       ls /usr/local/cuda/lib64/libcufile_rdma.so\n"
                "4. Install MLNX_OFED (includes RDMA drivers + nvidia_peermem):\n"
                "     sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma\n"
                "   Or install DOCA for ConnectX-7 / BlueField hardware."
            ),
        ))
        return issues  # sub-items are meaningless if RDMA is fully unsupported

    # --- rdma library ---
    rdma_lib = _driver_config_modes(output, "--rdma library")
    if rdma_lib and ("not" in rdma_lib.lower() or "loaded" not in rdma_lib.lower()):
        issues.append((
            "RDMA library not loaded (libcufile_rdma.so missing)",
            (
                "The rdma library is loaded automatically when MLNX_OFED/DOCA is installed.\n"
                "Check: ls /usr/local/cuda/lib64/libcufile_rdma.so\n"
                "Install MLNX_OFED: sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma\n"
                "After install, verify: gdscheck -p | grep 'rdma library'"
            ),
        ))

    # --- PeerDirect vs DmaBuf ---
    # Either path is sufficient — PeerDirect (nvidia_peermem) OR DmaBuf.
    # Only flag an issue when BOTH are unavailable.
    peerdirect = _driver_config_modes(output, "--Mellanox PeerDirect")
    dmabuf     = _driver_config_modes(output, "--DmaBuf support")

    peer_disabled  = bool(peerdirect and "disabled" in peerdirect.lower())
    dmabuf_enabled = bool(dmabuf and "enabled" in dmabuf.lower())

    if peer_disabled and not dmabuf_enabled:
        # Neither path available — only fix is to load nvidia_peermem
        issues.append((
            "Neither PeerDirect nor DmaBuf is available for RDMA",
            (
                "Load nvidia_peermem to enable the PeerDirect path:\n"
                "  sudo modprobe nvidia_peermem\n"
                "  echo 'nvidia_peermem' | sudo tee /etc/modules-load.d/nvidia-peermem.conf\n\n"
                "Note: DmaBuf path (alternative, no nvidia_peermem needed) requires kernel ≥ 5.12\n"
                "  and MLNX_OFED ≥ 5.6 or DOCA."
            ),
        ))

    # --- rdma devices (IPs in cufile.json) ---
    rdma_devices_val = _driver_config_modes(output, "--rdma devices")
    if rdma_devices_val and "not configured" in rdma_devices_val.lower():
        issues.append((
            "RDMA devices not configured — no RDMA client IPs set in cufile.json",
            (
                "Add your RDMA NIC IP addresses to /etc/cufile.json:\n"
                '  "properties": {\n'
                '    "rdma_dev_addr_list": ["<rdma-nic-ip-1>", "<rdma-nic-ip-2>"]\n'
                '  }\n\n'
                "Find your RDMA NIC IPs:\n"
                "  ip addr show   (look for the IB/RoCE interface)\n"
                "  ibstat | grep -A5 'CA '   (for InfiniBand)\n"
                "After adding, verify: gdscheck -p | grep 'rdma devices'"
            ),
        ))

    # --- rdma_dev_addr_list_status (link up/down) ---
    dev_status = _driver_config_modes(output, "--rdma_dev_addr_list_status")
    if dev_status:
        down_m = re.search(r"Down:\s*(\d+)", dev_status, re.I)
        up_m   = re.search(r"Up:\s*(\d+)",   dev_status, re.I)
        down   = int(down_m.group(1)) if down_m else 0
        up     = int(up_m.group(1))   if up_m   else 0
        if down > 0 and up == 0:
            issues.append((
                f"All RDMA devices are down ({dev_status.strip()})",
                (
                    "Check IB/RoCE port state:\n"
                    "  ibstat                          # look for 'State: Active'\n"
                    "  ibv_devinfo | grep port_state   # should show PORT_ACTIVE\n"
                    "Check cable connections and ensure subnet manager is running:\n"
                    "  systemctl status opensmd\n"
                    "Restart OpenIB stack if needed:\n"
                    "  sudo /etc/init.d/openibd restart"
                ),
            ))
        elif down > 0:
            issues.append((
                f"Some RDMA devices are down ({dev_status.strip()}) — bandwidth or failover may be reduced",
                (
                    "Check which links are down: ibstat\n"
                    "Check port state: ibv_devinfo | grep port_state"
                ),
            ))

    return issues


def _gdscheck_native_inactive_diagnosis(
    output: str,
    fs_type: str,
    nvme_backed: bool,
) -> list[tuple[str, str, str]]:
    """
    Explain why native nvidia-fs/nvfs is not active when gdscheck has a
    DRIVER CONFIGURATION line for the route but it lacks a native token.

    Returns (check, why, mitigation) tuples.
    """
    if not output:
        return []

    driver_keys = _FS_DRIVER_KEYS.get(fs_type)
    if driver_keys is None and nvme_backed:
        driver_keys = ("NVMe",)
    if not driver_keys:
        return []

    modes = _driver_config_modes_any(output, driver_keys)
    if modes is None or driver_config_has_native(modes):
        return []

    driver_label = " or ".join(driver_keys)
    if (
        driver_config_has_direct_p2pdma_token(modes)
        and _route_supports_direct_p2pdma(fs_type, nvme_backed)
    ):
        return [(
            "gdscheck nvidia-fs/nvfs route",
            (
                f"gdscheck reports {driver_label}: {modes}; nvidia-fs/nvfs is not "
                "the active route because direct P2PDMA/C2C is active."
            ),
            (
                "No action is required if P2PDMA/C2C is the intended direct path. "
                "If you specifically need nvidia-fs/nvfs, disable the direct P2P "
                "route and verify the storage stack supports nvfs."
            ),
        )]

    if driver_keys == ("NVMe",) or fs_type == "nvme-of":
        return [(
            "gdscheck nvidia-fs/nvfs route",
            (
                f"gdscheck reports {driver_label}: {modes}; no nvfs/native token "
                "is active even though the local nvidia-fs prerequisites passed. "
                "For NVMe/NVMe-oF, nvidia-fs mode also depends on a GDS-enabled "
                "storage stack, not just the nvidia_fs kernel module."
            ),
            (
                "For NVMe/NVMe-oF nvidia-fs mode, install or verify the NVIDIA "
                "GDS storage-stack patches from MLNX_OFED or DOCA/DOCA-OFED, then "
                "rerun gdscheck and confirm DRIVER CONFIGURATION shows an nvfs "
                "token for the route.\n"
                "GDS DOCA requirements:\n"
                "https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#doca-requirements-and-installation\n"
                "DOCA storage installation:\n"
                "https://docs.nvidia.com/doca/sdk/doca-host-installation-and-upgrade/index.html#storage-installation\n"
                "If you intend to use upstream PCI P2PDMA/C2C instead of nvfs, "
                "fix the P2PDMA/C2C findings for this mount."
            ),
        )]

    return [(
        "gdscheck nvidia-fs/nvfs route",
        f"gdscheck reports {driver_label}: {modes}; no native GDS token is active for this route.",
        "Review gdscheck -p DRIVER CONFIGURATION for the filesystem and verify the required client, driver, and nvidia-fs stack are installed.",
    )]


def _gdscheck_native_inactive_result(
    output: str,
    fs_type: str,
    nvme_backed: bool,
) -> Optional[CheckResult]:
    diagnoses = _gdscheck_native_inactive_diagnosis(output, fs_type, nvme_backed)
    if not diagnoses:
        return None

    check, why, mitigation = diagnoses[0]
    status = Status.INFO if "P2PDMA/C2C is active" in why else Status.WARN
    return CheckResult(
        check=check,
        mode=GDSMode.NATIVE,
        status=status,
        why=why,
        mitigation=mitigation,
    )


# ---------------------------------------------------------------------------
# Core: run all checks for a given filesystem type and path
# ---------------------------------------------------------------------------

def _report_for(reports: list[ModeReport], mode: GDSMode) -> Optional[ModeReport]:
    return next((r for r in reports if r.mode == mode), None)


def _has_static_fail(report: Optional[ModeReport]) -> bool:
    return report is not None and any(r.status == Status.FAIL for r in report.results)


def _enrich_with_gdscheck(
    reports: list[ModeReport],
    gds_output: str,
    fs_type: str,
    parse_fs_type: str,
    nvme_backed: bool,
    unsupported_raid_level: Optional[str] = None,
) -> None:
    """Append gdscheck-derived CheckResults to each ModeReport in-place.

    Called after all static checks are complete so that filter helpers have
    access to the full static result set when deciding which gdscheck
    blockers to surface.  After this call the ModeReports are complete and
    can be rendered or serialised without any further gdscheck invocation.
    """
    gds = _gdscheck_parse(gds_output, parse_fs_type, nvme_backed)
    driver_keys = _FS_DRIVER_KEYS.get(parse_fs_type)
    if driver_keys is None and nvme_backed:
        driver_keys = ("NVMe",)
    driver_label = " or ".join(driver_keys) if driver_keys else (fs_type or "filesystem")

    for report in reports:
        if not report.applicable:
            continue

        supported = gds.get(report.mode)  # True, False, or None

        if supported is None:
            # gdscheck ran but the FS client is not active — cannot verify.
            # Only emit this for modes gdscheck actually tracks via DRIVER
            # CONFIGURATION (NATIVE and P2PDMA when driver_keys is set).
            # COMPAT is never tracked by gdscheck; RDMA is handled separately
            # by _gdscheck_rdma_diagnosis when applicable.
            if driver_keys and report.mode in (GDSMode.NATIVE, GDSMode.P2PDMA):
                report.results.append(CheckResult(
                    check="gdscheck client detection",
                    mode=report.mode,
                    status=Status.INFO,
                    why=(
                        f"No active {driver_label} client found by gdscheck. "
                        "Mount the filesystem and re-run for an authoritative mode verdict."
                    ),
                ))

        elif not supported:
            # Mode is not active per gdscheck — surface the diagnosis
            if report.mode == GDSMode.NATIVE:
                result = _gdscheck_native_inactive_result(gds_output, parse_fs_type, nvme_backed)
                # Don't claim P2PDMA/C2C is the active alternative when this
                # mount's own static P2PDMA checks already failed.
                if result and not (
                    result.status == Status.INFO
                    and _has_static_fail(_report_for(reports, GDSMode.P2PDMA))
                ):
                    report.results.append(result)

            elif report.mode == GDSMode.P2PDMA:
                # gdscheck only sees the filesystem/driver state, not static
                # blockers found above (device-mapper, RAID level, ext4 data
                # mode, ...), so only claim nvfs is available if the Native
                # report itself has no failures.
                if (
                    gds.get(GDSMode.NATIVE) is True
                    and not _has_static_fail(_report_for(reports, GDSMode.NATIVE))
                    and not unsupported_raid_level
                    and parse_fs_type != "raid0"
                    and fs_type in {"ext4", "xfs", "nvme-of"}
                ):
                    report.results.append(CheckResult(
                        check="gdscheck P2PDMA/C2C route",
                        mode=GDSMode.P2PDMA,
                        status=Status.INFO,
                        why=(
                            "Direct GDS is still available via nvidia-fs/nvfs. "
                            "This is valid for NVMe/NVMe-oF stacks with the required "
                            "MLNX_OFED/DOCA GDS patches."
                        ),
                    ))
                else:
                    raw = _gdscheck_p2pdma_blockers(gds_output, parse_fs_type)
                    for why, mitigation in _filter_p2pdma_blockers_against_static_checks(raw, report):
                        report.results.append(CheckResult(
                            check="gdscheck P2PDMA/C2C route",
                            mode=GDSMode.P2PDMA,
                            status=Status.WARN,
                            why=why,
                            mitigation=mitigation,
                        ))

            elif report.mode == GDSMode.RDMA:
                for why, mitigation in _gdscheck_rdma_diagnosis(gds_output, fs_type):
                    report.results.append(CheckResult(
                        check="gdscheck RDMA route",
                        mode=GDSMode.RDMA,
                        status=Status.WARN,
                        why=why,
                        mitigation=mitigation,
                    ))


def build_mode_reports(path: str, fs_type: str) -> list[ModeReport]:
    # Canonicalize once so every raw fs_type comparison below (and in callers
    # of this function's return value) sees the same resolved name that
    # get_fs_capabilities() uses internally — avoids needing every comparison
    # site to separately enumerate alias spellings like ("nfs", "nfs4").
    fs_type = FS_ALIASES.get(fs_type, fs_type)
    caps = get_fs_capabilities(fs_type)
    reports: list[ModeReport] = []
    nvme_transport = get_nvme_transport(path) if path and fs_type in _LOCAL_BLOCK_FS else None
    raid_level = get_raid_level(path) if path and fs_type in _LOCAL_BLOCK_FS else None
    unsupported_raid_level = raid_level if raid_level and raid_level != "raid0" else None
    dm_kind = get_dm_info(path) if path and fs_type in _LOCAL_BLOCK_FS else None
    if raid_level == "raid0":
        p2pdma_block_key = "raid"
    elif nvme_transport and nvme_transport != "pcie":
        p2pdma_block_key = "nvmeof"
    else:
        p2pdma_block_key = None

    # Parse cufile.json once — results are shared across mode reports.
    _cufile_all = cufile_config.run_all(fs_type, p2pdma_block_key=p2pdma_block_key)

    # Run gdscheck once here for the open driver check (shared by NATIVE and P2PDMA).
    # _build_mode_support() runs it again for the mode verdict — two total invocations.
    _gds_raw_for_driver = _run_gdscheck_raw() or ""

    # ------------------------------------------------------------------ #
    # Native GDS (nvidia-fs)                                               #
    # ------------------------------------------------------------------ #
    native_applicable = caps.get("native") not in (False, None)
    native_report = ModeReport(mode=GDSMode.NATIVE, applicable=native_applicable)

    if native_applicable:
        # Kernel + driver + GPU checks (shared across native and p2pdma)
        native_report.results += kernel.run_all(GDSMode.NATIVE)
        native_report.results += iommu.run_all(GDSMode.NATIVE)
        native_report.results += nvidia_fs.run_all()
        native_report.results.append(nvidia_fs.check_open_driver(_gds_raw_for_driver, GDSMode.NATIVE))
        if unsupported_raid_level:
            native_report.results.append(
                _unsupported_raid_direct_gds_check(unsupported_raid_level, GDSMode.NATIVE)
            )
        if dm_kind:
            native_report.results.append(
                _unsupported_device_mapper_check(dm_kind, GDSMode.NATIVE)
            )
        # cufile.json: only include checks relevant to native GDS (e.g. force_compat_mode).
        # P2PDMA settings (use_pci_p2pdma) belong to the P2PDMA report, not here.
        native_report.results += [r for r in _cufile_all if r.mode != GDSMode.P2PDMA]

        # ext4: GDS requires explicit data=ordered. Do not treat the implicit
        # kernel default as sufficient because it is not a durable visible mount
        # option for operators validating GDS readiness.
        if fs_type == "ext4":
            if path:
                result = check_ext4_data_mode(path)
                if result is not None:
                    mode, evidence = result
                    mountpoint = evidence.split(" opts:", 1)[0]
                    if mode == "ordered":
                        native_report.results.append(CheckResult(
                            check="ext4 data mode",
                            mode=GDSMode.NATIVE,
                            status=Status.PASS,
                            why="ext4 is explicitly mounted with data=ordered, as required for GDS.",
                            evidence=evidence,
                        ))
                    else:
                        root_fs = mountpoint == "/"
                        if mode == "journal":
                            why = (
                                "ext4 is mounted with data=journal. This mode disables O_DIRECT, "
                                "which is required for all GDS modes. No GDS acceleration is possible."
                            )
                        elif mode == "default":
                            why = (
                                "ext4 is not explicitly mounted with data=ordered. GDS requires "
                                "data=ordered to be visible in the active mount options; the implicit "
                                "ext4 default is not sufficient for GDS readiness."
                            )
                        else:
                            why = (
                                f"ext4 is mounted with data={mode}. GDS requires ext4 to be "
                                "explicitly mounted with data=ordered."
                            )
                        if root_fs:
                            mitigation = (
                                "For the root filesystem, add rootflags=data=ordered to "
                                "GRUB_CMDLINE_LINUX in /etc/default/grub, then run:\n"
                                "  sudo update-grub              # Ubuntu/Debian\n"
                                "  sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/Rocky\n"
                                "  sudo reboot\n"
                                "Verify after reboot: findmnt -no OPTIONS / | tr ',' '\\n' | grep '^data=ordered$'"
                            )
                        else:
                            mitigation = (
                                f"Remount with explicit data=ordered:\n"
                                f"  sudo mount -o remount,data=ordered {mountpoint}\n"
                                "For persistence, add data=ordered to this filesystem's /etc/fstab options.\n"
                                f"Verify: findmnt -no OPTIONS {mountpoint} | tr ',' '\\n' | grep '^data=ordered$'"
                            )
                        native_report.results.append(CheckResult(
                            check="ext4 data mode",
                            mode=GDSMode.NATIVE,
                            status=Status.FAIL,
                            why=why,
                            mitigation=mitigation,
                            evidence=evidence,
                        ))
            else:
                # No path — cannot probe mount options; remind operator to verify.
                native_report.results.append(CheckResult(
                    check="ext4 data mode",
                    mode=GDSMode.NATIVE,
                    status=Status.WARN,
                    why=(
                        "Cannot verify ext4 mount mode without a path. "
                        "GDS requires ext4 to be explicitly mounted with data=ordered."
                    ),
                    mitigation=(
                        "Verify the mount option:\n"
                        "  findmnt -no OPTIONS <mountpoint> | tr ',' '\\n' | grep '^data=ordered$'\n"
                        "For non-root filesystems, mount or remount with -o data=ordered and add it to /etc/fstab.\n"
                        "For the root filesystem, add rootflags=data=ordered to GRUB_CMDLINE_LINUX and reboot."
                    ),
                ))

        # O_DIRECT probe
        if path:
            ok, ev = check_odirect(path)
            if ok is None:
                status, why, mitigation = (
                    Status.WARN,
                    "O_DIRECT support could not be verified for this filesystem mount.",
                    "Re-run with write access to this path for a definitive O_DIRECT probe.",
                )
            elif ok:
                status, why, mitigation = (
                    Status.PASS,
                    "O_DIRECT is supported — required for all native GDS I/O.",
                    None,
                )
            else:
                status, why, mitigation = (
                    Status.FAIL,
                    "O_DIRECT is NOT supported on this filesystem mount. "
                    "All GDS native modes require O_DIRECT to bypass the page cache.",
                    "Check mount options: mount | grep $(df --output=target {path} | tail -1)\n"
                    "Some filesystems need specific mount flags for O_DIRECT support.\n"
                    "For NFS: add 'rsize=1048576,wsize=1048576' and ensure server supports it.\n"
                    "If on tmpfs/overlayfs: O_DIRECT is not available — use a real block device.",
                )
            native_report.results.append(CheckResult(
                check="O_DIRECT support",
                mode=GDSMode.NATIVE,
                status=status,
                why=why,
                mitigation=mitigation,
                evidence=ev,
            ))

        # WekaFS write support is gated by fs.weka.rdma_write_support in cufile.json
        # (default false) — not a fixed architectural limitation. Report which state
        # is actually configured rather than assuming writes always fall back. The
        # top-level capability table now reports WekaFS native as a plain Yes, so
        # this check is gated on fs_type directly rather than a capability sentinel.
        if fs_type == "wekafs":
            native_report.results.append(cufile_config.check_weka_write_support())

    else:
        native_report.results.append(CheckResult(
            check="Filesystem GDS support",
            mode=GDSMode.NATIVE,
            status=Status.NA,
            why=(
                f"{fs_type} does not support native GDS (nvidia-fs direct DMA). "
                f"{caps.get('notes', '')}"
            ),
            mitigation=_native_fs_mitigation(fs_type),
        ))

    reports.append(native_report)

    # ------------------------------------------------------------------ #
    # P2P DMA (NVMe ↔ GPU direct)                                          #
    # ------------------------------------------------------------------ #
    p2pdma_cap = caps.get("p2pdma")
    p2pdma_applicable = p2pdma_cap is not False and p2pdma_cap is not None
    p2pdma_report = ModeReport(mode=GDSMode.P2PDMA, applicable=p2pdma_applicable)

    if p2pdma_applicable:
        local_block_p2pdma = fs_type in _LOCAL_BLOCK_FS
        p2pdma_report.results += kernel.run_all(GDSMode.P2PDMA)
        p2pdma_report.results += iommu.run_all(GDSMode.P2PDMA)
        if local_block_p2pdma:
            p2pdma_report.results += pcie.run_all()
        else:
            p2pdma_report.results.append(pcie.check_acs())
        p2pdma_report.results.append(nvidia_fs.check_open_driver(_gds_raw_for_driver, GDSMode.P2PDMA))
        p2pdma_registry_result = nvidia_fs.check_p2pdma_driver_registries()
        if p2pdma_registry_result is not None:
            p2pdma_report.results.append(p2pdma_registry_result)
        if raid_level == "raid0":
            p2pdma_report.results.append(_raid0_p2pdma_architecture_check())
        elif unsupported_raid_level:
            p2pdma_report.results.append(
                _unsupported_raid_direct_gds_check(unsupported_raid_level, GDSMode.P2PDMA)
            )
        if dm_kind:
            p2pdma_report.results.append(
                _unsupported_device_mapper_check(dm_kind, GDSMode.P2PDMA)
            )
        nvme_backed = _infer_nvme_backed(path, fs_type)
        if path and fs_type in _LOCAL_BLOCK_FS and nvme_transport:
            block_key = "block.nvmeof.use_pci_p2pdma" if nvme_transport != "pcie" else "block.nvme.use_pci_p2pdma"
            p2pdma_report.results.append(CheckResult(
                check="NVMe transport",
                mode=GDSMode.P2PDMA,
                status=Status.PASS,
                why=(
                    f"Backing NVMe transport is {nvme_transport}; mount-check will use "
                    f"{block_key} for the P2PDMA cufile.json check."
                ),
            ))

        # Reuse cufile results already computed for native GDS (avoids a second parse).
        # P2PDMA needs all results — including the P2PDMA settings check filtered out above.
        p2pdma_report.results += _cufile_all
        if path and local_block_p2pdma and not nvme_backed:
            # Only warn about non-NVMe backing when we have an actual path to check.
            # When no path is given, local FSes are assumed NVMe-backed.
            p2pdma_report.results.append(CheckResult(
                check="NVMe backing device",
                mode=GDSMode.P2PDMA,
                status=Status.WARN,
                why=(
                    "The path does not appear to be on an NVMe device. "
                    "For ext4/XFS, P2PDMA is only a GDS library-supported route "
                    "when the filesystem is backed by NVMe."
                ),
                mitigation="Verify the backing device: lsblk -s -o NAME,ROTA,TYPE $(df --output=source <path> | tail -1)",
            ))
        elif not path and nvme_backed:
            p2pdma_report.results.append(CheckResult(
                check="NVMe backing device",
                mode=GDSMode.P2PDMA,
                status=Status.PASS,
                why=f"Assuming NVMe backing for {fs_type} (filesystem-type check, no path given).",
            ))
    else:
        p2pdma_report.results.append(CheckResult(
            check="Filesystem P2PDMA support",
            mode=GDSMode.P2PDMA,
            status=Status.NA,
            why=(
                f"P2PDMA is not applicable for {fs_type}. "
                "The GDS library supports P2PDMA only for NVMe, NVMe-oF, "
                "virtiofs, and RAID0 routes. "
                f"{caps.get('notes', '').split('.')[0]}."
            ).rstrip(" .") + ".",
        ))

    reports.append(p2pdma_report)

    # ------------------------------------------------------------------ #
    # RDMA (network storage path)                                          #
    # ------------------------------------------------------------------ #
    rdma_applicable = caps.get("rdma", False) is True
    rdma_report = ModeReport(mode=GDSMode.RDMA, applicable=rdma_applicable)

    if rdma_applicable:
        rdma_report.results += rdma.run_all(fs_type)

        # NFS special case: check rdma mount option
        if fs_type == "nfs":
            rdma_report.results.append(_check_nfs_rdma_mount(path))

    else:
        rdma_report.results.append(CheckResult(
            check="Filesystem RDMA support",
            mode=GDSMode.RDMA,
            status=Status.NA,
            why=f"RDMA is not applicable for {fs_type}. {caps.get('notes', '').split('.')[0]}.",
        ))

    reports.append(rdma_report)

    # ------------------------------------------------------------------ #
    # Compat / bounce buffer (always available)                            #
    # ------------------------------------------------------------------ #
    compat_report = ModeReport(mode=GDSMode.COMPAT, applicable=True)
    compat_report.results += cufile_config.run_compat_checks()

    reports.append(compat_report)

    # Enrich reports with gdscheck-derived diagnosis now that all static
    # checks are in place.  parse_fs_type mirrors the logic in _build_mode_support
    # but uses the values already computed above.
    if _gds_raw_for_driver:
        if raid_level == "raid0":
            parse_fs_type = "raid0"
        elif unsupported_raid_level:
            parse_fs_type = unsupported_raid_level
        elif nvme_transport and nvme_transport != "pcie":
            parse_fs_type = "nvme-of"
        else:
            parse_fs_type = fs_type
        _enrich_with_gdscheck(
            reports,
            _gds_raw_for_driver,
            fs_type,
            parse_fs_type,
            _infer_nvme_backed(path, fs_type),
            unsupported_raid_level,
        )

    _mark_alternate_path_exemptions(reports, fs_type)

    return reports


def _native_fs_mitigation(fs_type: str) -> str:
    mitigations = {
        "tmpfs":    ("tmpfs does not support O_DIRECT — required for GDS. "
                     "Use a real block device mounted as ext4 or xfs."),
        "overlay":  ("OverlayFS (container filesystems) cannot use GDS. "
                     "Bind-mount a host ext4/xfs/NVMe path directly into the container "
                     "(not via overlay) to enable GDS inside containers."),
        "btrfs":    ("btrfs is not GDS-supported. Migrate to ext4 or xfs on the same device."),
        "virtiofs": ("VirtioFS (VM) cannot use GDS. "
                     "Use passthrough NVMe (VFIO) or a GDS-capable filesystem over the hypervisor network."),
        "fuse":     ("FUSE filesystems cannot use GDS due to userspace I/O interposition. "
                     "If this is a network filesystem, check if a native client is available (e.g. Lustre, BeeGFS)."),
        "cifs":     ("CIFS/SMB not supported. No GDS path available for SMB storage."),
    }
    return mitigations.get(fs_type, (
        f"GDS native mode is not available for '{fs_type}'. "
        "Consider using a supported filesystem: ext4, xfs, Lustre, BeeGFS, GPFS, WekaFS."
    ))


def _check_nfs_rdma_mount(path: str) -> CheckResult:
    """Check if NFS mount was done with an RDMA transport option."""
    if not path or path.strip() == "":
        return CheckResult(
            check="NFS rdma mount option",
            mode=GDSMode.RDMA,
            status=Status.WARN,
            why="Path was not provided; could not verify NFS rdma mount option.",
        )

    try:
        realpath = os.path.realpath(path)
        best_match = None
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 4:
                    continue
                mountpoint = parts[1]
                if realpath == mountpoint or realpath.startswith(mountpoint.rstrip("/") + "/"):
                    if best_match is None or len(mountpoint) > len(best_match[1]):
                        best_match = (line.strip(), mountpoint, parts[3])
        if best_match:
            line, _, mount_opts = best_match
            opt_set = set(mount_opts.split(","))
            opt_map = dict(
                opt.split("=", 1)
                for opt in opt_set
                if "=" in opt
            )
            if "rdma" in opt_set or opt_map.get("proto") == "rdma":
                return CheckResult(
                    check="NFS rdma mount option",
                    mode=GDSMode.RDMA,
                    status=Status.PASS,
                    why="NFS is mounted with RDMA transport — NFSoRDMA active.",
                    evidence=line,
                )
            return CheckResult(
                check="NFS rdma mount option",
                mode=GDSMode.RDMA,
                status=Status.FAIL,
                why=(
                    "NFS is mounted WITHOUT the 'rdma' transport option. "
                    "NFSoRDMA (required for RDMA GDS on NFS) needs an explicit rdma mount."
                ),
                mitigation=(
                    "Remount with the rdma transport:\n"
                    "  sudo umount /path/to/nfs\n"
                    "  sudo mount -t nfs -o proto=rdma,port=20049,rsize=1048576,wsize=1048576 \\\n"
                    "      server:/export /path/to/nfs\n"
                    "Or add to /etc/fstab:\n"
                    "  server:/export /mnt/nfs nfs proto=rdma,port=20049,rsize=1048576 0 0\n"
                    "Also ensure MLNX_OFED/DOCA or the vendor-supported NFS-RDMA client stack is installed."
                ),
                evidence=line,
            )
    except Exception:
        pass

    return CheckResult(
        check="NFS rdma mount option",
        mode=GDSMode.RDMA,
        status=Status.WARN,
        why="Could not read /proc/mounts to verify NFS rdma mount option.",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_text_report(
    path: str,
    fs_type: str,
    reports: list[ModeReport],
    verbose: bool = False,
) -> str:
    caps = get_fs_capabilities(fs_type)
    lines: list[str] = []

    lines.append("")
    lines.append(bold("═" * 70))
    lines.append(bold("  GDS Pre-Check Report"))
    lines.append(bold("═" * 70))
    if path:
        lines.append(f"  Path       : {path}")
    else:
        nvme_note = " (NVMe-backed assumed)" if fs_type in _LOCAL_BLOCK_FS else ""
        lines.append(f"  Path       : (filesystem-type check{nvme_note})")
    lines.append(f"  Filesystem : {fs_type}")
    if caps.get("notes"):
        _notes_prefix = "  Notes      : "
        lines.append(
            _notes_prefix + textwrap.fill(
                caps["notes"],
                width=65,
                subsequent_indent=" " * len(_notes_prefix),
                break_on_hyphens=False,
                break_long_words=False,
            )
        )
    lines.append("")

    # Summary table
    lines.append(bold("  Mode Availability"))
    lines.append("  " + "─" * 66)
    lines.append(f"  {'Mode':<38} {'Status':<14} {'Blockers'}")
    lines.append("  " + "─" * 66)

    for report in reports:
        icon = STATUS_ICON[report.status]
        if report.status == Status.NA:
            lines.append(f"  {report.mode.value:<38} {icon}")
        else:
            # Only hard FAILs in the Blockers column — WARNs are not blockers
            blockers = report.blockers()
            blocker_summary = (
                ", ".join(r.check for r in blockers[:2]) + ("…" if len(blockers) > 2 else "")
                if blockers else ""
            )
            lines.append(f"  {report.mode.value:<38} {icon}        {blocker_summary}")

    lines.append("  " + "─" * 66)
    _mode_parts = []
    _counts = {s: sum(1 for r in reports if r.status == s) for s in Status}
    if _counts[Status.PASS]:
        _mode_parts.append(green(f"{_counts[Status.PASS]} available"))
    if _counts[Status.INFO]:
        _mode_parts.append(cyan(f"{_counts[Status.INFO]} info"))
    if _counts[Status.WARN]:
        _mode_parts.append(yellow(f"{_counts[Status.WARN]} warning{'s' if _counts[Status.WARN] != 1 else ''}"))
    if _counts[Status.FAIL]:
        _mode_parts.append(red(f"{_counts[Status.FAIL]} blocked"))
    if _counts[Status.NA]:
        _mode_parts.append(dim(f"{_counts[Status.NA]} N/A"))
    lines.append("  " + ", ".join(_mode_parts))
    lines.append("")

    # Detailed findings — always shown, covers every mode
    lines.append(bold("  Detailed Findings"))
    lines.append("")

    detail_reports = reports if verbose else [
        report for report in reports if report.status in (Status.FAIL, Status.WARN, Status.INFO, Status.NA)
    ]
    if not detail_reports:
        lines.append("  No warnings or errors.")
        lines.append("")

    from .output import render_section as _render_section
    for report in detail_reports:
        lines.extend(_render_section(
            report.mode.value,
            report.results,
            verbose=verbose,
            status=report.status,
        ))

    from .output import render_mitigation_plan
    mitigation_sections = {
        report.mode.value: report.results
        for report in reports
        if report.results
    }
    lines.extend(render_mitigation_plan(mitigation_sections))

    return "\n".join(lines)


def _gdscheck_p2pdma_blockers(output: str, fs_type: str = "") -> list[tuple[str, str]]:
    """
    Determine why P2PDMA/C2C is not supported, in priority order:
      1. cufile.json settings (most common — properties and block-level must both be true)
      2. Hardware blockers from PLATFORM INFO (ACS, IOMMU)
    Returns list of (why, mitigation) tuples.
    If cufile.json is the blocker, returns early — no point surfacing ACS noise.
    """
    blockers = []

    # --- cufile.json settings (check first — this is the most common reason) ---
    # Only GDS library-supported P2PDMA/C2C routes should reach this blocker path:
    # local NVMe (ext4/xfs), NVMe-oF, virtiofs, and RAID0.
    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES, P2PDMA_CONFIG_KEY
    normalized_fs = FS_ALIASES.get(fs_type, fs_type)
    caps = FS_CAPABILITIES.get(normalized_fs, {})
    p2pdma_cap = caps.get("p2pdma")

    unsupported_raid_match = re.fullmatch(r"raid([1-9][0-9]*)", normalized_fs or "")
    if unsupported_raid_match and normalized_fs != "raid0":
        blockers.append((
            (
                f"{normalized_fs.upper()} is not a supported direct GDS RAID route. "
                "Only RAID0 is documented as a supported RAID route."
            ),
            (
                "Use RAID0 over supported NVMe devices, a non-RAID supported NVMe "
                "or NVMe-oF route, or compat mode."
            ),
        ))

    if normalized_fs == "raid0" and not _raid0_p2pdma_supported_by_arch_or_kernel():
        major, minor, patch = kernel._kernel_version()
        ver_str = f"{major}.{minor}.{patch}"
        blockers.append((
            (
                "RAID0 P2PDMA requires NVIDIA Grace or Linux kernel >= 7.1. "
                "This host is not detected as an NVIDIA Grace platform and is "
                f"running kernel {ver_str}, so upstream PCI P2PDMA is not active "
                "for this RAID0 route even if gdscheck reports an NVMe P2PDMA token."
            ),
            (
                "Use nvidia-fs/nvfs for direct GDS on this RAID0 mount when the NVMe "
                "stack has the required MLNX_OFED/DOCA GDS patches, or use compat mode. "
                "For upstream PCI P2PDMA on x86 RAID0, upgrade to Linux kernel >= 7.1 "
                "or test a non-RAID local NVMe or NVMe-oF route."
            ),
        ))

    config_issues = []
    if normalized_fs == "raid0":
        prop_val = _cufile_prop(output, "properties.use_pci_p2pdma")
        block_val = _cufile_prop(output, "block.raid.use_pci_p2pdma")
        failing = []
        if prop_val != "true":
            failing.append(f"properties.use_pci_p2pdma = {prop_val or 'not set'}")
        if block_val != "true":
            failing.append(f"block.raid.use_pci_p2pdma = {block_val or 'not set'}")
        if failing:
            config_issues.append((
                f"P2PDMA/C2C disabled in /etc/cufile.json: {', '.join(failing)}",
                (
                    "Both settings must be true in /etc/cufile.json:\n"
                    '  "properties": { "use_pci_p2pdma": true }\n'
                    '  "block": { "raid": { "use_pci_p2pdma": true } }\n'
                    "On x86 these keys enable upstream PCI P2PDMA; on supported "
                    "GH/GB ARM platforms they enable the C2C path when the "
                    "route is supported."
                ),
            ))
    elif p2pdma_cap == "config":
        # Supported fs-specific P2PDMA route (virtiofs): check both the
        # general properties key and the fs-specific key — gdscheck only
        # reports the route active when both are true.
        fs_key = P2PDMA_CONFIG_KEY.get(normalized_fs, f"fs.{normalized_fs}.use_pci_p2pdma")
        prop_val = _cufile_prop(output, "properties.use_pci_p2pdma")
        fs_val = _cufile_prop(output, fs_key)
        failing = []
        if prop_val != "true":
            failing.append(f"properties.use_pci_p2pdma = {prop_val or 'not set'}")
        if fs_val != "true":
            failing.append(f"{fs_key} = {fs_val or 'not set'}")
        if failing:
            config_issues.append((
                f"P2PDMA/C2C disabled in /etc/cufile.json: {', '.join(failing)}",
                (
                    "Both settings must be true in /etc/cufile.json:\n"
                    '  "properties": { "use_pci_p2pdma": true }\n'
                    f'  "fs": {{ "{normalized_fs}": {{ "use_pci_p2pdma": true }} }}'
                    "\nOn x86 these keys enable upstream PCI P2PDMA; on supported "
                    "GH/GB ARM platforms they enable the C2C path when the "
                    "route is supported."
                ),
            ))
    else:
        # Local NVMe FS: check global properties + block transport key
        prop_val  = _cufile_prop(output, "properties.use_pci_p2pdma")
        if fs_type == "nvme-of":
            block_key = "block.nvmeof.use_pci_p2pdma"
        elif fs_type == "raid0":
            block_key = "block.raid.use_pci_p2pdma"
        else:
            block_key = "block.nvme.use_pci_p2pdma"
        block_val = _cufile_prop(output, block_key)

        failing = []
        if prop_val != "true":
            failing.append(f"properties.use_pci_p2pdma = {prop_val or 'not set'}")
        if block_val != "true":
            failing.append(f"{block_key} = {block_val or 'not set'}")

        if failing:
            config_issues.append((
                f"P2PDMA/C2C disabled in /etc/cufile.json: {', '.join(failing)}",
                (
                    "Settings must be true in /etc/cufile.json:\n"
                    '  "properties": { "use_pci_p2pdma": true }\n'
                    f'  "block": {{ "{block_key.split(".")[1]}": {{ "use_pci_p2pdma": true }} }}\n'
                    "On x86 these keys enable upstream PCI P2PDMA; on supported "
                    "GH/GB ARM platforms they enable the C2C path when the "
                    "route is supported."
                ),
            ))

    if config_issues:
        blockers.extend(config_issues)
        # cufile.json is the primary fix, but also surface hardware blockers that will need
        # attention after the config is fixed — so the user can plan for both at once.
        hw_prefix = "Also (after fixing cufile.json): "
    else:
        hw_prefix = ""

    # --- Hardware blockers from gdscheck PLATFORM INFO ---
    for line in _gdscheck_section(output, "PLATFORM INFO"):
        stripped = line.strip()
        if not stripped:
            continue
        ll = stripped.lower()
        if "acs enabled" in ll:
            bdf_match = re.search(r"([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", stripped, re.I)
            bdf = bdf_match.group(1) if bdf_match else "<switch_bdf>"
            blockers.append((
                f"{hw_prefix}{stripped}",
                (
                    "Option 1 (persistent, requires reboot): Add pci=noacs to GRUB_CMDLINE_LINUX in /etc/default/grub, then update-grub + reboot\n"
                    "Option 2: Disable ACS in BIOS/UEFI for the PCIe switch\n"
                    f"Option 3 (immediate, non-persistent): sudo setpci -s {bdf} ECAP_ACS+6.w=0"
                ),
            ))
        elif (
            "iommu" in ll
            and "pass" not in ll          # skip "Pass-through" / "passthrough" (already good)
            and "disabled" not in ll      # skip "disabled" (already off)
            and not stripped.startswith("WARN")  # gdscheck warnings are advisory, not blockers
        ):
            blockers.append((
                f"{hw_prefix}{stripped}",
                "Add iommu=pt to kernel cmdline: intel_iommu=on iommu=pt or amd_iommu=on iommu=pt",
            ))

    return blockers


def _filter_p2pdma_blockers_against_static_checks(
    blockers: list[tuple[str, str]],
    report: ModeReport,
) -> list[tuple[str, str]]:
    acs_passed = any(
        result.check == "PCIe ACS redirect" and result.status == Status.PASS
        for result in report.results
    )
    iommu_passed = any(
        result.check == "IOMMU mode" and result.status == Status.PASS
        for result in report.results
    )
    filtered: list[tuple[str, str]] = []
    for why, mitigation in blockers:
        why_lower = why.lower()
        if acs_passed and "acs enabled" in why_lower:
            continue
        if iommu_passed and "iommu" in why_lower:
            continue
        filtered.append((why, mitigation))
    return filtered


def _build_mode_support(reports: list[ModeReport], fs_type: str, path: str = "") -> str:
    """
    Per-mode status section: shows each mode as Supported / Not supported / N/A
    and, for unsupported modes, explains why and what to do.
    Replaces the old single-paragraph Recommendation block.
    """
    nvme_backed = _infer_nvme_backed(path, fs_type)
    gds_output  = _run_gdscheck_raw()
    parse_fs_type = fs_type
    unsupported_raid_level = None
    if path and fs_type in _LOCAL_BLOCK_FS:
        raid_level = get_raid_level(path)
        transport = get_nvme_transport(path)
        if raid_level == "raid0":
            parse_fs_type = "raid0"
        elif raid_level:
            unsupported_raid_level = raid_level
            parse_fs_type = raid_level
        elif transport and transport != "pcie":
            parse_fs_type = "nvme-of"
    gds         = _gdscheck_parse(gds_output, parse_fs_type, nvme_backed) if gds_output else {}
    driver_keys = _FS_DRIVER_KEYS.get(parse_fs_type)
    if driver_keys is None and nvme_backed:
        driver_keys = ("NVMe",)
    driver_label = " or ".join(driver_keys) if driver_keys else (fs_type or "filesystem")

    def _mode_supported(mode: GDSMode, report: ModeReport) -> Optional[bool]:
        """
        True=supported, False=not supported, None=cannot determine.

        None has two sources:
          - report.applicable is False  → mode is N/A for this FS type
          - gds[mode] is None           → gdscheck ran but FS client not active
          - all static checks are WARNs → no positive evidence either way
        Caller distinguishes these via report.applicable.
        """
        if not report.applicable:
            return None
        if unsupported_raid_level and mode in (GDSMode.NATIVE, GDSMode.P2PDMA):
            return False
        if report.status == Status.FAIL:
            return False
        if mode in gds:
            return gds[mode]  # True, False, or None (client not active)
        if report.status == Status.PASS:
            return True
        # WARN with no gdscheck verdict: only claim Supported if ≥1 check actually passed
        if any(r.status == Status.PASS for r in report.results):
            return True
        # All WARNs, no PASSes, no gdscheck verdict — cannot verify
        return None

    lines: list[str] = []

    for report in reports:
        supported = _mode_supported(report.mode, report)
        mode_label = f"{report.mode.value}"
        _start = len(lines)

        if supported is None:
            if not report.applicable:
                pass  # N/A reason shown in Detailed Findings box
            else:
                # Mode is applicable but gdscheck has no evidence (FS client not active)
                lines.append(f"{dim('?')}  {mode_label:<40} {dim('Cannot verify')}")
                lines.append(f"   {dim('No active ' + driver_label + ' client found by gdscheck.')}")
                lines.append(f"   {dim('Mount the filesystem and re-run to get an authoritative verdict.')}")

        elif supported:
            pass  # Mode Availability table already shows ✓ for this mode

        else:
            lines.append(f"{dim('-')}  {mode_label:<40} {dim('Not active')}")

            alternate_direct = False
            if (
                not unsupported_raid_level
                and parse_fs_type != "raid0"
                and fs_type in {"ext4", "xfs", "nvme-of"}
            ):
                # Only claim the other route is available if its own static
                # checks did not already fail (gdscheck can't see those).
                if (
                    report.mode == GDSMode.P2PDMA
                    and gds.get(GDSMode.NATIVE) is True
                    and not _has_static_fail(_report_for(reports, GDSMode.NATIVE))
                ):
                    lines.append(
                        f"   {green('OK:')} Direct GDS is still available via nvidia-fs/nvfs. "
                        "This is valid for NVMe/NVMe-oF stacks with the required MLNX_OFED/DOCA GDS patches."
                    )
                    alternate_direct = True
                elif (
                    report.mode == GDSMode.NATIVE
                    and gds.get(GDSMode.P2PDMA) is True
                    and not _has_static_fail(_report_for(reports, GDSMode.P2PDMA))
                ):
                    lines.append(
                        f"   {green('OK:')} Direct GDS is available via P2PDMA/C2C. "
                        "The direct P2P path takes precedence when both P2PDMA/C2C and nvidia-fs are available."
                    )
                    alternate_direct = True

            # Collect why + mitigations from static check FAILs
            blockers = report.blockers()

            # For P2PDMA: pull authoritative blockers (cufile.json first, then hardware)
            hw_shown = False
            if alternate_direct:
                hw_shown = True
            elif report.mode == GDSMode.NATIVE and gds_output:
                for _, why, mitigation in _gdscheck_native_inactive_diagnosis(
                    gds_output,
                    parse_fs_type,
                    nvme_backed,
                ):
                    lines.append(f"   {yellow('Why:')} {why}")
                    for m_line in mitigation.splitlines():
                        lines.append(f"   {yellow('→')}    {m_line}")
                    hw_shown = True
            elif report.mode == GDSMode.P2PDMA and gds_output:
                p2p_blockers = _filter_p2pdma_blockers_against_static_checks(
                    _gdscheck_p2pdma_blockers(gds_output, parse_fs_type),
                    report,
                )
                for why, mitigation in p2p_blockers:
                    lines.append(f"   {yellow('Why:')} {why}")
                    for m_line in mitigation.splitlines():
                        lines.append(f"   {yellow('→')}    {m_line}")
                    hw_shown = True

            # For RDMA: parse gdscheck RDMA sub-items for specific diagnosis
            elif report.mode == GDSMode.RDMA and gds_output:
                rdma_issues = _gdscheck_rdma_diagnosis(gds_output, fs_type)
                for why, mitigation in rdma_issues:
                    lines.append(f"   {yellow('Why:')} {why}")
                    for m_line in mitigation.splitlines():
                        lines.append(f"   {yellow('→')}    {m_line}")
                    hw_shown = True

            # Show static check FAILs — but skip for P2PDMA/RDMA when gdscheck
            # already provided authoritative blockers (avoids repeating the same info).
            if not (hw_shown and report.mode in (GDSMode.P2PDMA, GDSMode.RDMA)):
                for result in blockers:
                    lines.append(f"   {yellow('Why:')} {textwrap.fill(result.why, width=62, subsequent_indent='        ')}")
                    if result.mitigation:
                        for m_line in result.mitigation.splitlines():
                            lines.append(f"   {yellow('→')}    {m_line}")

            if not hw_shown and not blockers:
                if report.mode == GDSMode.P2PDMA:
                    # P2PDMA absent from gdscheck but no specific blocker detected
                    lines.append(f"   {yellow('Why:')} P2PDMA not available — check PCIe topology, ACS, and IOMMU")
                    lines.append(f"   {yellow('→')}    sudo lspci -vvv | grep -A5 ACSCtl  # look for ReqRedir+")
                    lines.append(f"   {yellow('→')}    cat /proc/cmdline  # check iommu= settings")
                elif report.mode == GDSMode.RDMA:
                    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES as _FSAL
                    _rdma_type = FS_CAPABILITIES.get(_FSAL.get(fs_type, fs_type), {}).get("rdma_type", "userspace")
                    if _rdma_type == "kernel":
                        lines.append(f"   {yellow('Why:')} Kernel RDMA not available — check MLNX_OFED and nvidia-fs")
                        if fs_type == "nfs":
                            lines.append(f"   {yellow('→')}    Mount with: -o rdma,port=20049")
                        lines.append(f"   {yellow('→')}    ofed_info -s             # verify MLNX_OFED installed")
                        lines.append(f"   {yellow('→')}    ibv_devinfo              # check IB port state")
                        lines.append(f"   {yellow('→')}    lsmod | grep nvidia_fs   # nvidia-fs must be loaded for the kernel RDMA path")
                    else:
                        lines.append(f"   {yellow('Why:')} RDMA not available — run: gdscheck -p | grep -A10 'Userspace RDMA'")

        if len(lines) > _start:
            lines.append("")

    return "\n".join(lines).rstrip()


def _gdscheck_parse(output: str, fs_type: str, nvme_backed: bool) -> dict[GDSMode, Optional[bool]]:
    """
    Parse gdscheck -p output into {GDSMode: supported} for this filesystem.

    Values:
      True  — gdscheck confirms the mode is active
      False — gdscheck confirms the mode is not active
      None  — gdscheck ran but found no line for this FS driver
              (filesystem client not active on this host — cannot verify)
    """
    if not output:
        return {}

    verdict: dict[GDSMode, Optional[bool]] = {}

    driver_keys = _FS_DRIVER_KEYS.get(fs_type)
    if driver_keys is None and nvme_backed:
        driver_keys = ("NVMe",)

    if driver_keys:
        modes = _driver_config_modes_any(output, driver_keys)
        if modes is not None:
            # Native GDS tokens vary by FS type:
            #   nvfs           — traditional nvidia-fs path (NVMe, Lustre, BeeGFS)
            #   dmabuf         — DmaBuf-based direct path (GPFS, WekaFS)
            #   nvidia_peermem — PeerDirect-based direct path (GPFS, WekaFS)
            # Older gdscheck versions report "Supported" instead of these
            # tokens; that is also an affirmative direct GDS verdict.
            # Any one present means Native GDS is active for this FS.
            modes_lower = modes.lower()
            userspace_direct = "dmabuf" in modes_lower or "nvidia_peermem" in modes_lower
            verdict[GDSMode.NATIVE] = driver_config_has_native(modes)
            verdict[GDSMode.P2PDMA] = driver_config_has_direct_p2pdma_token(modes)
            if fs_type == "raid0" and not _raid0_p2pdma_supported_by_arch_or_kernel():
                verdict[GDSMode.P2PDMA] = False
            from .fs_matrix import FS_CAPABILITIES as _FS_CAPS, FS_ALIASES as _FS_ALIASES
            _driver_caps = _FS_CAPS.get(_FS_ALIASES.get(fs_type, fs_type), {})
            if _driver_caps.get("rdma_type") == "userspace" and userspace_direct:
                verdict[GDSMode.RDMA] = True
        else:
            # gdscheck ran but no line for this FS driver — client not loaded/active
            verdict[GDSMode.NATIVE] = None

    # "Userspace RDMA" in gdscheck is only relevant for GPFS and WekaFS.
    # Lustre uses kernel RDMA via LNet (nvidia-fs path) — its RDMA verdict
    # comes from static checks, not from the Userspace RDMA gdscheck section.
    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES as _ALIASES
    _caps = FS_CAPABILITIES.get(_ALIASES.get(fs_type, fs_type), {})
    if _caps.get("rdma_type") == "userspace":
        rdma_modes = _driver_config_modes(output, "Userspace RDMA")
        if rdma_modes is not None and verdict.get(GDSMode.RDMA) is not True:
            if "unsupported" in rdma_modes.lower():
                verdict[GDSMode.RDMA] = False
            else:
                # "Userspace RDMA : Supported" — check if either PeerDirect or DmaBuf
                # is available. Either path is sufficient; both go through libcufile_rdma.
                peerdirect = _driver_config_modes(output, "--Mellanox PeerDirect")
                dmabuf     = _driver_config_modes(output, "--DmaBuf support")
                peerdirect_ok = peerdirect is not None and "enabled" in peerdirect.lower()
                dmabuf_ok     = dmabuf     is not None and "enabled" in dmabuf.lower()
                verdict[GDSMode.RDMA] = peerdirect_ok or dmabuf_ok

    return verdict


def render_json_report(path: str, fs_type: str, reports: list[ModeReport]) -> str:
    out = {
        "path": path,
        "filesystem": fs_type,
        "modes": [],
    }
    for report in reports:
        mode_entry = {
            "mode": report.mode.value,
            "applicable": report.applicable,
            "status": report.status.value,
            "checks": [
                {
                    "check": r.check,
                    "status": r.status.value,
                    "exempted": r.exempted,
                    "why": r.why,
                    "mitigation": r.mitigation,
                    "evidence": r.evidence,
                }
                for r in report.results
            ],
        }
        out["modes"].append(mode_entry)
    return json.dumps(out, indent=2)


def _mark_alternate_path_exemptions(reports: list[ModeReport], fs_type: str) -> None:
    """
    Mark CheckResult.exempted=True on FAILs that are not actually blocking
    because a confirmed alternate direct GDS path is working.

    For NVMe-backed local filesystems and NVMe-oF, there are two direct GDS
    paths: upstream PCI P2PDMA and the nvidia-fs nvfs path (when the NVMe stack
    has the required MLNX_OFED/DOCA GDS patches). A failure in one path is not
    a blocking failure if the other path is confirmed clean/active. This is
    the single source of truth for that exemption — both has_blocking_failures()
    (exit code) and the report renderers (table/box/mitigation plan, via
    ModeReport.status) read the .exempted flag it sets, instead of each
    re-deriving the rule independently.

    Idempotent — safe to call more than once on the same reports.
    """
    report_by_mode = {r.mode: r for r in reports}
    alt_path_fs = {"ext4", "xfs", "nvme-of"}

    native = report_by_mode.get(GDSMode.NATIVE)
    p2pdma = report_by_mode.get(GDSMode.P2PDMA)

    def _has_active_p2pdma_signal(report: Optional[ModeReport]) -> bool:
        if report is None:
            return False
        return any(
            r.check == "gdscheck nvidia-fs/nvfs route"
            and "P2PDMA/C2C is active" in r.why
            for r in report.results
        )

    def _has_active_native_signal(report: Optional[ModeReport]) -> bool:
        # Set by _enrich_with_gdscheck() only when gdscheck confirms the
        # native nvidia-fs/nvfs route is the active one for this mount.
        if report is None:
            return False
        return any(
            r.check == "gdscheck P2PDMA/C2C route"
            and "Direct GDS is still available via nvidia-fs/nvfs" in r.why
            for r in report.results
        )

    def _clean_direct_report(report: Optional[ModeReport]) -> bool:
        return report is not None and report.applicable and report.status == Status.PASS

    native_failures_exempted_by_p2pdma = {
        "gdscheck nvidia-fs/nvfs route",
        "nvidia-fs module",
    }
    p2pdma_failures_exempted_by_native = {
        "P2PDMA kernel support",
        "CONFIG_PCI_P2PDMA",
        "PCIe ACS redirect",
        "NVIDIA P2PDMA driver registries",
        "nvidia-fs kernel log",
        "nvidia-fs kernel log messages",
        "cufile.json P2PDMA settings",
        "P2PDMA config key",
        "NVMe transport",
        "NVMe backing device",
        "RAID0 P2PDMA architecture",
    }

    direct_alt_fs = fs_type in alt_path_fs
    native_direct_available = (
        direct_alt_fs
        and _clean_direct_report(native)
        and _has_active_native_signal(p2pdma)
    )
    p2pdma_direct_available = direct_alt_fs and _has_active_p2pdma_signal(native)

    for report in reports:
        if not report.applicable:
            continue
        for result in report.results:
            if result.status != Status.FAIL:
                continue
            if result.mode == GDSMode.NATIVE and p2pdma_direct_available:
                result.exempted = result.check in native_failures_exempted_by_p2pdma
            elif result.mode == GDSMode.P2PDMA and native_direct_available:
                result.exempted = result.check in p2pdma_failures_exempted_by_native


def has_blocking_failures(reports: list[ModeReport], fs_type: str) -> bool:
    """
    Return whether the mount check should exit non-zero.

    Only non-exempted FAILs block — see _mark_alternate_path_exemptions().
    """
    _mark_alternate_path_exemptions(reports, fs_type)
    return any(
        r.status == Status.FAIL and not r.exempted
        for report in reports
        if report.applicable
        for r in report.results
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _check_cuda_toolkit() -> Optional[str]:
    """
    Verify that CUDA toolkit is installed and return the detected path.
    Returns the cuda root path if found, or None if not installed.

    Checks (in order):
      1. /usr/local/cuda symlink (standard install location)
      2. /usr/local/cuda-* versioned directories
      3. nvcc on PATH
      4. nvidia-smi to confirm GPU driver is present at all
    """
    import glob as _glob

    # Check standard symlink / directory
    cuda_candidates = ["/usr/local/cuda"] + sorted(_glob.glob("/usr/local/cuda-*"))
    for path in cuda_candidates:
        if os.path.isdir(path):
            return path

    # Check nvcc on PATH
    try:
        result = subprocess.run(
            ["which", "nvcc"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.dirname(os.path.dirname(result.stdout.strip()))
    except Exception:
        pass

    return None


def _print_cuda_install_guide() -> None:
    """Print actionable instructions for installing CUDA toolkit."""
    print(bold("\n  CUDA Toolkit Not Found"))
    print("  " + "─" * 66)
    print(textwrap.fill(
        "GDS requires the CUDA toolkit (cuda-toolkit package). "
        "gdscheck, nvidia_fs, and cufile.json are all part of this package.",
        width=68, initial_indent="  ", subsequent_indent="  ",
    ))
    print()
    print(bold("  Install CUDA toolkit:"))
    print()
    print("  Ubuntu / Debian:")
    print("    # Add NVIDIA package repo if not already present:")
    print("    wget https://developer.download.nvidia.com/compute/cuda/repos/")
    print("          ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb")
    print("    sudo dpkg -i cuda-keyring_1.1-1_all.deb")
    print("    sudo apt-get update")
    print("    sudo apt-get install -y cuda-toolkit")
    print()
    print("  RHEL / Rocky / CentOS:")
    print("    sudo dnf config-manager --add-repo \\")
    print("      https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo")
    print("    sudo dnf install -y cuda-toolkit")
    print()
    print("  Or download the installer from:")
    print("    https://developer.nvidia.com/cuda-downloads")
    print()
    print("  After installing, verify:")
    print("    ls /usr/local/cuda/gds/tools/gdscheck")
    print("    nvcc --version")
    print()
