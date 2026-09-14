# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IOMMU state checks for GDS.

IOMMU strict mode is a warning, not a hard blocker — GDS/P2PDMA may still
work on some hardware/kernel combinations, but performance or reliability
can be affected. Passthrough (iommu=pt on x86, iommu.passthrough=1 on
aarch64) and disabled are both fully supported.

Architecture-aware: parses arch-appropriate kernel cmdline tokens and emits
arch-appropriate mitigation advice. NVIDIA Grace platforms get special
treatment because GPU↔CPU-memory traffic uses NVLink-C2C, which bypasses
the SMMU entirely.

Source: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
"""
from __future__ import annotations

import os
import platform
import re
from typing import Optional

from . import kmsg
from .result import CheckResult, GDSMode, Status


class Arch:
    X86 = "x86_64"
    ARM = "aarch64"
    OTHER = "other"


class CpuVendor:
    INTEL = "intel"
    AMD = "amd"
    UNKNOWN = "unknown"


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):
        return Arch.ARM
    if m == "x86_64":
        return Arch.X86
    return Arch.OTHER


def _read_first_line(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError):
        return ""


def _cpu_vendor() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if not line.lower().startswith("vendor_id"):
                    continue
                _, _, value = line.partition(":")
                vendor = value.strip().lower()
                if vendor == "genuineintel":
                    return CpuVendor.INTEL
                if vendor == "authenticamd":
                    return CpuVendor.AMD
                return CpuVendor.UNKNOWN
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return CpuVendor.UNKNOWN


def _has_local_nvme_devices() -> bool:
    from . import pcie

    return bool(pcie.get_nvme_bdfs())


def _has_neoverse_v2() -> bool:
    """Neoverse V2 (CPU part 0xd4f) is the core used in NVIDIA Grace.
    Also used in AWS Graviton4, but Graviton instances do not ship with
    discrete NVIDIA GPUs, so combined with _has_nvidia_pcie_device() this
    is essentially a Grace marker."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("cpu part") and "0xd4f" in line.lower():
                    return True
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return False


def _has_nvidia_pcie_device() -> bool:
    """Look for any PCIe device with NVIDIA vendor ID (0x10de)."""
    try:
        for entry in os.listdir("/sys/bus/pci/devices"):
            vendor = _read_first_line(f"/sys/bus/pci/devices/{entry}/vendor")
            if vendor.lower() == "0x10de":
                return True
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return False


def _is_grace() -> bool:
    """Best-effort detection of an NVIDIA Grace platform.

    Tries, in order:
      1. DMI vendor/product strings (works when the OEM advertises NVIDIA/Grace).
      2. /proc/device-tree/compatible (DT-booted systems only).
      3. Heuristic: aarch64 + Neoverse V2 core + at least one NVIDIA PCIe
         device. Catches OEM Grace boards (QCT, Supermicro, etc.) where
         neither DMI nor device-tree mention Grace.

    Returns False on non-aarch64 hosts.
    """
    if _arch() != Arch.ARM:
        return False
    for f in (
        "/sys/class/dmi/id/sys_vendor",
        "/sys/class/dmi/id/product_family",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/board_vendor",
        "/sys/class/dmi/id/board_name",
    ):
        s = _read_first_line(f).lower()
        if "nvidia" in s or "grace" in s:
            return True
    dt = _read_first_line("/proc/device-tree/compatible").lower()
    if "nvidia" in dt or "grace" in dt:
        return True
    if _has_neoverse_v2() and _has_nvidia_pcie_device():
        return True
    return False


# Public aliases for cross-module use (pre-install, post-install, mount-check).
def arch() -> str:
    return _arch()


def is_grace() -> bool:
    return _is_grace()


def _read_cmdline() -> str:
    try:
        with open("/proc/cmdline") as fh:
            return fh.read().strip()
    except FileNotFoundError:
        return ""


def _kmsg_iommu_lines() -> kmsg.KernelLogResult:
    return kmsg.grep_kmsg_result(r"iommu|smmu")


def _iommu_sysfs_active() -> bool:
    """Return True if IOMMU domains are active (iommu/ entries exist in sysfs)."""
    try:
        entries = os.listdir("/sys/class/iommu")
        return len(entries) > 0
    except FileNotFoundError:
        return False


class IommuState:
    DISABLED    = "disabled"
    PASSTHROUGH = "passthrough"
    STRICT      = "strict"
    UNKNOWN     = "unknown"


def detect_iommu_state() -> tuple[str, str]:
    """
    Returns (state: IommuState, evidence: str).
    State is one of: disabled, passthrough, strict, unknown.

    Arch-aware: parses Intel/AMD tokens on x86_64 and iommu.passthrough on
    aarch64. The DISABLED state is x86-only — there's no equivalent cmdline
    knob to turn the ARM SMMU fully off the way intel_iommu=off does.
    """
    cmdline = _read_cmdline()
    evidence_lines = [f"cmdline: {cmdline}"]
    arch = _arch()

    if arch == Arch.ARM:
        # aarch64 / ARM SBSA. The kernel SMMU is controlled by:
        #   iommu.passthrough=1   → bypass DMA translation (≈ x86 iommu=pt)
        #   iommu.strict=0|1      → controls TLB flushing strategy, not P2P
        if re.search(r"iommu\.passthrough\s*=\s*1", cmdline):
            return IommuState.PASSTHROUGH, " | ".join(evidence_lines)
        # Anything else: fall through to sysfs / dmesg detection below.

    else:
        # x86_64 (and as a best-effort default for anything not aarch64).
        intel_off = bool(re.search(r"intel_iommu\s*=\s*off", cmdline))
        amd_off   = bool(re.search(r"amd_iommu\s*=\s*off",   cmdline))
        iommu_off = bool(re.search(r"\biommu\s*=\s*off\b",   cmdline))

        intel_on  = bool(re.search(r"intel_iommu\s*=\s*on",  cmdline))
        amd_on    = bool(re.search(r"amd_iommu\s*=\s*on",    cmdline))
        iommu_pt  = bool(re.search(r"\biommu\s*=\s*pt\b",    cmdline))

        if intel_off or amd_off or iommu_off:
            return IommuState.DISABLED, " | ".join(evidence_lines)

        if iommu_pt:
            return IommuState.PASSTHROUGH, " | ".join(evidence_lines)

        # IOMMU is explicitly on without passthrough → strict
        if intel_on or amd_on:
            return IommuState.STRICT, " | ".join(evidence_lines)

    # No conclusive cmdline flags: check sysfs to see if IOMMU is actually active
    if _iommu_sysfs_active():
        evidence_lines.append("sysfs /sys/class/iommu has active entries")
        return IommuState.STRICT, " | ".join(evidence_lines)

    # Check kernel log (journalctl preferred, dmesg fallback) for confirmation
    log_result = _kmsg_iommu_lines()
    if log_result.lines:
        evidence_lines.extend(log_result.lines[:5])
        combined = " ".join(log_result.lines).lower()
        if "passthrough" in combined:
            return IommuState.PASSTHROUGH, " | ".join(evidence_lines)
        if "enabled" in combined or "using" in combined:
            return IommuState.STRICT, " | ".join(evidence_lines)
    elif not log_result.available and log_result.error:
        evidence_lines.append(f"kernel log unavailable: {log_result.error}")

    return IommuState.UNKNOWN, " | ".join(evidence_lines)


# ---------------------------------------------------------------------------
# Mitigation strings (arch-aware)
# ---------------------------------------------------------------------------

TROUBLESHOOTING_URL = (
    "https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/"
    "index.html#before-you-install-gds"
)


def _x86_strict_mitigation() -> str:
    vendor = _cpu_vendor()
    if vendor == CpuVendor.INTEL:
        body = (
            "  Recommended (preserves interrupt remapping):\n"
            "    intel_iommu=on iommu=pt\n"
            "  Alternative (disables IOMMU entirely):\n"
            "    intel_iommu=off\n"
        )
    elif vendor == CpuVendor.AMD:
        body = (
            "  Recommended (preserves interrupt remapping):\n"
            "    amd_iommu=on iommu=pt\n"
            "  Alternative (disables IOMMU entirely):\n"
            "    amd_iommu=off\n"
        )
    else:
        body = (
            "  CPU vendor could not be detected; use the line matching this host:\n"
            "  Recommended (preserves interrupt remapping):\n"
            "    Intel: intel_iommu=on iommu=pt\n"
            "    AMD:   amd_iommu=on iommu=pt\n"
            "  Alternative (disables IOMMU entirely):\n"
            "    Intel: intel_iommu=off\n"
            "    AMD:   amd_iommu=off\n"
        )
    return (
        "Add the following to GRUB_CMDLINE_LINUX in /etc/default/grub:\n"
        f"{body}"
        "Then run:\n"
        "  sudo update-grub              # Ubuntu/Debian\n"
        "  sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/Rocky\n"
        "  sudo reboot\n"
        "Verify after reboot: cat /proc/cmdline"
    )


def _strict_mitigation(arch: str) -> str:
    if arch == Arch.ARM:
        body = (
            "Add the following to GRUB_CMDLINE_LINUX in /etc/default/grub:\n"
            "    iommu.passthrough=1\n"
            "Then run:\n"
            "  sudo update-grub              # Ubuntu/Debian\n"
            "  sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # RHEL/Rocky\n"
            "  sudo reboot\n"
            "Verify after reboot: cat /proc/cmdline"
        )
    else:
        body = _x86_strict_mitigation()
    return f"{body}\nReference: {TROUBLESHOOTING_URL}"


def _unknown_mitigation(arch: str) -> str:
    tail = (
        "GDS works best with iommu.passthrough=1 (aarch64)."
        if arch == Arch.ARM
        else "GDS requires iommu=pt (passthrough) or IOMMU disabled. Strict mode may block P2PDMA."
    )
    return (
        "Manually verify IOMMU state:\n"
        "  cat /proc/cmdline\n"
        "  sudo journalctl -k -b | grep -iE 'iommu|smmu'\n"
        "  sudo dmesg | grep -iE 'iommu|smmu'\n"
        "  ls /sys/class/iommu/\n"
        f"{tail}\n\n"
        "If verification shows IOMMU active without passthrough (strict mode),\n"
        "apply this fix:\n"
        f"{_strict_mitigation(arch)}"
    )


# ---------------------------------------------------------------------------
# Check function (arch-aware, single source of truth)
# ---------------------------------------------------------------------------

def _check_iommu(mode: GDSMode) -> CheckResult:
    state, evidence = detect_iommu_state()
    arch = _arch()
    grace = _is_grace()

    if state == IommuState.DISABLED:
        return CheckResult(
            check="IOMMU mode", mode=mode, status=Status.PASS,
            why="IOMMU is disabled. Direct GPU DMA is unrestricted.",
            evidence=evidence,
        )

    if state == IommuState.PASSTHROUGH:
        pt_token = "iommu.passthrough=1" if arch == Arch.ARM else "iommu=pt"
        return CheckResult(
            check="IOMMU mode", mode=mode, status=Status.PASS,
            why=f"IOMMU is in passthrough mode ({pt_token}). GDS supports this configuration.",
            evidence=evidence,
        )

    if state == IommuState.STRICT:
        if grace:
            # NVLink-C2C bypasses the SMMU for GPU↔CPU memory traffic, so the
            # x86 P2PDMA-blocking story doesn't apply on Grace. PCIe DMA from
            # storage still flows through SMMU but Grace's typical I/O path is
            # NVMe→CPU memory (PCIe)→GPU (NVLink-C2C), not GPU↔NVMe P2P.
            return CheckResult(
                check="IOMMU mode", mode=mode, status=Status.PASS,
                why=(
                    "NVIDIA Grace detected. SMMU is in default (strict) mode, but "
                    "GPU↔CPU-memory traffic uses NVLink-C2C and bypasses the SMMU. "
                    "The x86-style 'strict mode breaks P2PDMA' rule does not apply here."
                ),
                evidence=evidence + " | grace=detected",
            )
        if mode != GDSMode.NATIVE and not _has_local_nvme_devices():
            return CheckResult(
                check="IOMMU mode", mode=mode, status=Status.PASS,
                why=(
                    "IOMMU is in strict mode, but no local NVMe PCI devices were "
                    "found. The GPU↔NVMe P2PDMA IOMMU warning is not applicable "
                    "on this host."
                ),
                evidence=evidence + " | local_nvme_bdfs=none",
            )
        target = "Native GDS" if mode == GDSMode.NATIVE else "Direct GPU↔NVMe DMA"
        return CheckResult(
            check="IOMMU mode", mode=mode, status=Status.WARN,
            why=(
                f"IOMMU is in strict mode. {target} may be restricted depending on "
                f"hardware and kernel version. Consider switching to passthrough for "
                f"guaranteed {'native GDS' if mode == GDSMode.NATIVE else 'P2PDMA'} "
                f"compatibility."
            ),
            mitigation=_strict_mitigation(arch),
            evidence=evidence,
        )

    # Unknown
    if grace:
        # Same reasoning as the STRICT+Grace case above: GPU↔CPU-memory traffic
        # uses NVLink-C2C and bypasses the SMMU regardless of its mode, so an
        # undetermined SMMU state isn't a P2PDMA concern on Grace.
        return CheckResult(
            check="IOMMU mode", mode=mode, status=Status.PASS,
            why=(
                "NVIDIA Grace detected. SMMU state could not be determined, but "
                "GPU↔CPU-memory traffic uses NVLink-C2C and bypasses the SMMU "
                "regardless. The x86-style 'strict mode breaks P2PDMA' rule does "
                "not apply here."
            ),
            evidence=evidence + " | grace=detected",
        )
    return CheckResult(
        check="IOMMU mode", mode=mode, status=Status.WARN,
        why=(
            "IOMMU state could not be determined. No IOMMU flags found in /proc/cmdline "
            "and /sys/class/iommu is empty or inaccessible."
        ),
        mitigation=_unknown_mitigation(arch),
        evidence=evidence,
    )


def check_iommu_for_p2pdma() -> CheckResult:
    return _check_iommu(GDSMode.P2PDMA)


def check_iommu_for_native() -> CheckResult:
    return _check_iommu(GDSMode.NATIVE)


def run_all(mode: Optional[GDSMode] = None) -> list[CheckResult]:
    """Single IOMMU check, packaged as a list for symmetry with other check modules.

    The `mode` arg only affects wording (Native GDS vs P2PDMA) in the WARN
    message. Callers running pre-install style checks (mode-agnostic) can
    omit it entirely.
    """
    if mode == GDSMode.NATIVE:
        return [check_iommu_for_native()]
    # Default (None) and any other value → P2PDMA-flavored result. The check
    # itself is arch-aware; the mode arg only colours the WARN message.
    return [check_iommu_for_p2pdma()]
