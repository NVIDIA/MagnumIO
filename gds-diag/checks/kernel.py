# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kernel configuration checks for GDS mode support.

Reads the running kernel's .config from one of:
  /proc/config.gz        (if CONFIG_IKCONFIG_PROC=y)
  /boot/config-<uname>   (most distros)
  /lib/modules/<uname>/config
"""
from __future__ import annotations

import gzip
import os
import platform
import re
import subprocess
from typing import Optional

from .result import CheckResult, GDSMode, Status

# ---------------------------------------------------------------------------
# Minimum kernel versions for each mode
# ---------------------------------------------------------------------------
MIN_KERNEL = {
    "p2pdma": (6, 2),   # kernel 6.2+ required for stable GDS P2PDMA support
    "native": (4, 15),  # earliest nvidia-fs support
}


def _kernel_version() -> tuple[int, int, int]:
    release = platform.release()          # e.g. "5.15.0-91-generic"
    m = re.match(r"(\d+)\.(\d+)\.?(\d*)", release)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return (0, 0, 0)


def _read_kernel_config() -> Optional[dict[str, str]]:
    """
    Read the running kernel's .config. Returns a dict of CONFIG_KEY → value,
    or None if the config file cannot be found.
    """
    uname = platform.release()
    candidates = [
        "/proc/config.gz",
        f"/boot/config-{uname}",
        f"/lib/modules/{uname}/config",
    ]

    raw: Optional[str] = None
    for path in candidates:
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt") as fh:
                    raw = fh.read()
            else:
                with open(path) as fh:
                    raw = fh.read()
            break
        except (FileNotFoundError, PermissionError, OSError):
            continue

    if raw is None:
        return None

    config: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            config[k.strip()] = v.strip()
    return config


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_kernel_version(mode: GDSMode) -> CheckResult:
    major, minor, patch = _kernel_version()
    ver_str = f"{major}.{minor}.{patch}"

    mode_key = {
        GDSMode.NATIVE: "native",
        GDSMode.P2PDMA: "p2pdma",
    }.get(mode)

    if mode_key is None:
        return CheckResult(
            check="Kernel version", mode=mode, status=Status.PASS,
            why=f"No minimum kernel version requirement for {mode.value}.",
        )

    req = MIN_KERNEL[mode_key]
    current = (major, minor)

    if current >= req:
        return CheckResult(
            check="Kernel version", mode=mode, status=Status.PASS,
            why=f"Kernel {ver_str} meets minimum {req[0]}.{req[1]} for {mode.value}.",
        )
    else:
        return CheckResult(
            check="Kernel version", mode=mode, status=Status.FAIL,
            why=(
                f"Kernel {ver_str} is below the minimum {req[0]}.{req[1]} "
                f"required for {mode.value}."
            ),
            mitigation=(
                f"Upgrade to kernel >= {req[0]}.{req[1]}. "
                "On Ubuntu: apt-get install linux-image-generic-hwe-XX.XX. "
                "On RHEL/Rocky: use the latest kernel from BaseOS or ELRepo."
            ),
            evidence=f"uname -r → {platform.release()}",
        )


def check_p2pdma_kallsyms() -> CheckResult:
    """
    Check /proc/kallsyms for PCI P2PDMA support.

    NVIDIA's current GDS troubleshooting guide checks for p2pdma_pgmap_ops.
    Older guidance and some distro kernels expose pci_p2pdma_add_resource
    instead, so accept either symbol. This is more reliable than kernel version
    comparison because it checks the running kernel directly.
    """
    symbols = ("p2pdma_pgmap_ops", "pci_p2pdma_add_resource")
    try:
        with open("/proc/kallsyms", errors="replace") as fh:
            for line in fh:
                if any(sym in line for sym in symbols):
                    sym = next(s for s in symbols if s in line)
                    return CheckResult(
                        check="P2PDMA kernel support",
                        mode=GDSMode.P2PDMA,
                        status=Status.PASS,
                        why=f"{sym} found in /proc/kallsyms — PCI P2PDMA support is present in this kernel.",
                        evidence=line.strip(),
                    )

        kernel_ver = platform.release()
        return CheckResult(
            check="P2PDMA kernel support",
            mode=GDSMode.P2PDMA,
            status=Status.FAIL,
            why=(
                f"No known P2PDMA symbol ({', '.join(symbols)}) was found in /proc/kallsyms. "
                f"PCI P2PDMA is not compiled into this kernel (running {kernel_ver})."
            ),
            mitigation=(
                f"Kernel {kernel_ver} does not have PCI P2PDMA support. Upgrade to a kernel "
                "with CONFIG_PCI_P2PDMA=y, or use the nvidia-fs path with NVMe/NVMe-oF "
                "GDS patches from MLNX_OFED/DOCA when that path is available.\n"
                "Requires reboot after kernel or driver changes."
            ),
            evidence=f"uname -r -> {kernel_ver}; checked symbols: {', '.join(symbols)}",
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        # /proc/kallsyms unavailable — fall back to kernel version comparison
        return check_kernel_version(GDSMode.P2PDMA)


def check_pci_p2pdma_config() -> CheckResult:
    """Check CONFIG_PCI_P2PDMA=y in the running kernel."""
    config = _read_kernel_config()

    if config is None:
        return CheckResult(
            check="CONFIG_PCI_P2PDMA", mode=GDSMode.P2PDMA, status=Status.WARN,
            why=(
                "Cannot read kernel config — /proc/config.gz and /boot/config-$(uname -r) "
                "are both unavailable. CONFIG_PCI_P2PDMA status is unknown."
            ),
            mitigation=(
                "Enable CONFIG_IKCONFIG_PROC in your kernel to expose /proc/config.gz, "
                "or ensure /boot/config-$(uname -r) exists. "
                "Run: zcat /proc/config.gz | grep CONFIG_PCI_P2PDMA"
            ),
        )

    val = config.get("CONFIG_PCI_P2PDMA")
    if val == "y":
        return CheckResult(
            check="CONFIG_PCI_P2PDMA", mode=GDSMode.P2PDMA, status=Status.PASS,
            why="CONFIG_PCI_P2PDMA=y — kernel P2PDMA subsystem compiled in.",
        )
    elif val == "m":
        return CheckResult(
            check="CONFIG_PCI_P2PDMA", mode=GDSMode.P2PDMA, status=Status.WARN,
            why="CONFIG_PCI_P2PDMA=m — built as module. May not be loaded.",
            mitigation="Ensure pci_p2pdma module is loaded: modprobe pci_p2pdma",
        )
    else:
        return CheckResult(
            check="CONFIG_PCI_P2PDMA", mode=GDSMode.P2PDMA, status=Status.FAIL,
            why=(
                "CONFIG_PCI_P2PDMA is not set in the running kernel. "
                "The P2PDMA subsystem is compiled out — no direct NVMe↔GPU DMA is possible."
            ),
            mitigation=(
                "Option 1 (recommended): Switch to a distribution kernel that includes "
                "CONFIG_PCI_P2PDMA=y. Supported distros: Ubuntu 20.04 HWE+, RHEL 8.3+, "
                "SLES 15 SP3+, DGX OS 6+.\n"
                "Option 2: Recompile kernel with CONFIG_PCI_P2PDMA=y (not practical in most environments).\n"
                "Option 3: Use RDMA or compat mode instead of P2PDMA."
            ),
            evidence=f"CONFIG_PCI_P2PDMA={'not set' if val is None else val}",
        )


def check_zone_device_config() -> CheckResult:
    """CONFIG_ZONE_DEVICE is required for device memory mapping used by GDS."""
    config = _read_kernel_config()

    if config is None:
        return CheckResult(
            check="CONFIG_ZONE_DEVICE", mode=GDSMode.NATIVE, status=Status.WARN,
            why="Cannot read kernel config. CONFIG_ZONE_DEVICE status unknown.",
        )

    val = config.get("CONFIG_ZONE_DEVICE")
    if val == "y":
        return CheckResult(
            check="CONFIG_ZONE_DEVICE", mode=GDSMode.NATIVE, status=Status.PASS,
            why="CONFIG_ZONE_DEVICE=y — device memory zone support available.",
        )
    else:
        return CheckResult(
            check="CONFIG_ZONE_DEVICE", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                "CONFIG_ZONE_DEVICE not set. Device memory mapping may be limited. "
                "GDS may fall back to compat mode for some operations."
            ),
            mitigation="Kernel should have CONFIG_ZONE_DEVICE=y. Check distro support matrix.",
            evidence=f"CONFIG_ZONE_DEVICE={'not set' if val is None else val}",
        )


def run_all(mode: GDSMode) -> list[CheckResult]:
    """Run all kernel checks relevant to the given mode."""
    results = []

    if mode == GDSMode.P2PDMA:
        results.append(check_p2pdma_kallsyms())
        results.append(check_pci_p2pdma_config())

    elif mode == GDSMode.NATIVE:
        results.append(check_kernel_version(mode))
        results.append(check_zone_device_config())

    return results
