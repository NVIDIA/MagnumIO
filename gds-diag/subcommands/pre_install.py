# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
pre-install subcommand.

Validate that the system is GDS-capable *before* installing CUDA/GDS.
Checks hardware, kernel configuration, and existing filesystem mounts to
flag anything that would block GDS from working after install.
"""
from __future__ import annotations

import argparse
import json

from ._base import Subcommand

_DESCRIPTION = """\
Pre-install GDS readiness check.

Does NOT require GDS, gdscheck, nvidia_fs, or /etc/cufile.json.

Checks performed:

  System
    - OS: Linux required (GDS is Linux-only)
    - Architecture: x86_64 and aarch64 both supported

  GPU
    - NVIDIA GPU presence via lspci (no driver required)
    - NVIDIA Open Kernel Driver install via modinfo only (no nvidia-smi)
    - On Grace platforms only: CDMM mode detection
      (CoherentGPUMemoryMode from /proc/driver/nvidia/params)
    - nvidia-smi driver version, compute capability, and topology checks are
      intentionally deferred to post-install

  CUDA Toolkit
    - nvcc presence and version; missing toolkit is FAIL

  DOCA / MLNX_OFED
    - Advisory check for MLNX_OFED or DOCA/DOCA-OFED; missing stack is WARN
      because it is required for RDMA and some patched nvfs storage routes, but
      not for every possible GDS route

  Kernel
    - CONFIG_PCI_P2PDMA compiled in (via /proc/kallsyms symbol
      pci_p2pdma_add_resource, not kernel version)
    - Kernel version (informational only)

  IOMMU
    - Arch-aware /proc/cmdline parse:
        x86_64:  intel_iommu=, amd_iommu=, iommu=pt|off
        aarch64: iommu.passthrough=1
    - PASS for passthrough or disabled
    - WARN for strict mode (may restrict P2PDMA on x86)
    - On NVIDIA Grace, strict SMMU mode is downgraded to PASS because
      GPU↔CPU-memory traffic uses NVLink-C2C and bypasses the SMMU

  PCIe / ACS
    - lspci -vvv for ACS P2P Request Redirect on PCIe switches
    - Flags any switch with ACSCtl: SrcValid+ ... ReqRedir+
    - GPU/NVMe topology via nvidia-smi is intentionally deferred to post-install

  Mounted filesystems
    - For each plausible workload storage mount in /proc/mounts: filesystem
      type and GDS mode support from the static matrix
    - Suppresses runtime/system mounts such as /dev/shm and /run/*
    - Flags known issues (e.g. ext4 data=journal disables O_DIRECT) as
      mount-specific warnings; use mount-check for authoritative path checks
    - Probes O_DIRECT support on a temp file for candidate storage mounts
"""

# Kernel pseudo-FSes to skip when enumerating mounts
_PSEUDO_FS = {
    "proc", "sysfs", "devtmpfs", "devpts", "cgroup", "cgroup2",
    "pstore", "debugfs", "securityfs", "configfs", "fusectl",
    "hugetlbfs", "mqueue", "tracefs", "bpf", "autofs", "nsfs",
    "rpc_pipefs", "nfsd", "efivarfs", "binfmt_misc", "selinuxfs",
    "sockfs", "pipefs", "anon_inodefs", "fuse.portal",
    "squashfs", "fuse.snapfuse", "tmpfs", "ramfs",
}

_RUNTIME_MOUNT_PREFIXES = (
    "/dev/shm",
    "/run",
    "/var/run",
    "/sys",
)


def _is_runtime_mount(mountpoint: str) -> bool:
    return any(
        mountpoint == prefix or mountpoint.startswith(prefix.rstrip("/") + "/")
        for prefix in _RUNTIME_MOUNT_PREFIXES
    )


def _cap_label(caps: dict, key: str) -> str:
    if key == "p2pdma" and caps.get("p2pdma_display"):
        return caps["p2pdma_display"]
    value = caps.get(key, True) if key == "compat" else caps.get(key)
    if value is True:
        return "Yes"
    if value == "config":
        return "Config"
    if value is None:
        return "Unknown"
    return "No"


def _collect_mounted_filesystem_results(mount_lines: list[str]) -> list:
    from checks.fs_matrix import (
        FS_ALIASES, get_fs_capabilities, _decode_proc_mounts_field,
        check_ext4_data_mode, check_odirect,
    )
    from checks.result import CheckResult, Status, GDSMode

    mount_results: list[CheckResult] = []
    seen: set[tuple[str, str]] = set()
    for line in mount_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        device, fstype, opts = parts[0], parts[2], parts[3]
        mountpoint = _decode_proc_mounts_field(parts[1])
        fstype_key = FS_ALIASES.get(fstype, fstype).lower()
        if fstype.lower() in _PSEUDO_FS or fstype_key in _PSEUDO_FS:
            continue
        if _is_runtime_mount(mountpoint):
            continue
        key = (device, mountpoint)
        if key in seen:
            continue
        seen.add(key)

        caps = get_fs_capabilities(fstype_key)
        cap_summary = (
            f"Native={_cap_label(caps, 'native')}  "
            f"P2PDMA={_cap_label(caps, 'p2pdma')}  "
            f"RDMA={_cap_label(caps, 'rdma')}  "
            f"Compat={_cap_label(caps, 'compat')}"
        )

        # ext4 GDS requires explicit data=ordered.
        if fstype_key == "ext4":
            data_mode_result = check_ext4_data_mode(mountpoint)
            if data_mode_result:
                mode, evidence = data_mode_result
                if mode != "ordered":
                    if mode == "journal":
                        why = (
                            f"{mountpoint} is mounted with data=journal. "
                            "This disables O_DIRECT — all GDS modes are blocked."
                        )
                    elif mode == "default":
                        why = (
                            f"{mountpoint} is not explicitly mounted with data=ordered. "
                            "GDS requires data=ordered to be visible in the active mount options."
                        )
                    else:
                        why = (
                            f"{mountpoint} is mounted with data={mode}. "
                            "GDS requires ext4 to be explicitly mounted with data=ordered."
                        )
                    if mountpoint == "/":
                        mitigation = (
                            "For the root filesystem, add rootflags=data=ordered to "
                            "GRUB_CMDLINE_LINUX in /etc/default/grub, then run:\n"
                            "  sudo update-grub\n"
                            "  sudo reboot\n"
                            "Verify: findmnt -no OPTIONS / | tr ',' '\\n' | grep '^data=ordered$'"
                        )
                    else:
                        mitigation = (
                            "Remount with explicit data=ordered:\n"
                            f"  sudo mount -o remount,data=ordered {mountpoint}\n"
                            "For persistence, add data=ordered to this filesystem's /etc/fstab options.\n"
                            "See the following for more information once GDS is installed:\n"
                            f"  ./gds-diag.py mount-check {mountpoint} -v"
                        )
                    mount_results.append(CheckResult(
                        check=f"ext4 data mode ({mountpoint})",
                        mode=GDSMode.NATIVE,
                        status=Status.WARN,
                        why=why,
                        mitigation=mitigation,
                        evidence=evidence,
                    ))
                    continue

        # O_DIRECT probe
        ok, ev = check_odirect(mountpoint)
        if ok is None:
            status = Status.INFO
            why = (
                f"{mountpoint} ({fstype_key}): O_DIRECT support could not be verified. {ev}"
            )
        elif ok:
            status = Status.PASS
            why = f"{mountpoint} ({fstype_key}): O_DIRECT supported. {cap_summary}"
        else:
            status = Status.WARN
            why = (
                f"{mountpoint} ({fstype_key}): O_DIRECT NOT supported. "
                "If this mount will be used for GDS workloads, direct native modes are blocked."
            )
        mount_results.append(CheckResult(
            check=f"Mount: {mountpoint}",
            mode=GDSMode.NATIVE,
            status=status,
            why=why,
            evidence=ev,
            mitigation=(
                None if ok or ok is None else
                "Check mount options. For GDS, use ext4 or xfs on a real block device.\n"
                "See the following for more information once GDS is installed:\n"
                f"  ./gds-diag.py mount-check {mountpoint} -v"
            ),
        ))

    return mount_results


def _check_os() -> "CheckResult":
    import platform
    from checks.result import CheckResult, Status, GDSMode
    system = platform.system()
    if system != "Linux":
        return CheckResult(
            check="Operating system",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=f"GDS is Linux-only. Detected: {system}.",
            mitigation="GDS (GPUDirect Storage) requires a Linux host.",
            evidence=f"platform.system() = {system}",
        )
    distro = ""
    try:
        import subprocess
        r = subprocess.run(["cat", "/etc/os-release"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.splitlines():
            if line.startswith("PRETTY_NAME="):
                distro = line.split("=", 1)[1].strip().strip('"')
                break
    except Exception:
        pass
    return CheckResult(
        check="Operating system",
        mode=GDSMode.NATIVE,
        status=Status.PASS,
        why=f"Linux detected{f': {distro}' if distro else ''}. GDS is supported.",
        evidence=f"platform.system() = {system}" + (f", distro = {distro}" if distro else ""),
    )


def _check_arch() -> "CheckResult":
    import platform
    from checks.result import CheckResult, Status, GDSMode
    machine = platform.machine()
    supported = {"x86_64", "aarch64", "arm64"}
    if machine in supported:
        arch_display = "x86_64 (AMD64)" if machine == "x86_64" else "ARM64 (aarch64)"
        return CheckResult(
            check="CPU architecture",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"Architecture {arch_display} is supported by GDS.",
            evidence=f"platform.machine() = {machine}",
        )
    return CheckResult(
        check="CPU architecture",
        mode=GDSMode.NATIVE,
        status=Status.WARN,
        why=f"Architecture '{machine}' is not a known GDS-supported architecture (x86_64, aarch64).",
        evidence=f"platform.machine() = {machine}",
    )


def _check_cdmm():
    """Detect CDMM mode on Grace platforms only.

    Returns None on non-Grace systems so the caller can skip the section
    entirely. On Grace, returns a CheckResult describing the active
    CoherentGPUMemoryMode (CDMM = "driver", NUMA = "numa"), or a WARN if
    the param file is unreadable (driver not loaded yet).
    """
    from checks import iommu, nvidia_fs
    from checks.result import CheckResult, Status, GDSMode

    if not iommu.is_grace():
        return None

    value = nvidia_fs.coherent_gpu_memory_mode_value()

    if value is None:
        return CheckResult(
            check="CDMM mode",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why=(
                "Grace platform detected, but CoherentGPUMemoryMode cannot be "
                "read from /proc/driver/nvidia/params. The NVIDIA kernel "
                "driver is likely not loaded; this check will be more "
                "informative after the driver runtime stack is up."
            ),
            mitigation=(
                "Load the driver: sudo modprobe nvidia\n"
                "Then re-run, or defer this check to post-install."
            ),
        )

    if value == "driver":
        return CheckResult(
            check="CDMM mode",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=(
                "CDMM (Coherent Device Memory Management) is active: "
                "CoherentGPUMemoryMode=driver."
            ),
            evidence=f"/proc/driver/nvidia/params: CoherentGPUMemoryMode={value!r}",
        )

    if value == "numa":
        return CheckResult(
            check="CDMM mode",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=(
                "CDMM is not active. GPU memory is exposed as a NUMA node: "
                "CoherentGPUMemoryMode=numa."
            ),
            evidence=f"/proc/driver/nvidia/params: CoherentGPUMemoryMode={value!r}",
        )

    if value == "":
        # Param present but empty — driver loaded, but the CoherentGPUMemoryMode
        # module parameter was not explicitly set. The driver is using its
        # default behavior (typically NUMA on recent Grace stacks); CDMM is
        # not active. To enable CDMM, set the module parameter explicitly.
        return CheckResult(
            check="CDMM mode",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=(
                "CDMM is not explicitly enabled. CoherentGPUMemoryMode is empty, "
                "so the NVIDIA driver is operating in its default coherent-memory "
                "behavior (typically NUMA mode on recent Grace stacks)."
            ),
            mitigation=(
                "To explicitly enable CDMM, set the module parameter at load time:\n"
                "  echo 'options nvidia NVreg_RegistryDwords=\"RMNumaOnlining=0x0\"' \\\n"
                "      | sudo tee /etc/modprobe.d/nvidia-cdmm.conf\n"
                f"{nvidia_fs.REBUILD_INITRAMFS_CMDS}\n"
                "Refer to the NVIDIA GDS docs for the exact procedure for your "
                "driver version and distro."
            ),
            evidence="/proc/driver/nvidia/params: CoherentGPUMemoryMode=''",
        )

    return CheckResult(
        check="CDMM mode",
        mode=GDSMode.NATIVE,
        status=Status.WARN,
        why=f"CoherentGPUMemoryMode has an unexpected value: '{value}'.",
        evidence=f"/proc/driver/nvidia/params: CoherentGPUMemoryMode={value!r}",
    )


def _check_kernel_version_info():
    """Return the running kernel version as an informational CheckResult.

    Not a health check on its own — informational only. Advisory WARNs for
    performance-impacting upstream kernel issues are deliberately not
    computed here: the underlying fix is expected to be backported across a
    range of stable kernels, so the kernel version reported by uname is not
    a reliable signal for whether a given kernel is affected.
    """
    import platform
    from checks import kernel as kernel_mod
    from checks.result import CheckResult, Status, GDSMode

    major, minor, patch = kernel_mod._kernel_version()
    ver_str = f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"

    return [CheckResult(
        check="Kernel version",
        mode=GDSMode.NATIVE,
        status=Status.PASS,
        why=f"Running kernel {ver_str}.",
        evidence=f"uname -r → {platform.release()}",
    )]


def _preinstall_p2pdma_results(results: list) -> list:
    """Present missing PCI P2PDMA support as a route advisory.

    The kernel checks are hard failures when evaluating the P2PDMA route
    directly.  Pre-install evaluates overall GDS readiness, where nvfs, RDMA,
    and compat routes may still be available, so consolidate those failures
    into one warning with route-appropriate guidance.
    """
    from checks.result import CheckResult, GDSMode, Status

    failures = [result for result in results if result.status == Status.FAIL]
    if not failures:
        return results

    remaining = [result for result in results if result.status != Status.FAIL]
    evidence = "; ".join(
        f"{result.check}: {result.evidence or result.why}"
        for result in failures
    )
    remaining.append(CheckResult(
        check="PCI P2PDMA route",
        mode=GDSMode.P2PDMA,
        status=Status.WARN,
        why=(
            "This kernel does not include PCI P2PDMA support. This does not "
            "prevent GDS from using the nvidia-fs/nvfs, RDMA, or compat routes."
        ),
        mitigation=(
            "Use the nvidia-fs/nvfs route for NVMe/NVMe-oF GDS on this "
            "platform when the storage stack has the required MLNX_OFED/DOCA "
            "patches. After GDS is installed, verify the active route with:\n"
            "  ./gds-diag.py post-install -v\n"
            "  ./gds-diag.py mount-check <path> -v\n"
            "Only change kernels if this deployment specifically requires "
            "upstream PCI P2PDMA."
        ),
        evidence=evidence,
    ))
    return remaining


def _collect_sections(verbose: bool) -> dict[str, list]:
    from checks import kernel, iommu, pcie, nvidia_fs, rdma
    from checks.result import CheckResult, Status, GDSMode

    sections: dict[str, list] = {}

    # --- System ---
    sections["System"] = [_check_os(), _check_arch()]

    # --- GPU ---
    gpu_results = [
        nvidia_fs.check_gpu_presence_lspci(),
        nvidia_fs.check_open_driver_preinstall(),
    ]
    cdmm_result = _check_cdmm()
    if cdmm_result is not None:
        gpu_results.append(cdmm_result)
    sections["GPU"] = gpu_results

    # --- CUDA Toolkit ---
    sections["CUDA Toolkit"] = [nvidia_fs.check_cuda_toolkit()]

    # --- DOCA / MLNX_OFED ---
    sections["DOCA / MLNX_OFED"] = [rdma.check_ofed_preinstall()]

    # --- Kernel ---
    kernel_results = _preinstall_p2pdma_results(
        list(kernel.run_all(GDSMode.P2PDMA))
    )
    kernel_results.extend(_check_kernel_version_info())
    sections["Kernel"] = kernel_results

    # --- IOMMU ---
    sections["IOMMU"] = iommu.run_all()

    # --- PCIe / ACS ---
    sections["PCIe / ACS"] = [pcie.check_acs()]

    # --- Mounted filesystems ---
    mount_results: list[CheckResult] = []
    try:
        with open("/proc/mounts") as fh:
            mount_results = _collect_mounted_filesystem_results(fh.readlines())

    except FileNotFoundError:
        mount_results.append(CheckResult(
            check="Mounted filesystems",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why="/proc/mounts not available — cannot enumerate mounts.",
        ))

    sections["Mounted Filesystems"] = mount_results
    return sections


def _run_text(args: argparse.Namespace) -> int:
    from checks.output import (
        bold, render_mitigation_plan, render_sections, render_summary,
        render_version_context, overall_exit_code,
    )
    from checks import nvidia_fs
    from checks.version import version_string

    print()
    print(bold("═" * 70))
    print(bold("  GDS Pre-Install Readiness Check"))
    print(bold("═" * 70))
    if args.verbose:
        print(f"  Tool: {version_string()}")
    print()

    for line in render_version_context(
        nvidia_fs.collect_version_context(include_runtime_driver=False)
    ):
        print(line)

    sections = _collect_sections(args.verbose)

    for line in render_sections(sections, verbose=args.verbose):
        print(line)

    print(render_summary(sections))
    print()
    for line in render_mitigation_plan(sections):
        print(line)
    print()
    return overall_exit_code(sections)


def _run_json(args: argparse.Namespace) -> int:
    from checks.output import results_to_json, overall_exit_code
    from checks import nvidia_fs
    from checks.version import tool_metadata

    sections = _collect_sections(args.verbose)
    all_results = [r for results in sections.values() for r in results]

    out = {
        "tool": tool_metadata(),
        "mode": "pre-install",
        "version_context": nvidia_fs.collect_version_context(include_runtime_driver=False),
        "summary": {
            "pass": sum(1 for r in all_results if r.status.value == "PASS"),
            "info": sum(1 for r in all_results if r.status.value == "INFO"),
            "warn": sum(1 for r in all_results if r.status.value == "WARN"),
            "fail": sum(1 for r in all_results if r.status.value == "FAIL"),
        },
        "checks": {
            title: results_to_json(results)
            for title, results in sections.items()
        },
    }
    print(json.dumps(out, indent=2))
    return overall_exit_code(sections)


class PreInstallCommand(Subcommand):
    name = "pre-install"
    help = "validate the system is GDS-capable before installing CUDA/GDS"
    description = _DESCRIPTION
    order = 20

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        return None

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = PreInstallCommand()
