# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
post-install subcommand.

Confirm that GDS was installed correctly and is configured properly.
System-wide; no path required. Bridges "hardware is capable" (pre-install)
and "this workload can use GDS" (mount-check).
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import re
import subprocess
from typing import Optional

from checks import iommu
from checks.gdscheck import _find_gdscheck, _gdscheck_section, _run_gdscheck_raw
from checks.gdscheck import (
    driver_config_has_direct_p2pdma_token,
    driver_config_status_supported,
)
from ._base import Subcommand

_DESCRIPTION = """\
Post-install GDS validation.

Requires: CUDA Toolkit, NVIDIA driver, NVIDIA Open Kernel Driver, and GDS
packages such as gds-tools/libcufile/nvidia-fs. If the base prerequisites are
missing, post-install stops before checking GDS packages and asks the user to
fix those first.

Checks performed:

  Installation
    - gdscheck binary at /usr/local/cuda/gds/tools/gdscheck
    - nvidia_fs module loaded (lsmod) or installed (modinfo)
    - NVIDIA Open Driver installed (per gdscheck -p PLATFORM INFO line:
      "Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)")

  gdscheck -p summary
    - Run `gdscheck -p` and parse DRIVER CONFIGURATION
    - Show which filesystem types have which modes active
    - Show active P2PDMA/C2C routes separately from nvidia-fs/nvfs routes
    - Surface any ERROR or WARNING lines

  P2PDMA/C2C Direct Routes
    - Check running-kernel PCI P2PDMA symbols and CONFIG_PCI_P2PDMA
    - Check NVIDIA driver registry dwords required for PCI P2PDMA on x86
    - Show cufile.json route preferences for local NVMe, NVMe-oF, virtiofs, and RAID0
    - Show which gdscheck DRIVER CONFIGURATION lines currently contain p2pdma or c2c

  cufile.json
    - File present at /etc/cufile.json (or CUFILE_ENV_PATH_JSON)
    - Parseable (JSONC — strip comments before parsing)
    - Key settings:
        properties.force_compat_mode  (WARN if true)
        properties.allow_compat_mode  (WARN if false)
        properties.use_pci_p2pdma     (informational)
        logging.level                 (suggest DEBUG when diagnosing)
"""

DOC_LINKS = {
    "cuda": (
        "1",
        "CUDA Toolkit Downloads",
        "https://developer.nvidia.com/cuda-downloads",
    ),
    "driver": (
        "2",
        "CUDA Downloads (NVIDIA driver)",
        "https://developer.nvidia.com/cuda-downloads",
    ),
    "open-driver": (
        "3",
        "CUDA Downloads (Open Kernel Module)",
        "https://developer.nvidia.com/cuda-downloads",
    ),
    "gds": (
        "4",
        "GDS Installation and Troubleshooting Guide",
        "https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html",
    ),
}

def _driver_config_entries(output: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in _gdscheck_section(output, "DRIVER CONFIGURATION"):
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        k, v = key.strip(), val.strip()
        if k and v:
            entries.append((k, v))
    return entries


def _supported_p2pdma_driver(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in {
        "nvme",
        "nvmep2pdma",
        "nvmeof",
        "virtiofs",
        "raid",
        "raid0",
    }


def _old_p2pdma_driver_config_entry(key: str, modes: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in {"nvmep2pdma"} and driver_config_status_supported(modes)


def _p2pdma_routes_from_gdscheck(gds_raw: Optional[str]) -> list:
    from checks.result import CheckResult, Status, GDSMode

    if not gds_raw:
        return [CheckResult(
            check="Active P2PDMA/C2C routes",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why="Could not run gdscheck, so active P2PDMA/C2C routes cannot be determined.",
            mitigation="Run manually: /usr/local/cuda/gds/tools/gdscheck -p",
        )]

    entries = _driver_config_entries(gds_raw)
    p2pdma_entries: list[str] = []
    unsupported_tokens: list[str] = []
    for key, modes in entries:
        if not (
            driver_config_has_direct_p2pdma_token(modes)
            or _old_p2pdma_driver_config_entry(key, modes)
        ):
            continue
        rendered = f"{key}: {modes}"
        if _supported_p2pdma_driver(key):
            p2pdma_entries.append(rendered)
        else:
            unsupported_tokens.append(rendered)

    results = []
    if p2pdma_entries:
        results.append(CheckResult(
            check="Active P2PDMA/C2C routes",
            mode=GDSMode.P2PDMA,
            status=Status.PASS,
            why=(
                "gdscheck reports direct P2PDMA/C2C route token(s) as active for "
                "the following DRIVER CONFIGURATION route(s)."
            ),
            evidence="\n".join(p2pdma_entries),
        ))

        if unsupported_tokens:
            results.append(CheckResult(
                check="Unsupported P2PDMA tokens",
                mode=GDSMode.P2PDMA,
                status=Status.WARN,
                why=(
                    "gdscheck includes p2pdma/c2c tokens for filesystem routes that "
                    "the GDS library does not support for direct P2PDMA/C2C routes. "
                    "Treat these as JSON/global preference artifacts, not active routes."
                ),
                evidence="\n".join(unsupported_tokens),
            ))
        return results

    active = [f"{key}: {modes}" for key, modes in entries]
    why = (
        "gdscheck did not report any active P2PDMA/C2C route on a GDS-supported "
        "direct route. Direct GDS may still be available through nvidia-fs/nvfs "
        "routes listed in DRIVER CONFIGURATION."
    )
    evidence = "\n".join(active) if active else None
    if unsupported_tokens:
        why += " Unsupported p2pdma/c2c tokens were present but ignored."
        evidence = "\n".join(unsupported_tokens)

    return [CheckResult(
        check="Active P2PDMA/C2C routes",
        mode=GDSMode.P2PDMA,
        status=Status.WARN,
        why=why,
        mitigation=(
            "If direct P2P mode is intended, enable both properties.use_pci_p2pdma "
            "and the matching block.* or fs.* key. On x86 this enables upstream "
            "PCI P2PDMA; on Grace Hopper and Grace Blackwell ARM platforms "
            "the same keys enable the C2C path when the route is supported. "
            "Confirm gdscheck reports p2pdma or c2c for the relevant route."
        ),
        evidence=evidence,
    )]


def _has_active_p2pdma_route(gds_raw: Optional[str]) -> bool:
    if not gds_raw:
        return False
    for key, modes in _driver_config_entries(gds_raw):
        if (
            _supported_p2pdma_driver(key)
            and (
                driver_config_has_direct_p2pdma_token(modes)
                or _old_p2pdma_driver_config_entry(key, modes)
            )
        ):
            return True
    return False


def _p2pdma_config_routes() -> list:
    from checks import cufile_config
    from checks.result import CheckResult, Status, GDSMode

    route_specs = [
        ("local NVMe", "local-nvme", "block.nvme.use_pci_p2pdma"),
        ("NVMe-oF", "nvmeof", "block.nvmeof.use_pci_p2pdma"),
        ("virtiofs", "virtiofs", "fs.virtiofs.use_pci_p2pdma"),
        ("RAID0", "raid0", "block.raid.use_pci_p2pdma"),
    ]

    lines: list[str] = []
    enabled: list[str] = []
    mismatches: list[str] = []
    parse_errors: list[str] = []

    for label, profile, scoped_key in route_specs:
        audit = cufile_config.audit_config(profile)
        if audit.get("parse_error"):
            parse_errors.append(f"{label}: {audit['parse_error']}")
            continue

        by_path = {entry["path"]: entry for entry in audit["entries"]}
        global_entry = by_path.get("properties.use_pci_p2pdma")
        scoped_entry = by_path.get(scoped_key)
        global_on = global_entry is not None and global_entry["value"] is True
        scoped_on = scoped_entry is not None and scoped_entry["value"] is True
        route_on = global_on and scoped_on
        state = "enabled" if route_on else "disabled"
        lines.append(
            f"{label}: properties.use_pci_p2pdma={global_on}, "
            f"{scoped_key}={scoped_on} -> {state}"
        )
        if route_on:
            enabled.append(label)
        elif scoped_on and not global_on:
            mismatches.append(label)

    if parse_errors:
        return [CheckResult(
            check="cufile.json P2PDMA/C2C route preferences",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why="Could not fully evaluate P2PDMA cufile.json routes due to parse errors.",
            mitigation="Fix cufile.json parsing, then re-run post-install.",
            evidence="\n".join(parse_errors),
        )]

    if enabled:
        status = Status.PASS if not mismatches else Status.WARN
        return [CheckResult(
            check="cufile.json P2PDMA/C2C route preferences",
            mode=GDSMode.P2PDMA,
            status=status,
            why=(
                f"cufile.json has the direct P2P preference enabled for: {', '.join(enabled)}. "
                "This permits the route when the kernel, topology, filesystem, and GDS "
                "library support it; active routes are confirmed separately by gdscheck "
                "as p2pdma on x86 or c2c on supported GH/GB ARM platforms."
            ) + (
                f" Some other route(s) have the scoped key enabled but the global preference is off: {', '.join(mismatches)}."
                if mismatches else ""
            ),
            mitigation=(
                "For half-enabled routes, set both properties.use_pci_p2pdma and "
                "the matching block.* or fs.* key, or disable both if the nvfs route "
                "is intended instead."
            ) if mismatches else None,
            evidence="\n".join(lines),
        )]

    return [CheckResult(
        check="cufile.json P2PDMA/C2C route preferences",
        mode=GDSMode.P2PDMA,
        status=Status.WARN,
        why=(
            "No P2PDMA/C2C route is fully enabled in cufile.json. This is fine if the "
            "deployment intends to use nvidia-fs/nvfs, RDMA, or compat mode instead."
        ),
        mitigation=(
            "For direct P2P mode, set properties.use_pci_p2pdma=true and also "
            "enable the matching route key such as block.nvme.use_pci_p2pdma=true "
            "or block.nvmeof.use_pci_p2pdma=true. The same keys enable upstream "
            "PCI P2PDMA on x86 and C2C on supported GH/GB ARM platforms. "
            "Direct P2P is library-supported only for NVMe, NVMe-oF, virtiofs, "
            "and RAID0 routes."
        ),
        evidence="\n".join(lines),
    )]


def _optional_p2pdma_result(result):
    from checks.result import CheckResult, Status

    if result.status != Status.FAIL:
        return result

    return CheckResult(
        check=result.check,
        mode=result.mode,
        status=Status.WARN,
        why=(
            result.why
            + " This blocks the direct P2P route (upstream PCI P2PDMA on x86, "
              "C2C on supported GH/GB ARM platforms), but it does not by "
              "itself block nvidia-fs/nvfs, RDMA, or compat routes."
        ),
        mitigation=result.mitigation,
        evidence=result.evidence,
    )


def _fail_prerequisite(result, reason: str, mitigation: Optional[str] = None):
    from checks.result import CheckResult, Status

    if result.status == Status.PASS:
        return result
    return CheckResult(
        check=result.check,
        mode=result.mode,
        status=Status.FAIL,
        why=f"{result.why} {reason}",
        mitigation=mitigation or result.mitigation,
        evidence=result.evidence,
    )


def _post_install_prerequisites() -> list:
    from checks import nvidia_fs
    from checks.result import CheckResult, Status, GDSMode

    cuda_result = _fail_prerequisite(
        nvidia_fs.check_cuda_toolkit(),
        "Post-install validation requires CUDA Toolkit before checking GDS packages.",
        (
            "Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads\n"
            "Then verify:\n"
            "  nvcc --version || ls /usr/local/cuda*"
        ),
    )
    gpu_result = _fail_prerequisite(
        nvidia_fs.check_gpu_presence_lspci(),
        "Post-install GPU-memory GDS validation requires an NVIDIA GPU.",
    )
    driver_installed_result = _fail_prerequisite(
        nvidia_fs.check_driver_installed(),
        "Post-install validation requires a working NVIDIA driver before checking GDS packages.",
        (
            "Install the NVIDIA driver from https://developer.nvidia.com/cuda-downloads "
            "and select the Open Kernel Module / NVIDIA Open Driver option."
        ),
    )

    results = [cuda_result, gpu_result, driver_installed_result]
    runtime_driver_results: list[CheckResult] = []
    if driver_installed_result.status == Status.PASS:
        runtime_driver_results.append(_fail_prerequisite(
            nvidia_fs.check_driver_version(),
            "Post-install validation requires a working NVIDIA driver before checking GDS packages.",
        ))
        runtime_driver_results.append(_fail_prerequisite(
            nvidia_fs.check_driver_type(),
            "Post-install validation requires the NVIDIA Open Kernel Driver before checking GDS packages.",
        ))
        results.extend(runtime_driver_results)

    if (
        cuda_result.status == Status.PASS
        and (
            gpu_result.status != Status.PASS
            or driver_installed_result.status != Status.PASS
            or any(result.status != Status.PASS for result in runtime_driver_results)
        )
    ):
        results.append(CheckResult(
            check="libcufile GPU memory support",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why=(
                "CUDA Toolkit is present, but this host does not have both a "
                "visible NVIDIA GPU and a working NVIDIA driver. Only "
                "system-memory buffers are supported on this host. GPU-memory "
                "buffers are not supported with libcufile until an NVIDIA GPU "
                "and working NVIDIA driver are present."
            ),
            mitigation=(
                "Use libcufile only with system-memory buffers on this host, or "
                "install/expose an NVIDIA GPU plus the NVIDIA Open Driver, reboot "
                "if needed, verify nvidia-smi -L, and rerun post-install."
            ),
        ))

    if any(result.status == Status.FAIL for result in results):
        results.append(CheckResult(
            check="Post-install prerequisite gate",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                "Stopping before GDS package/module checks because the base "
                "post-install prerequisites are not ready."
            ),
            mitigation=(
                "Fix the failed prerequisite(s) above, reboot if the NVIDIA driver "
                "changed, then rerun:\n"
                "  python3 gds-diag.py post-install"
            ),
        ))
    return results


def _prerequisite_rows(results: list) -> list[dict[str, str]]:
    from checks.result import Status

    rows: list[dict[str, str]] = []
    for result in results:
        if result.check == "CUDA Toolkit":
            docs = "[1]"
            issue = "CUDA Toolkit is missing or not usable" if result.status != Status.PASS else "Ready"
            mitigation = "Install CUDA Toolkit; verify nvcc or /usr/local/cuda*"
            recommendation = "Do this before installing or debugging GDS packages."
        elif result.check == "NVIDIA driver installed":
            docs = "[2]"
            issue = "NVIDIA driver is missing or not loaded" if result.status != Status.PASS else "Ready"
            mitigation = "Install NVIDIA driver from CUDA Downloads; select Open Kernel Module"
            recommendation = "Fix this before checking driver version, Open Driver, or GDS packages."
        elif result.check == "NVIDIA GPU (lspci)":
            docs = "[4]"
            issue = "NVIDIA GPU is missing or not visible" if result.status != Status.PASS else "Ready"
            mitigation = result.mitigation or "Expose or install a NVIDIA GPU and verify lspci sees it"
            recommendation = "Required for GPU-memory GDS validation."
        elif result.check == "NVIDIA driver version":
            docs = "[2]"
            if result.status == Status.PASS:
                issue = "Ready"
                mitigation = "Install NVIDIA driver from CUDA Downloads; select Open Kernel Module"
                recommendation = "Use the Open Kernel Driver path for GDS hosts."
            elif "driver module is installed" in result.why:
                issue = "Driver installed, but runtime GPU validation is unavailable"
                mitigation = "Expose/install NVIDIA GPU; verify nvidia-smi -L"
                recommendation = "No GPU-memory GDS validation until a GPU is visible."
            else:
                issue = "NVIDIA driver is missing or nvidia-smi is not usable"
                mitigation = "Install NVIDIA driver from CUDA Downloads; select Open Kernel Module"
                recommendation = "Use the Open Kernel Driver path for GDS hosts."
        elif result.check == "NVIDIA driver type":
            docs = "[2], [3]"
            issue = "NVIDIA Open Kernel Driver is missing or not confirmed" if result.status != Status.PASS else "Ready"
            mitigation = "Install/switch to the Open Kernel Module from CUDA Downloads"
            recommendation = "Required before nvfs or P2PDMA GDS validation."
        elif result.check == "libcufile GPU memory support":
            docs = "[4]"
            issue = (
                "Only system-memory buffers are supported on this host. "
                "GPU-memory buffers are not supported with libcufile until an "
                "NVIDIA GPU and working NVIDIA driver are present."
            )
            mitigation = "Use system-memory buffers only, or fix GPU/driver prerequisites"
            recommendation = "No libcufile GPU-memory path should be expected yet."
        elif result.check == "Post-install prerequisite gate":
            docs = "[4]"
            issue = "Base host prerequisites are not ready"
            mitigation = "Fix failed rows, reboot if driver changed, rerun post-install"
            recommendation = "Do not chase gdscheck or nvidia-fs package errors yet."
        else:
            docs = "[4]"
            issue = result.why
            mitigation = result.mitigation or "Fix this prerequisite and rerun post-install."
            recommendation = "Resolve before GDS package/module validation."

        rows.append({
            "component": result.check,
            "status": result.status.value,
            "missing_or_issue": issue,
            "mitigation": mitigation,
            "docs": docs,
            "recommendation": recommendation,
        })
    return rows



def _render_prerequisite_table(results: list, verbose: bool = False) -> list[str]:
    from checks.output import _wrap_cell, bold
    from checks.result import Status

    visible_results = results if verbose else [r for r in results if r.status != Status.PASS]
    rows = _prerequisite_rows(visible_results)
    widths = {
        "component": 28,
        "status": 6,
        "missing_or_issue": 34,
        "mitigation": 46,
        "docs": 8,
        "recommendation": 44,
    }
    columns = [
        ("component", "Component"),
        ("status", "Status"),
        ("missing_or_issue", "Missing / Issue"),
        ("mitigation", "Mitigation"),
        ("docs", "Docs"),
        ("recommendation", "Recommendation"),
    ]

    def line(char: str = "-") -> str:
        body = "+".join(char * (widths[key] + 2) for key, _ in columns)
        return f"  +{body}+"

    def render_row(row: dict[str, str]) -> list[str]:
        wrapped = {
            key: _wrap_cell(row[key], widths[key])
            for key, _ in columns
        }
        height = max(len(value) for value in wrapped.values())
        rendered: list[str] = []
        for idx in range(height):
            cells = []
            for key, _ in columns:
                value = wrapped[key][idx] if idx < len(wrapped[key]) else ""
                cells.append(f" {value:<{widths[key]}} ")
            rendered.append("  |" + "|".join(cells) + "|")
        return rendered

    header = {key: title for key, title in columns}
    lines = [
        bold("  Prerequisite Remediation Table"),
        "  Post-install stopped before GDS package/module checks. Fix these first.",
        line("="),
    ]
    lines.extend(render_row(header))
    lines.append(line("="))
    for row in rows:
        lines.extend(render_row(row))
        lines.append(line())

    lines.append("  Docs:")
    for key in ("cuda", "driver", "open-driver", "gds"):
        number, title, url = DOC_LINKS[key]
        lines.append(f"    [{number}] {title}: {url}")
    lines.append("")
    return lines


def _prerequisites_blocked(sections: dict[str, list]) -> bool:
    from checks.result import Status

    prereqs = sections.get("Prerequisites", [])
    return any(result.check == "Post-install prerequisite gate" and result.status == Status.FAIL for result in prereqs)


def _p2pdma_topology_candidates() -> list:
    from checks import pcie
    from checks.result import CheckResult, Status, GDSMode

    gpu_bdfs, gpu_error = pcie.get_gpu_bdfs_with_error()
    nvme_bdfs = pcie.get_nvme_bdfs()

    if not gpu_bdfs:
        topo_result = pcie.check_topo_nvme()
        if topo_result.status == Status.PASS:
            topo_result.why = (
                "GPU PCI bus IDs were not available from nvidia-smi query, "
                "so using nvidia-smi topo -m -nvme for P2PDMA topology candidates."
            )
            if gpu_error:
                topo_result.evidence = (
                    f"GPU BDF query detail: {gpu_error}\n"
                    + (topo_result.evidence or "")
                )
            return [topo_result]

        why = "No GPU topology could be discovered via nvidia-smi; cannot list GPU/NVMe P2PDMA topology candidates."
        topo_detail = (topo_result.why or "").replace(
            "Could not read nvidia-smi NVMe topology: ",
            "",
            1,
        )
        details = []
        if gpu_error:
            details.append(f"GPU BDF query: {gpu_error}")
        if topo_detail:
            details.append(f"topo -m -nvme: {topo_detail}")
        if details:
            why += " Detail: " + "; ".join(details)
        return [CheckResult(
            check="P2PDMA topology candidates",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why=why,
            mitigation=(
                "Ensure the NVIDIA kernel driver is loaded and healthy, then verify:\n"
                "  nvidia-smi -L\n"
                "If running in a container, ensure GPU devices and driver libraries are exposed."
            ),
        )]

    if not nvme_bdfs:
        return [CheckResult(
            check="P2PDMA topology candidates",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why="No local NVMe block devices found. Local NVMe P2PDMA/C2C routes cannot be evaluated on this host.",
        )]

    optimal: list[str] = []
    suboptimal: list[str] = []
    for gpu in gpu_bdfs:
        for nvme in nvme_bdfs:
            same, evidence = pcie.same_root_complex(gpu, nvme)
            label = f"GPU {gpu} <-> NVMe {nvme}"
            if same:
                optimal.append(f"{label}: {evidence}")
            else:
                suboptimal.append(f"{label}: different root complex")

    if optimal:
        why = "At least one GPU/NVMe pair appears optimal for local NVMe P2PDMA."
        if suboptimal:
            why += (
                " Other pairs cross root ports/root complexes and may still work, "
                "but are expected to be less performant."
            )
        return [CheckResult(
            check="P2PDMA topology candidates",
            mode=GDSMode.P2PDMA,
            status=Status.PASS,
            why=why,
            evidence="\n".join((optimal + suboptimal)[:12]),
        )]

    return [CheckResult(
        check="P2PDMA topology candidates",
        mode=GDSMode.P2PDMA,
        status=Status.INFO,
        why=(
            "No local GPU/NVMe pair appears to share a root complex. GDS can "
            "still operate across root ports, but expect lower performance than "
            "a same-root-port or same-switch path."
        ),
        mitigation=(
            "For best performance, prefer the closest GPU/NVMe pair from "
            "nvidia-smi topo -m -nvme, use NUMA binding near that path, or place "
            "devices under the same root port/P2P-capable switch when possible. "
            "Do not treat cross-root-port topology alone as a GDS blocker; verify "
            "actual route activation with gdscheck or a workload."
        ),
        evidence="\n".join(suboptimal[:12]),
    )]


def _collect_p2pdma_sections(gds_raw: Optional[str]) -> list:
    from checks import kernel, nvidia_fs, pcie
    from checks.result import GDSMode

    results = []
    results.extend(
        _optional_p2pdma_result(result)
        for result in kernel.run_all(GDSMode.P2PDMA)
    )
    registry_result = nvidia_fs.check_p2pdma_driver_registries()
    if registry_result is not None:
        results.append(_optional_p2pdma_result(registry_result))
    results.extend(_p2pdma_config_routes())
    results.extend(_p2pdma_topology_candidates())
    results.append(_optional_p2pdma_result(pcie.check_acs()))
    results.extend(_p2pdma_routes_from_gdscheck(gds_raw))
    return results


def _collect_sections(verbose: bool) -> dict[str, list]:
    from checks import nvidia_fs, cufile_config
    from checks.result import CheckResult, Status, GDSMode

    sections: dict[str, list] = {}

    prereq_results = _post_install_prerequisites()
    sections["Prerequisites"] = prereq_results
    if any(result.status == Status.FAIL for result in prereq_results):
        return sections

    # --- Installation checks ---
    install_results: list[CheckResult] = []

    gdscheck_path = _find_gdscheck()
    if gdscheck_path:
        install_results.append(CheckResult(
            check="gdscheck binary",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"gdscheck found at {gdscheck_path}.",
            evidence=gdscheck_path,
        ))
    else:
        install_results.append(CheckResult(
            check="gdscheck binary",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                "gdscheck not found. CUDA Toolkit and the matching gds-tools package must be installed. "
                "Expected at /usr/local/cuda/gds/tools/gdscheck."
            ),
            mitigation=(
                "First verify CUDA Toolkit is installed:\n"
                "  nvcc --version || ls /usr/local/cuda*\n"
                "If missing, install CUDA from https://developer.nvidia.com/cuda-downloads\n\n"
                "Then install the matching GDS tools package:\n"
                "  Ubuntu: sudo apt-get install gds-tools-13-1\n"
                "  RHEL:   sudo dnf install gds-tools-13-3\n"
                "  Replace the CUDA version suffix so it matches the installed toolkit.\n"
                "Verify packages:\n"
                "  rpm -qa | grep gds\n"
                "  dpkg -l | grep gds\n"
                "Then verify: ls /usr/local/cuda*/gds/tools/gdscheck"
            ),
        ))

    # Run gdscheck once; share output across remaining checks
    gds_raw, _ = _run_gdscheck_raw(gdscheck_path) if gdscheck_path else (None, None)

    install_results.append(nvidia_fs.check_libcufile_version(gds_raw))

    nvidia_fs_result = nvidia_fs.check_nvidia_fs_loaded()
    if nvidia_fs_result.status == Status.FAIL and _has_active_p2pdma_route(gds_raw):
        nvidia_fs_result = CheckResult(
            check=nvidia_fs_result.check,
            mode=nvidia_fs_result.mode,
            status=Status.WARN,
            why=(
                nvidia_fs_result.why
                + " An active p2pdma/c2c route is present, so this is an nvfs-route "
                  "issue rather than a blocker for direct P2P workloads."
            ),
            mitigation=nvidia_fs_result.mitigation,
            evidence=nvidia_fs_result.evidence,
        )
    install_results.append(nvidia_fs_result)

    install_results.append(
        nvidia_fs.check_open_driver(gds_raw or "", GDSMode.NATIVE)
    )
    sections["Installation"] = install_results

    # --- gdscheck DRIVER CONFIGURATION summary ---
    driver_results: list[CheckResult] = []
    if gds_raw:
        active = [f"{key}: {modes}" for key, modes in _driver_config_entries(gds_raw)]

        if active:
            driver_results.append(CheckResult(
                check="GDS active modes",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="gdscheck reports the following filesystem modes are active.",
                evidence="\n".join(active),
            ))
        else:
            driver_results.append(CheckResult(
                check="GDS active modes",
                mode=GDSMode.NATIVE,
                status=Status.WARN,
                why="DRIVER CONFIGURATION section in gdscheck output is empty.",
            ))

        # Surface gdscheck ERROR/WARNING lines
        for line in gds_raw.splitlines():
            ll = line.strip().lower()
            if ll.startswith("error") or ll.startswith("[error]"):
                driver_results.append(CheckResult(
                    check="gdscheck error",
                    mode=GDSMode.NATIVE,
                    status=Status.FAIL,
                    why=f"gdscheck reported an error: {line.strip()}",
                ))
            elif ll.startswith("warn") or ll.startswith("[warn]"):
                why = f"gdscheck reported a warning: {line.strip()}"
                mitigation = None
                status = Status.WARN
                if "iommu" in ll:
                    state, _ = iommu.detect_iommu_state()
                    if state == iommu.IommuState.STRICT:
                        mitigation = iommu._strict_mitigation(iommu.arch())
                    elif state == iommu.IommuState.PASSTHROUGH:
                        # Passthrough is already GDS's recommended IOMMU state
                        # (checks/iommu.py's dedicated check agrees). Don't tell
                        # the operator to redo a passthrough config they already
                        # have. On x86 there's still a real, actionable
                        # alternative (disabling IOMMU entirely), so stay WARN
                        # and surface it. On ARM there is no such knob, so
                        # there's nothing useful to say — downgrade to PASS so
                        # it stays quiet like the rest of the tool's PASS results.
                        if iommu.arch() == iommu.Arch.ARM:
                            status = Status.PASS
                            why += (
                                " IOMMU is already in passthrough mode, which is "
                                "GDS's recommended configuration — no cmdline "
                                "change needed."
                            )
                        else:
                            why += (
                                " IOMMU is already in passthrough mode, which is "
                                "GDS's recommended configuration — no cmdline "
                                "change needed. This warning is advisory: GDS is "
                                "not guaranteed to be fully performant even with "
                                "passthrough enabled, so some workloads may still "
                                "see reduced throughput."
                            )
                            mitigation = (
                                "If you need IOMMU for another purpose (e.g. VM device "
                                "passthrough), keep it enabled; otherwise disabling it "
                                "entirely (intel_iommu=off / amd_iommu=off / iommu=off) "
                                "can remove the last bit of DMA translation overhead."
                            )
                driver_results.append(CheckResult(
                    check="gdscheck warning",
                    mode=GDSMode.NATIVE,
                    status=status,
                    why=why,
                    mitigation=mitigation,
                ))
    else:
        driver_results.append(CheckResult(
            check="GDS active modes",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why="Could not run gdscheck — cannot determine which GDS modes are active.",
            mitigation="Run manually: /usr/local/cuda/gds/tools/gdscheck -p",
        ))
    sections["GDS Driver Configuration"] = driver_results

    # --- P2PDMA direct-route capability and active route summary ---
    sections["P2PDMA/C2C Direct Routes"] = _collect_p2pdma_sections(gds_raw)

    # --- cufile.json ---
    cufile_results = cufile_config.run_compat_checks() + [
        result for result in cufile_config.run_all("ext4", gdscheck_output=gds_raw)
        if result.mode != GDSMode.P2PDMA
    ]
    sections["cufile.json"] = cufile_results

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
    print(bold("  GDS Post-Install Validation"))
    print(bold("═" * 70))
    if args.verbose:
        print(f"  Tool: {version_string()}")
    print()

    for line in render_version_context(
        nvidia_fs.collect_version_context(include_runtime_driver=True)
    ):
        print(line)

    sections = _collect_sections(args.verbose)

    if _prerequisites_blocked(sections):
        for line in _render_prerequisite_table(sections["Prerequisites"], verbose=args.verbose):
            print(line)
    else:
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
        "mode": "post-install",
        "version_context": nvidia_fs.collect_version_context(include_runtime_driver=True),
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
    if _prerequisites_blocked(sections):
        out["prerequisite_remediation"] = {
            "rows": _prerequisite_rows(sections["Prerequisites"]),
            "docs": [
                {"ref": number, "title": title, "url": url}
                for number, title, url in DOC_LINKS.values()
            ],
        }
    print(json.dumps(out, indent=2))
    return overall_exit_code(sections)


class PostInstallCommand(Subcommand):
    name = "post-install"
    help = "confirm GDS was installed correctly and configured properly"
    description = _DESCRIPTION
    order = 30

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        return None

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = PostInstallCommand()
