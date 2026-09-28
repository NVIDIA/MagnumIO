# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
mount-check subcommand.

Deep, path-specific GDS diagnostic. Given a real directory, determines
exactly which GDS modes are available for I/O at that path, explains any
blockers, and provides performance recommendations based on hardware
topology.
"""
from __future__ import annotations

import argparse
import json
import os
import sys  # used by _perf_recommendations (subprocess) and mount warning stderr

from ._base import Subcommand

_DESCRIPTION = """\
Path-specific GDS diagnostic.

Requires: CUDA toolkit + GDS installed; a real, accessible path.

Checks performed:

  Filesystem detection
    - Resolve real path; match against /proc/mounts (longest prefix)
    - Detect filesystem type, backing device, mount options
    - Look up GDS capability from FS_CAPABILITIES

  Per-mode checks
    Native GDS:  nvidia_fs loaded, driver version, GPU compute cap, IOMMU,
                 kernel-log nvidia_fs messages, Open Driver, ext4 data mode,
                 O_DIRECT probe
    P2PDMA:      kernel P2PDMA support, IOMMU, ACS, cufile.json per-FS keys
                 (block.nvme, fs.nfs, fs.virtiofs), Open Driver
    RDMA:        MLNX_OFED/DOCA, nvidia_peermem or DmaBuf, rdma_dev_addr_list,
                 IB/RoCE link state, NFS rdma mount option
    Compat:      allow_compat_mode in cufile.json

  Performance recommendations
    - PCIe topology: GPU <-> NVMe distance from nvidia-smi topo -m -nvme
    - NUMA: GPU and NVMe NUMA node alignment
    - NVMe queue depth: /sys/block/nvmeXnY/queue/nr_requests
    - For network FS: MTU, RDMA link speed, active IB links
"""


def _check_cuda_toolkit():
    import glob as _glob
    candidates = ["/usr/local/cuda"] + sorted(_glob.glob("/usr/local/cuda-*"))
    for path in candidates:
        if os.path.isdir(path):
            return path
    try:
        import subprocess
        r = subprocess.run(["which", "nvcc"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.dirname(os.path.dirname(r.stdout.strip()))
    except Exception:
        pass
    return None


def _perf_recommendations(path: str, fs_type: str) -> list[str]:
    """
    Return a list of human-readable performance recommendation strings
    based on PCIe topology, NUMA alignment, and device queue depth.
    """
    import subprocess
    recs: list[str] = []

    # NVMe queue depth
    try:
        from checks.fs_matrix import get_backing_device
        device = get_backing_device(path)
        if device:
            # e.g. /dev/nvme0n1 → nvme0n1
            dev_name = os.path.basename(device)
            qd_path = f"/sys/block/{dev_name}/queue/nr_requests"
            if os.path.exists(qd_path):
                with open(qd_path) as fh:
                    qd = int(fh.read().strip())
                if qd < 1024:
                    recs.append(
                        f"NVMe queue depth is {qd} (recommended ≥ 1024 for GDS throughput).\n"
                        f"  echo 1024 | sudo tee {qd_path}"
                    )
                else:
                    recs.append(f"NVMe queue depth: {qd} ✓")
    except Exception:
        pass

    # NUMA alignment: GPU NUMA node vs NVMe NUMA node
    try:
        from checks.fs_matrix import get_backing_device
        device = get_backing_device(path)
        if device:
            dev_name = os.path.basename(device)
            # Strip partition suffix: nvme0n1p1 → nvme0n1
            dev_base = dev_name.rstrip("0123456789p").rstrip("p") if "nvme" in dev_name else dev_name
            numa_path = f"/sys/block/{dev_name}/device/numa_node"
            if not os.path.exists(numa_path):
                numa_path = f"/sys/block/{dev_base}/device/numa_node"
            nvme_numa: int | None = None
            if os.path.exists(numa_path):
                with open(numa_path) as fh:
                    nvme_numa = int(fh.read().strip())

            gpu_numa: int | None = None
            gpu_numa_path = "/sys/bus/pci/drivers/nvidia"
            if os.path.isdir(gpu_numa_path):
                for bdf in os.listdir(gpu_numa_path):
                    np = f"{gpu_numa_path}/{bdf}/numa_node"
                    if os.path.exists(np):
                        with open(np) as fh:
                            gpu_numa = int(fh.read().strip())
                        break

            if nvme_numa is not None and gpu_numa is not None:
                if nvme_numa == gpu_numa:
                    recs.append(f"NUMA alignment: GPU and NVMe both on NUMA node {gpu_numa} ✓")
                else:
                    recs.append(
                        f"NUMA mismatch: GPU on node {gpu_numa}, NVMe on node {nvme_numa}. "
                        "Cross-NUMA I/O adds latency — prefer placing both on the same node."
                    )
    except Exception:
        pass

    # NVIDIA's topology matrix is the most actionable GPU<->NVMe placement view.
    # Use it for NVMe-backed mount points so users can pick the best NVMe for
    # each GPU, or the best GPU for this mount's NVMe.
    try:
        from checks import pcie
        from checks.fs_matrix import get_backing_device, get_raid_level, is_nvme_backed

        device = get_backing_device(path)
        raid_level = get_raid_level(path) if device else None
        if device and raid_level and is_nvme_backed(path):
            recs.append(
                f"{raid_level.upper()} spans multiple NVMe-backed devices. "
                "GPU locality may be uneven unless the RAID layout was built with "
                "topology in mind; a workload can stripe I/O across NVMe devices "
                "attached to different GPU/root-complex neighborhoods."
            )
        elif device and (is_nvme_backed(path) or fs_type in ("nvme-of", "raid0")):
            topo_recs, topo_err = pcie.topo_nvme_recommendations(os.path.basename(device))
            if topo_recs:
                recs.extend(topo_recs)
            elif topo_err:
                recs.append(
                    "GPU/NVMe topology unavailable from nvidia-smi topo -m -nvme: "
                    f"{topo_err}"
                )
    except Exception:
        pass

    # Network FS: RDMA link speed
    if fs_type in ("lustre", "gpfs", "mmfs", "wekafs", "nfs", "beegfs", "nvme-of"):
        try:
            r = subprocess.run(
                ["ibv_devinfo"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                speeds = [
                    line.split(":")[1].strip()
                    for line in r.stdout.splitlines()
                    if "active_speed" in line
                ]
                widths = [
                    line.split(":")[1].strip()
                    for line in r.stdout.splitlines()
                    if "active_width" in line
                ]
                if speeds:
                    recs.append(f"RDMA link speed: {', '.join(speeds)} @ {', '.join(widths)}")
        except Exception:
            pass

        if fs_type in ("gpfs", "mmfs", "wekafs"):
            try:
                from checks import rdma
                recs.extend(rdma.rdma_policy_recommendations(fs_type))
            except Exception:
                pass

    return recs


def _run_text(args: argparse.Namespace) -> int:
    from checks import gds_report as gp
    from checks.fs_matrix import get_fs_type, FS_CAPABILITIES
    from checks.output import bold, yellow, green, dim
    from checks.version import version_string

    path = args.path

    # CUDA check
    if gp._check_cuda_toolkit() is None:
        gp._print_cuda_install_guide()
        return 2

    if not os.path.exists(path):
        print(
            f"WARNING: Path does not exist: {path}\n"
            "         Checking the filesystem that would contain it based on mount table.\n",
            file=sys.stderr,
        )

    fs_type = get_fs_type(path)
    if not fs_type:
        print(f"ERROR: Could not detect filesystem type for: {path}", file=sys.stderr)
        return 3

    reports = gp.build_mode_reports(path, fs_type)
    if args.verbose:
        print(f"Tool: {version_string()}")
        print()
    print(gp.render_text_report(path, fs_type, reports, verbose=args.verbose))

    # Performance recommendations
    recs = _perf_recommendations(path, fs_type)
    if recs:
        print(bold("  Performance Recommendations"))
        print("  " + "─" * 66)
        for rec in recs:
            for i, line in enumerate(rec.splitlines()):
                prefix = "  • " if i == 0 else "    "
                print(f"{prefix}{line}")
        print()

    return 1 if gp.has_blocking_failures(reports, fs_type) else 0


def _run_json(args: argparse.Namespace) -> int:
    from checks import gds_report as gp
    from checks.fs_matrix import get_fs_type
    from checks.version import tool_metadata

    path = args.path
    path_exists = os.path.exists(path)

    if gp._check_cuda_toolkit() is None:
        print(json.dumps({
            "tool": tool_metadata(),
            "error": "cuda_toolkit_not_found",
            "message": "CUDA toolkit is not installed.",
        }, indent=2))
        return 2

    fs_type = get_fs_type(path)
    if not fs_type:
        print(json.dumps({
            "tool": tool_metadata(),
            "error": "fs_type_unknown",
            "path": path,
        }, indent=2))
        return 3

    reports = gp.build_mode_reports(path, fs_type)
    out = json.loads(gp.render_json_report(path, fs_type, reports))
    out["tool"] = tool_metadata()
    out["mode"] = "mount-check"
    out["path_exists"] = path_exists
    if not path_exists:
        out["note"] = (
            "Path does not exist; diagnosis is based on the filesystem that "
            "would contain it, per the mount table."
        )

    recs = _perf_recommendations(path, fs_type)
    out["performance"] = recs

    print(json.dumps(out, indent=2))

    return 1 if gp.has_blocking_failures(reports, fs_type) else 0


class MountCheckCommand(Subcommand):
    name = "mount-check"
    help = "deep path-specific GDS diagnostic for a given mount/directory"
    description = _DESCRIPTION
    order = 40

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "path",
            help="filesystem path to diagnose (must exist and be readable)",
        )

    def run(self, args: argparse.Namespace) -> int:
        return _run_json(args) if args.json else _run_text(args)


COMMAND = MountCheckCommand()
