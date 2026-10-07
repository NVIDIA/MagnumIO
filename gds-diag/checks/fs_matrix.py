# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Filesystem → GDS mode capability matrix.

Source: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html

Capability values:
  True      — mode supported for this filesystem
  False     — mode NOT supported (architectural limitation, not fixable)
  "config"  — supported but requires a matching cufile.json key

Adding a new filesystem
-----------------------
FS_CAPABILITIES (this file) is the entry point, but four locations must be updated:

1. FS_CAPABILITIES (this file)
   Add the capability entry.  If the FS is a known alias for an existing type,
   add it to FS_ALIASES instead.

2. checks/gds_report.py — _FS_DRIVER_KEYS dict
   Map the fs_type string to the key name(s) that appear under DRIVER
   CONFIGURATION in `gdscheck -p` output.  Without this, gdscheck verdicts are
   silently ignored for the new FS.
   Also add FS-specific runtime checks (mount option, write-path warning, etc.)
   in build_mode_reports() if the FS needs them.

3. checks/cufile_config.py — CONFIG_SCHEMA list
   Add schema entries for every fs.<name>.* cufile.json key the FS supports,
   with the appropriate "profiles": ["<name>"] tag.

4. subcommands/config_audit.py — _PROFILES list
   Add the FS name so it is selectable via `config-audit --profile <name>`.
   If the FS uses cuFile userspace RDMA (rdma_type="userspace"), also add it
   to RDMA_POLICY_FS_TYPES in checks/rdma.py.

See doc/DESIGN.md § "Adding a new filesystem" for rationale and examples.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

# GDS library direct P2P support is limited to these storage route families.
# On x86, the route is PCI P2PDMA. On supported NVIDIA Grace Hopper and
# Grace Blackwell ARM platforms, the same use_pci_p2pdma keys
# can enable the coherent C2C path and gdscheck may report "c2c".
# cufile.json can expose other use_pci_p2pdma-looking keys, but those settings
# do not create library support for unsupported filesystems.
P2PDMA_LIBRARY_ROUTES: dict[str, str] = {
    "nvme": "NVMe / local NVMe-backed ext4 or XFS",
    "nvmeof": "NVMe-oF",
    "virtiofs": "virtiofs",
    "raid0": "RAID0",
}

P2PDMA_SUPPORTED_ROUTE_KEYS: dict[str, str] = {
    "nvme": "block.nvme.use_pci_p2pdma",
    "nvmeof": "block.nvmeof.use_pci_p2pdma",
    "virtiofs": "fs.virtiofs.use_pci_p2pdma",
    "raid0": "block.raid.use_pci_p2pdma",
}

P2PDMA_UNSUPPORTED_ROUTE_KEYS: set[str] = {
    "fs.lustre.use_pci_p2pdma",
    "fs.beegfs.use_pci_p2pdma",
    "fs.scatefs.use_pci_p2pdma",
    "fs.gpfs.use_pci_p2pdma",
}

# ---------------------------------------------------------------------------
# Capability matrix
# ---------------------------------------------------------------------------

FS_CAPABILITIES: dict[str, dict] = {
    # ---- Local block-backed ------------------------------------------------
    # rdma_type field (only present when rdma is True):
    #   "userspace" — uses cuFile userspace RDMA (requires nvidia_peermem or rdma_peer_type=dmabuf)
    #   "kernel"    — uses kernel-level RDMA via nvidia-fs (e.g. Lustre/BeeGFS LNet-style transport,
    #                 NFS's NFSoRDMA); no nvidia_peermem needed. NFS additionally needs the rdma
    #                 mount option — that's fs-specific remediation, not a distinct rdma_type.
    "ext4": {
        "native": True, "p2pdma": True, "rdma": False, "compat": True,
        "p2pdma_display": "NoMP Config",
        "p2pdma_note": (
            "NVMe P2PDMA/C2C requires properties.use_pci_p2pdma=true, "
            "block.nvme.use_pci_p2pdma=true, and (on x86) NVMe multipath disabled "
            "unless the system has a specialized multipath patch; not required on "
            "NVIDIA CPUs, where multipath works fine alongside C2C."
        ),
        "notes": (
            "ext4 is fully supported by GDS. Requires O_DIRECT. "
            "For NVMe-backed ext4, direct GDS can use either upstream PCI P2PDMA "
            "on x86 (configured with block.nvme.use_pci_p2pdma and NVMe multipath "
            "disabled) or C2C on supported NVIDIA CPUs (configured with "
            "block.nvme.use_pci_p2pdma; NVMe multipath is not a blocker there) "
            "or the nvidia-fs nvfs path when the NVMe stack has GDS patches from "
            "MLNX_OFED or DOCA/DOCA-OFED."
        ),
    },
    "xfs": {
        "native": True, "p2pdma": True, "rdma": False, "compat": True,
        "p2pdma_display": "NoMP Config",
        "p2pdma_note": (
            "NVMe P2PDMA/C2C requires properties.use_pci_p2pdma=true, "
            "block.nvme.use_pci_p2pdma=true, and (on x86) NVMe multipath disabled "
            "unless the system has a specialized multipath patch; not required on "
            "NVIDIA CPUs, where multipath works fine alongside C2C."
        ),
        "notes": (
            "XFS is the preferred local filesystem for GDS due to its extent-based layout. "
            "Requires O_DIRECT. For NVMe-backed XFS, direct GDS can use either upstream "
            "PCI P2PDMA on x86 (configured with block.nvme.use_pci_p2pdma and NVMe "
            "multipath disabled) or C2C on supported NVIDIA CPUs (configured with "
            "block.nvme.use_pci_p2pdma; NVMe multipath is not a blocker there) "
            "or the nvidia-fs nvfs path when the NVMe stack has GDS patches "
            "from MLNX_OFED or DOCA/DOCA-OFED."
        ),
    },
    "nvme-of": {
        "native": True, "p2pdma": True, "rdma": True, "rdma_type": "kernel", "compat": True,
        "p2pdma_display": "Config",
        "p2pdma_note": (
            "NVMe-oF P2PDMA/C2C requires properties.use_pci_p2pdma=true and "
            "block.nvmeof.use_pci_p2pdma=true."
        ),
        "notes": (
            "NVMe over Fabrics (NVMeOF): direct GDS can use either upstream PCI P2PDMA "
            "on x86 or C2C on supported GH/GB ARM platforms "
            "or the nvidia-fs nvfs path when the NVMe-oF stack has GDS patches from "
            "MLNX_OFED/DOCA — same kernel-level RDMA dependency as Lustre/BeeGFS/NFS/ScaTeFS. "
            "P2PDMA requires block.nvmeof.use_pci_p2pdma=true in "
            "cufile.json (separate key from block.nvme)."
        ),
    },
    "raid0": {
        "native": True, "p2pdma": "config", "rdma": False, "compat": True,
        "p2pdma_display": "Arch Config",
        "p2pdma_note": (
            "RAID0 P2PDMA/C2C requires NVIDIA Grace or Linux kernel >= 7.1, "
            "block.raid.use_pci_p2pdma=true, and runtime confirmation."
        ),
        "notes": (
            "RAID0 over supported NVMe devices. The nvidia-fs/nvfs direct path is supported. "
            "P2PDMA/C2C for RAID0 requires NVIDIA Grace or Linux kernel >= 7.1, "
            "block.raid.use_pci_p2pdma=true alongside properties.use_pci_p2pdma=true, "
            "and runtime confirmation from gdscheck/workload testing."
        ),
    },
    "scsi": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "SCSI block devices: compat mode only. Not in the GDS native-support list. "
            "No O_DIRECT path through nvidia-fs for legacy SCSI. "
            "Migrate to NVMe for native GDS support."
        ),
    },
    "nvmesh": {
        "native": True, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "Excelero NVMesh: native GDS supported via nvidia-fs (nvfs path). "
            "P2PDMA not supported. Requires NVMesh client and compatible NVIDIA driver."
        ),
    },
    "ddn exascaler": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": (
            "DDN EXAScaler is Lustre-based — identical GDS capabilities to Lustre. "
            "Native GDS via nvidia-fs + LNet (nvfs path). "
            "RDMA is kernel-level via LNet over InfiniBand/RoCE — nvidia_peermem is NOT required. "
            "Requires MLNX_OFED or DOCA for IB drivers."
        ),
    },
    "scaleflux": {
        "native": True, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "ScaleFlux CSD (Computational Storage Device): native GDS supported "
            "via nvidia-fs (nvfs path) — the nvidia-fs driver documents XFS/ext4 "
            "in ordered mode on ScaleFlux CSD devices as a direct peer to "
            "NVMe/NVMe-oF, not compat-only. P2PDMA not supported."
        ),
    },
    "scatefs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": (
            "ScaTeFS (NEC Scalable Technology File System — unrelated to "
            "ScaleFlux despite the similar name): native GDS supported via "
            "nvidia-fs (nvfs path) — NVIDIA's GDS release notes list ScaTeFS "
            "support as a nvidia-fs driver capability, activated via kernel-level "
            "RDMA, same shape as Lustre and BeeGFS. P2PDMA not supported."
        ),
    },
    "ext3": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "ext3 is not in the GDS supported filesystem list. "
            "Only ext4 and XFS are supported local block filesystems for GDS. "
            "Migrate to ext4: tune2fs -j /dev/sdX or reformat."
        ),
    },
    "ext2": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "ext2 is not in the GDS supported filesystem list. "
            "Only ext4 and XFS are supported local block filesystems for GDS. "
            "Migrate to ext4: tune2fs -j /dev/sdX (adds journal) or reformat."
        ),
    },

    # ---- Network / parallel filesystems ------------------------------------
    "lustre": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": (
            "Lustre has strong GDS support. Native GDS uses the nvidia-fs + LNet path. "
            "RDMA is kernel-level (LNet over InfiniBand/RoCE) — nvidia_peermem is NOT required. "
            "Requires MLNX_OFED or DOCA for IB drivers. "
            "Small I/O below fs.lustre.posix_gds_min_kb falls back to POSIX automatically."
        ),
    },
    "nfs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": (
            "NFS GDS support is through NFSoRDMA — a different direct path than the "
            "NVMe nvfs route, but nvidia-fs (nvfs) still has to be loaded to activate it. "
            "Covers both NFSv3 and NFSv4 mounts (nfs4 is an alias to this entry) — "
            "do not diagnose either as local NVMe or NVMe-oF P2PDMA. "
            "For VAST and other NFS-over-RDMA deployments, mount with proto=rdma or rdma, "
            "port=20049, and the required NFS-RDMA client support from MLNX_OFED/DOCA "
            "or the vendor-supported VAST client stack."
        ),
    },
    "wekafs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "userspace", "compat": True,
        "notes": (
            "WekaFS: native GDS supported for reads via cuFile userspace RDMA (dmabuf). "
            "Writes use the POSIX fallback path by default; this is controlled by "
            "'fs.weka.rdma_write_support' in cufile.json (default false), not a fixed "
            "WekaFS architectural limit. Enable it if your WekaFS deployment/client "
            "supports RDMA writes. "
            "Requires nvidia_peermem loaded OR rdma_peer_type=dmabuf in cufile.json. "
            "Requires WekaFS client + MLNX_OFED or DOCA."
        ),
    },
    "gpfs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "userspace", "compat": True,
        "notes": (
            "IBM Spectrum Scale (GPFS). Native GDS via cuFile userspace RDMA (dmabuf) supported. "
            "P2PDMA is NOT supported for GPFS. The GDS library only supports P2PDMA "
            "for NVMe, NVMe-oF, virtiofs, and RAID0 routes. gdscheck may report "
            "'p2pdma' in the IBM Spectrum Scale line when properties.use_pci_p2pdma=true "
            "is set globally in cufile.json, but that JSON setting does not create "
            "library support for GPFS. "
            "Requires nvidia_peermem loaded OR rdma_peer_type=dmabuf in cufile.json. "
            "Requires GPFS client (mmfs) and MLNX_OFED or DOCA."
        ),
    },
    "mmfs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "userspace", "compat": True,
        "notes": "IBM Spectrum Scale (GPFS, legacy mount type name). Same capabilities as gpfs.",
    },
    "beegfs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": (
            "BeeGFS native GDS supported via kernel-level RDMA — nvidia-fs (nvfs) has to be "
            "loaded to activate it, same as Lustre. P2PDMA not supported. "
            "Small I/O falls back to POSIX automatically. "
            "Requires BeeGFS client + MLNX_OFED or DOCA for IB drivers."
        ),
    },
    "fhgfs": {
        "native": True, "p2pdma": False, "rdma": True, "rdma_type": "kernel", "compat": True,
        "notes": "BeeGFS (legacy name fhgfs). Same capabilities as beegfs. P2PDMA not supported.",
    },

    # ---- Cloud ---------------------------------------------------------------
    "fuse.fsx_lustre": {
        "native": True, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "Amazon FSx for Lustre. Native GDS supported via nvidia-fs + LNet. "
            "No RDMA in cloud environments. P2PDMA not supported."
        ),
    },

    # ---- Compat-only (cuFile source: these FSes go through CPU bounce buffer only) ----
    # cuFile recognises: ramfs, tmpfs, squashfs, overlayfs, zfs, btrfs as compat-only.
    # Compat mode works ONLY when allow_compat_mode=true in cufile.json.
    # If allow_compat_mode=false, cuFile I/O on these filesystems returns an error.
    "squashfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.16+",
        "compat_min_version": (1, 16),
        "compat_since_note": "GDS v1.16+ supports squashfs through the compatibility path.",
        "notes": (
            "squashfs: read-only compressed filesystem. cuFile compat-only (CPU bounce). "
            "Requires allow_compat_mode=true in cufile.json."
        ),
    },
    "tmpfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.16+",
        "compat_min_version": (1, 16),
        "compat_since_note": "GDS v1.16+ supports tmpfs through the compatibility path.",
        "notes": (
            "tmpfs: cuFile compat-only (CPU bounce). "
            "Requires allow_compat_mode=true in cufile.json. "
            "No O_DIRECT support — native GDS/P2PDMA not possible."
        ),
    },
    "ramfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.16+",
        "compat_min_version": (1, 16),
        "compat_since_note": "GDS v1.16+ supports ramfs through the compatibility path.",
        "notes": (
            "ramfs: cuFile compat-only (CPU bounce). "
            "Requires allow_compat_mode=true in cufile.json."
        ),
    },
    "overlay": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.16+",
        "compat_min_version": (1, 16),
        "compat_since_note": "GDS v1.16+ supports OverlayFS through the compatibility path.",
        "notes": (
            "OverlayFS (Docker/container runtimes): cuFile compat-only. "
            "Requires allow_compat_mode=true in cufile.json. "
            "For GDS in containers, bind-mount a native ext4/xfs path directly."
        ),
    },
    "overlayfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.16+",
        "compat_min_version": (1, 16),
        "compat_since_note": "GDS v1.16+ supports OverlayFS through the compatibility path.",
        "notes": "OverlayFS (alternate name). cuFile compat-only. Requires allow_compat_mode=true.",
    },
    "zfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.17+",
        "compat_min_version": (1, 17),
        "compat_since_note": "GDS v1.17 release notes explicitly add ZFS support in the compatibility path across all I/O APIs.",
        "notes": (
            "ZFS: cuFile compat-only (CPU bounce). "
            "Requires allow_compat_mode=true in cufile.json. "
            "ZFS is not in the GDS native-support list."
        ),
    },
    "btrfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "compat_since": "1.17+",
        "compat_min_version": (1, 17),
        "compat_since_note": "GDS v1.17 release notes explicitly add BTRFS support in the compatibility path across all I/O APIs.",
        "notes": (
            "btrfs: cuFile compat-only (CPU bounce). "
            "Requires allow_compat_mode=true in cufile.json. "
            "btrfs CoW is incompatible with O_DIRECT for overwrites."
        ),
    },
    # ---- No GDS support (not even compat) ------------------------------------
    "vfat": {
        "native": False, "p2pdma": False, "rdma": False, "compat": False,
        "notes": "FAT/vFAT: no O_DIRECT support — no GDS mode possible, including compat.",
    },
    "fat": {
        "native": False, "p2pdma": False, "rdma": False, "compat": False,
        "notes": "FAT: no O_DIRECT support — no GDS mode possible, including compat.",
    },
    # ---- Other unsupported ---------------------------------------------------
    "virtiofs": {
        "native": False, "p2pdma": "config", "rdma": False, "compat": True,
        "notes": (
            "VirtioFS: supported in P2PDMA mode only (no nvfs/native GDS). "
            "Requires both properties.use_pci_p2pdma=true and "
            "fs.virtiofs.use_pci_p2pdma=true in cufile.json — the fs-specific "
            "key alone is not enough for gdscheck to report the route active. "
            "gdscheck DRIVER CONFIGURATION shows 'VIRTIOFS : p2pdma, compat' when active. "
            "These cufile.json keys are necessary but not sufficient: the virtio-fs "
            "backend itself must be an optimized implementation with P2PDMA support. "
            "A stock/software-only virtiofs backend can report this config as active "
            "without actually moving data via P2P — verify with a real I/O test, not "
            "just gdscheck output."
        ),
    },
    "fuse": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": (
            "FUSE: userspace layer breaks direct kernel-to-GPU DMA. "
            "Compat mode only. Requires allow_compat_mode=true in cufile.json."
        ),
    },
    "cifs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": "CIFS/SMB: compat mode only. Requires allow_compat_mode=true in cufile.json.",
    },
    "smbfs": {
        "native": False, "p2pdma": False, "rdma": False, "compat": True,
        "notes": "SMB: compat mode only. Requires allow_compat_mode=true in cufile.json.",
    },
}

# Map known aliases to canonical names
FS_ALIASES: dict[str, str] = {
    "fuse.beegfs": "beegfs",
    "fuse.gpfs":   "gpfs",
    "lustre2":     "lustre",
    "nfs4":        "nfs",
    "nfsv4":       "nfs",
}

# Filesystems where P2PDMA requires an fs-specific cufile.json key.
# Used by cufile_config.check_p2pdma_config_for_fs() to know which key to check.
# Local NVMe-backed FSes (ext4, xfs) use block.nvme.use_pci_p2pdma instead — not here.
P2PDMA_CONFIG_KEY: dict[str, str] = {
    "virtiofs": "fs.virtiofs.use_pci_p2pdma",
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _decode_proc_mounts_field(value: str) -> str:
    """Decode octal escapes (e.g. "\\040" for space) used by /proc/mounts
    for spaces, tabs, backslashes, and newlines in paths."""
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _path_under_mount(realpath: str, mount_point: str) -> bool:
    """True if realpath is at or under mount_point, on a path-component boundary.

    A plain str.startswith() also matches unrelated sibling paths (e.g. a
    "/data" mount would match "/data2/file"). Require an exact match or a
    trailing-separator boundary instead.
    """
    if realpath == mount_point:
        return True
    return realpath.startswith(mount_point.rstrip("/") + "/")


def get_fs_type(path: str) -> Optional[str]:
    """
    Detect filesystem type for a given path.
    Reads /proc/mounts for longest-prefix match.
    """
    try:
        realpath = os.path.realpath(path)
        best_mount = ""
        best_fstype: Optional[str] = None
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = _decode_proc_mounts_field(parts[1])
                fstype = parts[2]
                if _path_under_mount(realpath, mount_point) and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = fstype
        if best_fstype:
            normalized = FS_ALIASES.get(best_fstype, best_fstype).lower()
            return normalized
    except FileNotFoundError:
        pass  # not Linux — fall through to stat

    # Fallback for non-Linux or /proc unavailable
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", path],
            capture_output=True, text=True, timeout=5,
        )
        fstype = result.stdout.strip()
        return FS_ALIASES.get(fstype, fstype).lower() if fstype else None
    except Exception:
        return None


def get_fs_capabilities(fs_type: str) -> dict:
    """Return capability dict for a filesystem type."""
    key = FS_ALIASES.get(fs_type, fs_type).lower()
    return FS_CAPABILITIES.get(key, {
        "native": None, "p2pdma": None, "rdma": None, "compat": None,
        "notes": (
            f"Unknown filesystem type '{fs_type}'. GDS support status cannot be determined. "
            "Check https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html "
            "for your filesystem."
        ),
    })


def get_backing_device(path: str) -> Optional[str]:
    """Return the block device backing a path, e.g. /dev/nvme0n1."""
    try:
        result = subprocess.run(
            ["df", "--output=source", path],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) >= 2:
            return lines[1].strip()
    except Exception:
        pass
    return None


def _sysfs_block_name(device: str) -> str:
    """
    Return the /sys/class/block name for a device path.

    Device-mapper nodes such as /dev/mapper/vg0-lv0 are symlinks to /dev/dm-N,
    and only the dm-N name exists in sysfs, so resolve symlinks first.
    """
    return os.path.basename(os.path.realpath(device))


def _nvme_controller_from_device(device: str) -> Optional[str]:
    name = os.path.basename(device)
    m = re.match(r"^(nvme\d+)n\d+(p\d+)?$", name)
    if m:
        return m.group(1)
    return None


def _block_device_has_nvme_leaf(device_name: str, seen: Optional[set[str]] = None) -> bool:
    """
    Return True if a block device is itself NVMe or is stacked on NVMe slaves.

    Filesystems can sit on mdraid, dm-crypt, LVM, or other stacked block
    devices. In sysfs those devices expose lower-level devices through
    /sys/class/block/<dev>/slaves, so walk that graph before deciding a path is
    not NVMe-backed.
    """
    if seen is None:
        seen = set()
    name = os.path.basename(device_name)
    if not name or name in seen:
        return False
    seen.add(name)

    if name.startswith("nvme"):
        return True

    slaves_dir = f"/sys/class/block/{name}/slaves"
    try:
        slaves = os.listdir(slaves_dir)
    except (FileNotFoundError, PermissionError, OSError):
        slaves = []
    return any(_block_device_has_nvme_leaf(slave, seen) for slave in slaves)


def _block_device_raid_levels(device_name: str, seen: Optional[set[str]] = None) -> list[str]:
    """Return mdraid levels found at this block device or below it."""
    if seen is None:
        seen = set()
    name = os.path.basename(device_name)
    if not name or name in seen:
        return []
    seen.add(name)

    levels: list[str] = []
    level_path = f"/sys/class/block/{name}/md/level"
    try:
        with open(level_path) as fh:
            level = fh.read().strip().lower()
            if level:
                levels.append(level)
    except (FileNotFoundError, PermissionError, OSError):
        pass

    slaves_dir = f"/sys/class/block/{name}/slaves"
    try:
        slaves = os.listdir(slaves_dir)
    except (FileNotFoundError, PermissionError, OSError):
        slaves = []
    for slave in slaves:
        levels.extend(_block_device_raid_levels(slave, seen))
    return levels


def get_raid_level(path: str) -> Optional[str]:
    """Return the backing mdraid level for a path, if visible in sysfs."""
    device = get_backing_device(path)
    if not device:
        return None
    levels = _block_device_raid_levels(_sysfs_block_name(device))
    return levels[0] if levels else None


def _classify_dm_uuid(uuid: str) -> str:
    """
    Classify a device-mapper UUID prefix into a human-readable kind.

    DM_UUID is set by whichever userspace tool built the table (LVM,
    cryptsetup, multipath-tools, mdadm's dm-raid target) — dm itself does not
    require or validate any particular format, and a device created with a
    bare `dmsetup create` (no --uuid) has an empty uuid. Treat these prefixes
    as a best-effort label, not a detection signal: detection is the caller's
    job, based on the dm/ sysfs directory existing at all.
    """
    if uuid.startswith("LVM-"):
        return "LVM"
    if uuid.startswith("mpath-"):
        return "device-mapper multipath"
    if uuid.startswith(("CRYPT-", "crypt-")):
        return "dm-crypt"
    if uuid.startswith("RAID-"):
        return "dm-raid"
    return "device-mapper"


def _block_device_dm_info(device_name: str, seen: Optional[set[str]] = None) -> list[str]:
    """Return device-mapper kinds (e.g. LVM, dm-crypt) found at this block device or below it."""
    if seen is None:
        seen = set()
    name = os.path.basename(device_name)
    if not name or name in seen:
        return []
    seen.add(name)

    kinds: list[str] = []
    dm_dir = f"/sys/class/block/{name}/dm"
    if os.path.isdir(dm_dir):
        # dm/ is created by the device-mapper core for every dm device
        # regardless of target type or uuid content, so its presence alone
        # is proof of device-mapper; only the label below depends on uuid.
        uuid = ""
        try:
            with open(f"{dm_dir}/uuid") as fh:
                uuid = fh.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            pass
        kinds.append(_classify_dm_uuid(uuid))

    slaves_dir = f"/sys/class/block/{name}/slaves"
    try:
        slaves = os.listdir(slaves_dir)
    except (FileNotFoundError, PermissionError, OSError):
        slaves = []
    for slave in slaves:
        kinds.extend(_block_device_dm_info(slave, seen))
    return kinds


def get_dm_info(path: str) -> Optional[str]:
    """Return the device-mapper kind backing a path (e.g. LVM, dm-crypt), if any."""
    device = get_backing_device(path)
    if not device:
        return None
    kinds = _block_device_dm_info(_sysfs_block_name(device))
    return kinds[0] if kinds else None


def get_nvme_transport(path: str) -> Optional[str]:
    """
    Return the NVMe controller transport for a path, e.g. pcie, rdma, tcp, fc.

    Filesystems such as ext4/XFS can sit on top of either local PCIe NVMe or an
    NVMe-oF block device. The cufile.json P2PDMA key differs between those
    transports (`block.nvme` vs `block.nvmeof`), so mount-check needs this
    signal in addition to filesystem type.
    """
    device = get_backing_device(path)
    if not device or "nvme" not in device.lower():
        return None

    controller = _nvme_controller_from_device(device)
    if not controller:
        return None

    transport_path = f"/sys/class/nvme/{controller}/transport"
    try:
        with open(transport_path) as fh:
            return fh.read().strip().lower()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def is_nvme_backed(path: str) -> bool:
    """Return True if path resides on an NVMe device."""
    device = get_backing_device(path)
    return bool(device and _block_device_has_nvme_leaf(_sysfs_block_name(device)))


def check_ext4_data_mode(path: str) -> Optional[tuple[str, str]]:
    """
    For ext4: check the mounted journaling mode used for GDS validation.
    GDS requires ext4 to be explicitly mounted with data=ordered. The kernel's
    implicit ext4 default is not treated as sufficient because operators need a
    durable, visible mount option for GDS readiness.

    Returns (mode, mount_line), where mode is the explicit data= value or
    "default" when no data= option is present. Returns None if the mount cannot
    be read or path is not on ext4.
    """
    try:
        realpath = os.path.realpath(path)
        best_mount = ""
        best_opts = ""
        best_fstype = ""
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 4:
                    continue
                mount_point = _decode_proc_mounts_field(parts[1])
                fstype, opts = parts[2], parts[3]
                if _path_under_mount(realpath, mount_point) and len(mount_point) > len(best_mount):
                    best_mount = mount_point
                    best_fstype = fstype
                    best_opts = opts
        if best_fstype != "ext4":
            return None
        for opt in best_opts.split(","):
            if opt.startswith("data="):
                return opt[len("data="):], f"{best_mount} opts: {best_opts}"
        return "default", f"{best_mount} opts: {best_opts} (implicit ext4 default)"
    except Exception:
        return None


def check_odirect(path: str) -> tuple[Optional[bool], str]:
    """
    Probe O_DIRECT support by attempting to open a temp file with O_DIRECT.
    Returns (supported, evidence), where supported is True/False when the
    probe actually ran, or None when it could not run at all (e.g. no write
    permission) — callers must not treat None as a confirmed pass.
    """
    import tempfile
    import errno

    dirpath = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        with tempfile.NamedTemporaryFile(dir=dirpath, suffix=".gds_probe", delete=True) as tmp:
            # Write a byte so the file isn't empty
            tmp.write(b"\x00")
            tmp.flush()
            fd = os.open(tmp.name, os.O_RDONLY | getattr(os, "O_DIRECT", 0o40000))
            os.close(fd)
            return True, f"O_DIRECT open succeeded on {dirpath}"
    except OSError as exc:
        if exc.errno == errno.EINVAL:
            return False, f"O_DIRECT rejected by filesystem (EINVAL): {exc}"
        if exc.errno == errno.EACCES:
            return None, f"O_DIRECT probe skipped — no write permission in {dirpath}"
        return None, f"O_DIRECT probe inconclusive ({exc.errno}): {exc}"
    except Exception as exc:
        return None, f"O_DIRECT probe skipped: {exc}"
