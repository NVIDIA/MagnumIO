# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parse effective cuFile configuration to check GDS runtime settings.

Relevant settings:
  properties.allow_compat_mode       — allows fallback to bounce buffer (use_compat_mode is an accepted alias)
  properties.force_compat_mode       — forces POSIX-only (disables all GDS modes)
  fs.<fstype>.use_pci_p2pdma         — enables P2PDMA/C2C for a specific filesystem
  fs.<fstype>.posix_gds_min_kb       — I/O below this threshold uses POSIX
  logging.level                       — diagnostic logging verbosity
  execution.max_io_threads            — thread pool size
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Optional

from .result import CheckResult, GDSMode, Status
from .gdscheck import _find_gdscheck, _run_gdscheck_raw, _gdscheck_section

CUFILE_JSON_PATH = "/etc/cufile.json"
P2P_C2C_DIRECT_PATH_NOTE = (
    "The use_pci_p2pdma keys are also the knobs for C2C direct mode on "
    "NVIDIA Grace Hopper and Grace Blackwell ARM platforms; gdscheck "
    "may report the active route as c2c instead of p2pdma."
)


CONFIG_SCHEMA: list[dict[str, Any]] = [
    # Logging / profiling
    {"path": "logging.dir", "default": "cwd", "scope": "logging", "type": "string", "profiles": ["all"], "description": "Directory for cufile.log output."},
    {"path": "logging.level", "default": "ERROR", "scope": "logging", "type": "string", "profiles": ["all"], "description": "GDS log verbosity."},
    {"path": "profile.nvtx", "default": False, "scope": "profiling", "type": "bool", "profiles": ["all"], "description": "Enable NVTX tracing."},
    {"path": "profile.cufile_stats", "default": 0, "scope": "profiling", "type": "int", "profiles": ["all"], "description": "Enable cuFile user-level statistics."},
    {"path": "profile.io_batchsize", "default": 128, "scope": "profiling", "type": "int", "profiles": ["all"], "description": "Maximum profiling batch size."},

    # Core data path
    {"path": "properties.force_compat_mode", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Force all IO through compatibility mode."},
    {"path": "properties.allow_compat_mode", "aliases": ["properties.use_compat_mode"], "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Allow CPU/POSIX fallback when direct GDS is unavailable."},
    {"path": "properties.use_pci_p2pdma", "default": False, "scope": "data-path", "type": "bool", "profiles": ["local-nvme", "nvmeof", "nfs-rdma", "virtiofs", "raid0", "lustre", "beegfs", "scatefs", "wekafs", "gpfs"], "description": "Prefer direct P2P mode when supported by the GDS library route; this is PCI P2PDMA on x86 and C2C on supported Grace/Blackwell ARM platforms."},
    {"path": "properties.gds_rdma_write_support", "default": True, "scope": "data-path", "type": "bool", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "Enable GDS writes for RDMA-backed storage."},
    {"path": "properties.force_odirect_mode", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Force O_DIRECT mode behavior."},
    {"path": "properties.prefer_iouring", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Prefer io_uring where supported."},
    {"path": "properties.vanilla_posix_io_mode", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Allow POSIX reads/writes through cuFile without platform checks."},
    {"path": "properties.gds_fallback_io", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Allow RDMA IO to retry compat mode if nvfs IO fails."},
    {"path": "properties.io_batchsize", "default": 128, "scope": "data-path", "type": "int", "profiles": ["all"], "description": "Default cuFile batch IO size."},
    {"path": "properties.io_priority", "default": "default", "scope": "data-path", "type": "string", "profiles": ["all"], "description": "Default IO priority policy."},
    {"path": "properties.allow_rdma_token_reset", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Allow cuFile to reset RDMA tokens when the stack supports token reset."},
    {"path": "properties.compat_odirect_unaligned_read_split", "default": False, "scope": "data-path", "type": "bool", "profiles": ["all"], "description": "Split unaligned O_DIRECT reads for compatibility handling."},
    {"path": "properties.compat_odirect_unaligned_read_split_min_size_kb", "default": 128, "scope": "data-path", "type": "int", "profiles": ["all"], "description": "Minimum IO size for compat O_DIRECT unaligned-read splitting."},

    # Buffering / registration
    {"path": "properties.max_direct_io_size_kb", "default": 16384, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Maximum GDS IO chunk size."},
    {"path": "properties.max_batch_io_size", "default": 128, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Maximum batch IO size."},
    {"path": "properties.max_batch_io_timeout_msecs", "default": 5, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Maximum batch IO timeout."},
    {"path": "properties.max_device_cache_size_kb", "default": 131072, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Total internal GPU bounce buffer cache size."},
    {"path": "properties.per_buffer_cache_size_kb", "default": 1024, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Size of each internal GPU bounce buffer."},
    {"path": "properties.max_device_pinned_mem_size_kb", "default": 33554432, "scope": "buffering", "type": "int", "profiles": ["all"], "description": "Maximum per-GPU pinned memory."},
    {"path": "properties.posix_pool_slab_size_kb", "default": [4, 1024, 16384], "scope": "buffering", "type": "list", "profiles": ["compat-safe"], "description": "POSIX fallback slab sizes."},
    {"path": "properties.posix_pool_slab_count", "default": [128, 64, 64], "scope": "buffering", "type": "list", "profiles": ["compat-safe"], "description": "POSIX fallback slab counts."},
    {"path": "properties.gpu_bounce_buffer_slab_size_kb", "default": None, "scope": "buffering", "type": "list", "profiles": ["all"], "description": "Flattened GDS GPU bounce buffer slab sizes."},
    {"path": "properties.gpu_bounce_buffer_slab_count", "default": None, "scope": "buffering", "type": "list", "profiles": ["all"], "description": "Flattened GDS GPU bounce buffer slab counts."},
    {"path": "properties.gpu_bounce_buffer_slab_config", "default": None, "scope": "buffering", "type": "dict", "profiles": ["all"], "description": "GDS 1.17 per-mode GPU bounce buffer slab config."},

    # Polling / small IO
    {"path": "properties.use_poll_mode", "default": False, "scope": "small-io", "type": "bool", "profiles": ["all"], "description": "Poll for completion for small IO."},
    {"path": "properties.poll_mode_max_size_kb", "default": 4, "scope": "small-io", "type": "int", "profiles": ["all"], "description": "Maximum size for poll mode."},
    {"path": "sparse.min_p2pdma_threshold_kb", "default": 8, "scope": "small-io", "type": "int", "profiles": ["local-nvme", "nvmeof"], "description": "Minimum sparse IO size before direct P2P mode is preferred."},
    {"path": "sparse.p2p_compat_enable", "default": True, "scope": "small-io", "type": "bool", "profiles": ["local-nvme", "nvmeof"], "description": "Allow sparse IO compatibility handling for P2P paths."},

    # RDMA
    {"path": "properties.rdma_dev_addr_list", "default": [], "scope": "rdma", "type": "list", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "Client IPv4 addresses for RDMA devices."},
    {"path": "properties.rdma_peer_type", "default": "peer_mem", "scope": "rdma", "type": "string", "profiles": ["wekafs", "gpfs"], "description": "Userspace RDMA peer transport: peer_mem or dmabuf."},
    {"path": "properties.rdma_load_balancing_policy", "aliases": ["properties.rdma_peer_affinity_policy"], "default": "RoundRobin", "scope": "rdma", "type": "string", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "RDMA memory registration peer selection policy."},
    {"path": "properties.rdma_topN_ranks", "default": 1, "scope": "rdma", "type": "int", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "Number of distinct nearest GPU/NIC distance ranks considered by the RDMA policy."},
    {"path": "properties.rdma_dynamic_routing", "default": False, "scope": "rdma", "type": "bool", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "Enable dynamic routing for GPU/NIC topology."},
    {"path": "properties.rdma_dynamic_routing_order", "default": ["GPU_MEM_NVLINKS", "GPU_MEM", "SYS_MEM", "P2P"], "scope": "rdma", "type": "list", "profiles": ["lustre", "wekafs", "gpfs", "nfs-rdma"], "description": "Dynamic routing policy preference order."},
    {"path": "properties.rdma_transport_type", "default": "DC_V1", "scope": "rdma", "type": "string", "profiles": ["wekafs", "gpfs"], "description": "RDMA transport flavor used between cuFile and cuObject/filesystem consumers."},
    {"path": "properties.rdma_dc_key", "default": None, "scope": "rdma", "type": "hex32", "profiles": ["wekafs", "gpfs"], "description": "Optional 32-bit DC key for userspace RDMA."},
    {"path": "properties.rdma_access_mask", "default": None, "scope": "rdma", "type": "hex32", "profiles": ["wekafs", "gpfs"], "description": "Optional RDMA operation access mask."},

    # Filesystem overrides
    {"path": "fs.generic.posix_unaligned_writes", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["all"], "description": "Use POSIX writes for unaligned writes."},
    {"path": "fs.lustre.posix_gds_min_kb", "default": 4, "scope": "filesystem", "type": "int", "profiles": ["lustre"], "description": "Small Lustre IO threshold for POSIX path."},
    {"path": "fs.lustre.rdma_dev_addr_list", "default": [], "scope": "filesystem", "type": "list", "profiles": ["lustre"], "description": "Lustre-specific RDMA client addresses."},
    {"path": "fs.lustre.mount_table", "default": {}, "scope": "filesystem", "type": "dict", "profiles": ["lustre"], "description": "Per-mount Lustre RDMA routing table."},
    {"path": "fs.lustre.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["lustre"], "description": "Legacy/unsupported P2PDMA-looking key; the GDS library does not support P2PDMA for Lustre."},
    {"path": "fs.beegfs.posix_gds_min_kb", "default": 0, "scope": "filesystem", "type": "int", "profiles": ["beegfs"], "description": "Small BeeGFS IO threshold for POSIX path."},
    {"path": "fs.beegfs.rdma_dev_addr_list", "default": [], "scope": "filesystem", "type": "list", "profiles": ["beegfs"], "description": "BeeGFS-specific RDMA client addresses."},
    {"path": "fs.beegfs.mount_table", "default": {}, "scope": "filesystem", "type": "dict", "profiles": ["beegfs"], "description": "Per-mount BeeGFS RDMA routing table."},
    {"path": "fs.beegfs.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["beegfs"], "description": "Legacy/unsupported P2PDMA-looking key; the GDS library does not support P2PDMA for BeeGFS."},
    {"path": "fs.scatefs.posix_gds_min_kb", "default": 0, "scope": "filesystem", "type": "int", "profiles": ["scatefs"], "description": "Small ScaTeFS IO threshold for POSIX path."},
    {"path": "fs.scatefs.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["scatefs"], "description": "Legacy/unsupported P2PDMA-looking key; the GDS library does not support P2PDMA for ScaTeFS."},
    {"path": "fs.nfs.rdma_dev_addr_list", "default": [], "scope": "filesystem", "type": "list", "profiles": ["nfs-rdma"], "description": "NFS-specific RDMA client addresses."},
    {"path": "fs.nfs.mount_table", "default": {}, "scope": "filesystem", "type": "dict", "profiles": ["nfs-rdma"], "description": "Per-mount NFS RDMA routing table."},
    {"path": "fs.nfs.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": [], "description": "Legacy NFS P2PDMA-looking key; NFS GDS validation should use the NFSoRDMA path."},
    {"path": "fs.weka.rdma_write_support", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["wekafs"], "description": "Enable WekaFS RDMA writes instead of POSIX writes."},
    {"path": "fs.weka.rdma_dev_addr_list", "default": [], "scope": "filesystem", "type": "list", "profiles": ["wekafs"], "description": "WekaFS-specific RDMA client addresses."},
    {"path": "fs.weka.mount_table", "default": {}, "scope": "filesystem", "type": "dict", "profiles": ["wekafs"], "description": "Per-mount WekaFS RDMA routing table."},
    {"path": "fs.gpfs.gds_write_support", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["gpfs"], "description": "Enable GPFS GDS write support when supported by the stack."},
    {"path": "fs.gpfs.gds_async_support", "default": True, "scope": "filesystem", "type": "bool", "profiles": ["gpfs"], "description": "Enable GPFS async GDS support."},
    {"path": "fs.gpfs.rdma_dev_addr_list", "default": [], "scope": "filesystem", "type": "list", "profiles": ["gpfs"], "description": "GPFS-specific RDMA client addresses."},
    {"path": "fs.gpfs.mount_table", "default": {}, "scope": "filesystem", "type": "dict", "profiles": ["gpfs"], "description": "Per-mount GPFS RDMA routing table."},
    {"path": "fs.gpfs.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["gpfs"], "description": "Unsupported P2PDMA-looking key; the GDS library does not support P2PDMA for GPFS."},
    {"path": "fs.virtiofs.use_pci_p2pdma", "default": False, "scope": "filesystem", "type": "bool", "profiles": ["virtiofs"], "description": "Enable P2PDMA/C2C for virtiofs."},

    # Block transports
    {"path": "block.nvme.use_pci_p2pdma", "default": False, "scope": "block", "type": "bool", "profiles": ["local-nvme"], "description": "Enable P2PDMA/C2C for local NVMe."},
    {"path": "block.nvmeof.use_pci_p2pdma", "default": False, "scope": "block", "type": "bool", "profiles": ["nvmeof"], "description": "Enable P2PDMA/C2C for NVMe-oF."},
    {"path": "block.raid.use_pci_p2pdma", "default": False, "scope": "block", "type": "bool", "profiles": ["raid0"], "description": "Enable P2PDMA/C2C for RAID0 when the installed GDS stack supports that route."},
    {"path": "block.raid1.use_pci_p2pdma", "default": False, "scope": "block", "type": "bool", "profiles": [], "description": "Known cufile.json P2PDMA/C2C setting for RAID1 stacks; route support must be confirmed with the installed GDS stack."},
    {"path": "block.raid10.use_pci_p2pdma", "default": False, "scope": "block", "type": "bool", "profiles": [], "description": "Known cufile.json P2PDMA/C2C setting for RAID10 stacks; route support must be confirmed with the installed GDS stack."},

    # Administration / execution
    {"path": "denylist.drivers", "aliases": ["blacklist.drivers"], "default": [], "scope": "admin", "type": "list", "profiles": ["all"], "description": "Disable supported storage drivers on this node."},
    {"path": "denylist.devices", "aliases": ["blacklist.devices"], "default": [], "scope": "admin", "type": "list", "profiles": ["all"], "description": "Disable specific block devices on this node."},
    {"path": "denylist.mounts", "aliases": ["blacklist.mounts"], "default": [], "scope": "admin", "type": "list", "profiles": ["all"], "description": "Disable specific supported mounts on this node."},
    {"path": "denylist.filesystems", "aliases": ["blacklist.filesystems"], "default": [], "scope": "admin", "type": "list", "profiles": ["all"], "description": "Disable specific supported filesystem types on this node."},
    {"path": "miscellaneous.api_check_aggressive", "default": False, "scope": "misc", "type": "bool", "profiles": ["all"], "description": "Enable aggressive API checking."},
    {"path": "miscellaneous.skip_topology_detection", "default": False, "scope": "misc", "type": "bool", "profiles": ["compat-safe"], "description": "Skip topology detection to reduce compat-mode startup latency."},
    {"path": "miscellaneous.stream_memops_bypass", "default": False, "scope": "misc", "type": "bool", "profiles": ["all"], "description": "Bypass stream memory operations where supported."},
    {"path": "miscellaneous.enable_static_routing", "aliases": ["sparse.enable_static_routing"], "default": False, "scope": "misc", "type": "bool", "profiles": ["local-nvme", "nvmeof"], "description": "Enable static topology routing using a user-supplied topology.json file."},
    {"path": "miscellaneous.static_routing_filepath", "aliases": ["sparse.static_routing_filepath"], "default": "/etc/topology.json", "scope": "misc", "type": "string", "profiles": ["local-nvme", "nvmeof"], "description": "Path to the static topology JSON consumed when static routing is enabled."},
    {"path": "miscellaneous.rdma_token_reset_timeout_secs", "default": 30, "scope": "misc", "type": "int", "profiles": ["all"], "description": "Timeout in seconds for RDMA token reset handling."},
    {"path": "execution.max_io_queue_depth", "default": 128, "scope": "execution", "type": "int", "profiles": ["all"], "description": "Maximum pending work items in cuFile threadpool."},
    {"path": "execution.max_io_threads", "default": 4, "scope": "execution", "type": "int", "profiles": ["all"], "description": "Threadpool threads per GPU."},
    {"path": "execution.parallel_io", "default": True, "scope": "execution", "type": "bool", "profiles": ["all"], "description": "Enable threadpool parallel IO."},
    {"path": "execution.min_io_threshold_size_kb", "default": 8192, "scope": "execution", "type": "int", "profiles": ["all"], "description": "Threshold for splitting IO into parallel work."},
    {"path": "execution.max_request_parallelism", "default": 4, "scope": "execution", "type": "int", "profiles": ["all"], "description": "Maximum parallel buffers per request."},
]

ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "CUFILE_FORCE_COMPAT_MODE": ("properties.force_compat_mode", "bool"),
    "CUFILE_ALLOW_COMPAT_MODE": ("properties.allow_compat_mode", "bool"),
    "CUFILE_USE_PCIP2PDMA": ("properties.use_pci_p2pdma", "bool"),
    "CUFILE_LOGGING_LEVEL": ("logging.level", "string"),
    "CUFILE_LOGFILE_PATH": ("logging.dir", "string"),
    "CUFILE_SKIP_TOPOLOGY_DETECTION": ("miscellaneous.skip_topology_detection", "bool"),
    "CUFILE_NVTX": ("profile.nvtx", "bool"),
}

_STATUS_ORDER = {"OK": 0, "INFO": 1, "WARN": 2}
_LOG_LEVELS = {"NOTICE", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"}
_IO_PRIORITIES = {"default", "low", "med", "high"}
_RDMA_POLICIES = {"FirstFit", "MaxMinFit", "RoundRobin", "RoundRobinMaxMin", "Randomized"}
_RDMA_PEER_TYPES = {"peer_mem", "dmabuf"}
_RDMA_TRANSPORT_TYPES = {"DC_V1", "EXT_RC_V1"}
_ROUTING_POLICIES = {"GPU_MEM_NVLINKS", "GPU_MEM", "SYS_MEM", "P2P"}
_IGNORED_UNKNOWN_CONFIG_PATHS = {
    # Newer RDMA multipath/health knobs. Keep quiet until audit has
    # release-specific validation rules for these values.
    "properties.rdma_multipath_enabled",
    "properties.rdma_max_backup_devices",
    "properties.rdma_io_retry_count",
    "properties.rdma_io_retry_delay_ms",
    "properties.rdma_failback_enabled",
    "properties.rdma_failback_delay_ms",
    "properties.rdma_health_check_interval_ms",
    "properties.rdma_async_event_monitoring",
    "properties.rdma_unhealthy_threshold",
}
_GPU_BOUNCE_SLAB_LEAF_PATHS = {
    "properties.gpu_bounce_buffer_slab_config.slab_size_kb",
    "properties.gpu_bounce_buffer_slab_config.slab_count",
}


def _strip_jsonc_comments(text: str) -> str:
    """
    Strip C-style comments from JSONC text so it can be parsed by stdlib json.
    NVIDIA ships cufile.json as JSONC (JSON with Comments) — both // and /* */ forms.
    Handles comments inside strings correctly by skipping quoted regions.
    """
    result = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        c = text[i]

        # Track string boundaries (respect escaped quotes)
        if c == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
            result.append(c)
            i += 1
            continue

        if in_string:
            result.append(c)
            i += 1
            continue

        # Block comment /* ... */
        if c == '/' and i + 1 < n and text[i + 1] == '*':
            end = text.find('*/', i + 2)
            if end == -1:
                break  # unterminated block comment — stop
            i = end + 2
            continue

        # Line comment // ...
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            end = text.find('\n', i)
            i = end + 1 if end != -1 else n
            continue

        result.append(c)
        i += 1

    return ''.join(result)


def _load_cufile_json_with_path() -> tuple[Optional[dict], Optional[str]]:
    """
    Load and parse /etc/cufile.json. NVIDIA ships this file as JSONC
    (JSON with C-style comments) — strip comments before parsing.
    """
    paths = [
        os.environ.get("CUFILE_ENV_PATH_JSON", ""),
        CUFILE_JSON_PATH,
        "/usr/local/cuda/gds/cufile.json",
    ]
    for path in paths:
        if not path:
            continue
        try:
            with open(path) as fh:
                raw = fh.read()
            clean = _strip_jsonc_comments(raw)
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                return parsed, path
            return {"_parse_error": "top-level cufile.json value must be an object", "_path": path}, path
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            return {"_parse_error": str(exc), "_path": path}, path
    return None, None


def _load_cufile_json_from_path(path: str) -> tuple[Optional[dict], Optional[str]]:
    """Load an explicit cufile.json path without env/default fallback."""
    try:
        with open(path) as fh:
            raw = fh.read()
        clean = _strip_jsonc_comments(raw)
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed, path
        return {"_parse_error": "top-level cufile.json value must be an object", "_path": path}, path
    except FileNotFoundError:
        return {"_parse_error": f"file not found: {path}", "_path": path}, path
    except OSError as exc:
        return {"_parse_error": str(exc), "_path": path}, path
    except json.JSONDecodeError as exc:
        return {"_parse_error": str(exc), "_path": path}, path


def _load_cufile_json() -> Optional[dict]:
    config, _ = _load_cufile_json_with_path()
    return config



def _coerce_gdscheck_value(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if lowered in {"not set", "not configured"}:
        return ""
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?\d+(\s+[-+]?\d+)+", text):
        return [int(part) for part in text.split()]
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if text.startswith("[") and text.endswith("]"):
                inner = text[1:-1].strip()
                if not inner:
                    return []
                return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
    return text.strip("'\"")


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    node: dict[str, Any] = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _deep_merge_config(base: Optional[dict], overlay: Optional[dict]) -> Optional[dict]:
    """Return base + overlay, with overlay values taking precedence."""
    if not isinstance(base, dict) or "_parse_error" in base:
        base_copy: dict[str, Any] = {}
    else:
        base_copy = copy.deepcopy(base)
    if not isinstance(overlay, dict) or "_parse_error" in overlay:
        return base_copy or None

    def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    merge(base_copy, overlay)
    return base_copy


def _parse_gdscheck_cufile_config(output: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line in _gdscheck_section(output, "CUFILE CONFIGURATION"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        dotted = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_/-]+)+", dotted):
            continue
        _set_dotted(config, dotted, _coerce_gdscheck_value(value))
    return config


def _load_cufile_config_from_gdscheck(apply_env: bool = True) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    path = _find_gdscheck()
    if not path:
        return None, None, "gdscheck not found"

    output, error = _run_gdscheck_raw(path, apply_env=apply_env)
    if error:
        return None, path, error
    assert output is not None

    config = _parse_gdscheck_cufile_config(output)
    if not config:
        return None, path, "gdscheck -p returned no parsed cuFile configuration entries"
    return config, path, None


def _coerce_env_value(value: str, value_type: str) -> Any:
    if value_type == "bool":
        return value.strip().lower() in ("1", "true", "yes", "on")
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _append_unique(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


def _schema_known_paths() -> set[str]:
    known: set[str] = set()
    for item in CONFIG_SCHEMA:
        known.add(item["path"])
        known.update(item.get("aliases", []))
        for key in [item["path"], *item.get("aliases", [])]:
            parts = key.split(".")
            for idx in range(1, len(parts)):
                known.add(".".join(parts[:idx]))
    return known


def _iter_leaf_paths(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        if not value and prefix:
            yield prefix, value
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_paths(child, child_prefix)
        return
    yield prefix, value


def _is_mount_table_leaf(path: str) -> bool:
    for fs_name in ("beegfs", "gpfs", "lustre", "nfs", "weka"):
        if path.startswith(f"fs.{fs_name}.mount_table."):
            return True
    return False


def _is_known_nested_leaf(path: str) -> bool:
    return path in _GPU_BOUNCE_SLAB_LEAF_PATHS


def _unknown_config_entries(config: Optional[dict], known_paths: set[str], source_label: str = "file") -> list[dict[str, Any]]:
    if not isinstance(config, dict) or "_parse_error" in config:
        return []
    entries: list[dict[str, Any]] = []
    for path, value in _iter_leaf_paths(config):
        if path in _IGNORED_UNKNOWN_CONFIG_PATHS:
            continue
        if path in known_paths or _is_mount_table_leaf(path) or _is_known_nested_leaf(path):
            continue
        entries.append({
            "path": path,
            "resolved_key": path,
            "value": value,
            "default": None,
            "source": source_label,
            "status": "INFO",
            "risk": "Unknown cufile.json key for this tool's documented schema",
            "recommendation": (
                "Verify the key against the installed CUDA/GDS cufile.json template. "
                "If it is a vendor or newer-release key, treat this as informational; "
                "if it is a typo, fix or remove it."
            ),
            "detail": "",
            "scope": "unknown",
            "description": "Unrecognized key found while scanning the JSON file.",
        })
    return entries


def _type_issue(value: Any, value_type: str) -> Optional[str]:
    if value is None:
        return None
    if value_type == "bool" and not isinstance(value, bool):
        return "expected a JSON boolean true/false"
    if value_type == "int" and not isinstance(value, int):
        return "expected an integer"
    if value_type == "int" and isinstance(value, bool):
        return "expected an integer, not a boolean"
    if value_type == "string" and not isinstance(value, str):
        return "expected a string"
    if value_type == "list" and not isinstance(value, list):
        return "expected a list"
    if value_type == "dict" and not isinstance(value, dict):
        return "expected an object/dictionary"
    if value_type == "hex32":
        if not isinstance(value, str) or not re.fullmatch(r"0[xX][0-9a-fA-F]{1,8}", value):
            return "expected a 32-bit hex string such as 0xffeeddcc"
    return None


def _generic_value_issues(item: dict[str, Any], value: Any) -> list[tuple[str, str]]:
    path = item["path"]
    issues: list[tuple[str, str]] = []
    type_problem = _type_issue(value, item.get("type", ""))
    if type_problem:
        issues.append((type_problem, f"Set {path} to the documented {item.get('type')} value."))
        return issues

    if value is None:
        return issues

    if path == "logging.level" and str(value).upper() not in _LOG_LEVELS:
        issues.append(("unsupported logging level", "Use one of: NOTICE, ERROR, WARN, INFO, DEBUG, TRACE."))
    elif path == "profile.cufile_stats" and value not in (0, 1, 2, 3):
        issues.append(("cufile_stats must be 0 through 3", "Use 0 to disable stats, or 1-3 for increasing detail."))
    elif path == "properties.io_priority" and str(value) not in _IO_PRIORITIES:
        issues.append(("unsupported IO priority", "Use one of: default, low, med, high."))
    elif path == "properties.rdma_peer_type" and str(value) not in _RDMA_PEER_TYPES:
        issues.append(("unsupported RDMA peer type", "Use peer_mem or dmabuf."))
    elif path == "properties.rdma_transport_type" and str(value) not in _RDMA_TRANSPORT_TYPES:
        issues.append(("unsupported RDMA transport type", "Use DC_V1 unless the installed GDS stack explicitly documents another transport."))
    elif path == "properties.rdma_load_balancing_policy" and str(value) not in _RDMA_POLICIES:
        issues.append(("unknown cuFile RDMA load-balancing policy", "Use one of: FirstFit, MaxMinFit, RoundRobin, RoundRobinMaxMin, Randomized."))
    elif path == "properties.rdma_dynamic_routing_order":
        order = _as_list(value)
        invalid = [str(policy) for policy in order if policy not in _ROUTING_POLICIES]
        if invalid:
            issues.append((f"unknown dynamic routing policy value(s): {', '.join(invalid)}", "Use only GPU_MEM_NVLINKS, GPU_MEM, SYS_MEM, and P2P."))
        if len(order) != len(set(order)):
            issues.append(("dynamic routing order contains duplicates", "Remove duplicate policies so fallback order is unambiguous."))

    if item.get("type") == "int" and isinstance(value, int) and not isinstance(value, bool):
        if path in {
            "properties.max_direct_io_size_kb",
            "properties.max_device_cache_size_kb",
            "properties.max_device_pinned_mem_size_kb",
            "properties.poll_mode_max_size_kb",
            "properties.compat_odirect_unaligned_read_split_min_size_kb",
            "fs.lustre.posix_gds_min_kb",
            "fs.beegfs.posix_gds_min_kb",
            "fs.scatefs.posix_gds_min_kb",
            "execution.max_io_queue_depth",
            "execution.max_io_threads",
            "execution.min_io_threshold_size_kb",
            "execution.max_request_parallelism",
            "properties.io_batchsize",
            "profile.io_batchsize",
            "miscellaneous.rdma_token_reset_timeout_secs",
        } and value < 0:
            issues.append(("negative numeric value is invalid", f"Set {path} to a non-negative integer."))
        if path in {
            "execution.max_io_queue_depth",
            "execution.max_io_threads",
            "execution.max_request_parallelism",
            "properties.io_batchsize",
            "profile.io_batchsize",
        } and value < 1:
            issues.append(("value must be at least 1", f"Set {path} to an integer >= 1."))
        if path in {
            "properties.max_direct_io_size_kb",
            "properties.max_device_cache_size_kb",
            "properties.max_device_pinned_mem_size_kb",
            "properties.poll_mode_max_size_kb",
            "properties.compat_odirect_unaligned_read_split_min_size_kb",
            "fs.lustre.posix_gds_min_kb",
            "fs.beegfs.posix_gds_min_kb",
            "fs.scatefs.posix_gds_min_kb",
        } and value % 4 != 0:
            issues.append(("value should be 4 KB aligned", f"Use a multiple of 4 for {path}."))
        if path == "properties.per_buffer_cache_size_kb":
            if value < 1024 or value > 16384:
                issues.append(("per-buffer cache size should be 1024-16384 KB", "Use a value in the documented range, usually the default 1024."))
            if value % 64 != 0:
                issues.append(("per-buffer cache size should be a multiple of 64 KB", "Use a 64 KB aligned value."))
        if path == "properties.rdma_topN_ranks" and value < 1:
            issues.append(("invalid K-nearest RDMA rank count", "Set properties.rdma_topN_ranks to an integer >= 1."))

    if path in (
        "properties.posix_pool_slab_size_kb",
        "properties.posix_pool_slab_count",
        "properties.gpu_bounce_buffer_slab_size_kb",
        "properties.gpu_bounce_buffer_slab_count",
    ) and isinstance(value, list):
        bad = [repr(item_value) for item_value in value if not _is_positive_int(item_value)]
        if bad:
            issues.append((f"list contains non-positive/non-integer entries: {', '.join(bad)}", f"Use positive integer entries for {path}."))
        elif path == "properties.gpu_bounce_buffer_slab_size_kb":
            bad_alignment = [repr(size) for size in value if size % 4 != 0]
            if bad_alignment:
                issues.append((f"slab sizes are not 4 KB aligned: {', '.join(bad_alignment)}", "Use positive 4 KB aligned slab sizes."))
            elif value != sorted(value):
                issues.append(("slab_size_kb entries must be in ascending order", "Sort slab_size_kb from smallest to largest and keep slab_count aligned with the same entries."))
    if path == "properties.gpu_bounce_buffer_slab_config" and isinstance(value, dict):
        sizes = value.get("slab_size_kb")
        counts = value.get("slab_count")
        if sizes is None or counts is None:
            issues.append(("slab config must include slab_size_kb and slab_count", "Use {'slab_size_kb': [...], 'slab_count': [...]} or remove the key."))
        elif not isinstance(sizes, list) or not isinstance(counts, list):
            issues.append(("slab_size_kb and slab_count must both be lists", "Use list values for both slab_size_kb and slab_count."))
        elif len(sizes) != len(counts):
            issues.append(("slab_size_kb and slab_count lengths differ", "Keep slab_size_kb and slab_count lists the same length."))
        else:
            bad_sizes = [repr(size) for size in sizes if not _is_positive_int(size) or size % 4 != 0]
            bad_counts = [repr(count) for count in counts if not _is_positive_int(count)]
            if bad_sizes:
                issues.append((f"invalid slab_size_kb entries: {', '.join(bad_sizes)}", "Use positive 4 KB aligned slab sizes."))
            elif sizes != sorted(sizes):
                issues.append(("slab_size_kb entries must be in ascending order", "Sort slab_size_kb from smallest to largest and keep slab_count aligned with the same entries."))
            if bad_counts:
                issues.append((f"invalid slab_count entries: {', '.join(bad_counts)}", "Use positive integer slab counts."))

    return issues


def _make_synthetic_entry(path: str, status: str, risk: str, recommendation: str, value: Any = None, detail: str = "") -> dict[str, Any]:
    return {
        "path": path,
        "resolved_key": path,
        "value": value,
        "default": None,
        "source": "derived",
        "status": status,
        "risk": risk,
        "recommendation": recommendation,
        "detail": detail,
        "scope": "cross-field",
        "description": "Derived validation from multiple cufile.json settings.",
    }


def _has_gpu_bounce_buffer_slab_config(props: dict[str, Any]) -> bool:
    nested = props.get("gpu_bounce_buffer_slab_config")
    if isinstance(nested, dict) and (
        "slab_size_kb" in nested or "slab_count" in nested
    ):
        return True
    return (
        "gpu_bounce_buffer_slab_size_kb" in props
        or "gpu_bounce_buffer_slab_count" in props
    )


def _append_cross_field_entries(config: Optional[dict], entries: list[dict[str, Any]]) -> None:
    if not isinstance(config, dict) or "_parse_error" in config:
        return

    props = config.get("properties", {}) if isinstance(config.get("properties"), dict) else {}
    cache = props.get("max_device_cache_size_kb")
    per_buffer = props.get("per_buffer_cache_size_kb")
    batch = props.get("io_batchsize")
    if (
        not _has_gpu_bounce_buffer_slab_config(props)
        and _is_positive_int(cache)
        and _is_positive_int(per_buffer)
        and _is_positive_int(batch)
    ):
        if cache // per_buffer < batch:
            entries.append(_make_synthetic_entry(
                "properties.max_device_cache_size_kb/properties.per_buffer_cache_size_kb/properties.io_batchsize",
                "WARN",
                "Device bounce-buffer cache cannot satisfy the configured cuFile batch size",
                "Increase max_device_cache_size_kb, lower per_buffer_cache_size_kb, or lower properties.io_batchsize so cache/per_buffer >= io_batchsize.",
                {"max_device_cache_size_kb": cache, "per_buffer_cache_size_kb": per_buffer, "io_batchsize": batch},
            ))

    sizes = props.get("posix_pool_slab_size_kb")
    counts = props.get("posix_pool_slab_count")
    if isinstance(sizes, list) and isinstance(counts, list) and len(sizes) != len(counts):
        entries.append(_make_synthetic_entry(
            "properties.posix_pool_slab_size_kb/properties.posix_pool_slab_count",
            "WARN",
            "POSIX slab size and count arrays have different lengths",
            "Keep posix_pool_slab_size_kb and posix_pool_slab_count arrays one-to-one.",
            {"posix_pool_slab_size_kb": sizes, "posix_pool_slab_count": counts},
        ))

    gpu_slab_sizes = props.get("gpu_bounce_buffer_slab_size_kb")
    gpu_slab_counts = props.get("gpu_bounce_buffer_slab_count")
    has_gpu_slab_sizes = "gpu_bounce_buffer_slab_size_kb" in props
    has_gpu_slab_counts = "gpu_bounce_buffer_slab_count" in props
    if has_gpu_slab_sizes != has_gpu_slab_counts:
        entries.append(_make_synthetic_entry(
            "properties.gpu_bounce_buffer_slab_size_kb/properties.gpu_bounce_buffer_slab_count",
            "WARN",
            "GPU bounce-buffer slab config is incomplete",
            "Set both gpu_bounce_buffer_slab_size_kb and gpu_bounce_buffer_slab_count, or remove the partial config.",
            {"gpu_bounce_buffer_slab_size_kb": gpu_slab_sizes, "gpu_bounce_buffer_slab_count": gpu_slab_counts},
        ))
    elif isinstance(gpu_slab_sizes, list) and isinstance(gpu_slab_counts, list) and len(gpu_slab_sizes) != len(gpu_slab_counts):
        entries.append(_make_synthetic_entry(
            "properties.gpu_bounce_buffer_slab_size_kb/properties.gpu_bounce_buffer_slab_count",
            "WARN",
            "GPU bounce-buffer slab size and count arrays have different lengths",
            "Keep gpu_bounce_buffer_slab_size_kb and gpu_bounce_buffer_slab_count arrays one-to-one.",
            {"gpu_bounce_buffer_slab_size_kb": gpu_slab_sizes, "gpu_bounce_buffer_slab_count": gpu_slab_counts},
        ))

    fs_cfg = config.get("fs", {}) if isinstance(config.get("fs"), dict) else {}
    for fs_name in ("beegfs", "gpfs", "lustre", "nfs", "weka"):
        section = fs_cfg.get(fs_name, {})
        if not isinstance(section, dict):
            continue
        addr_list = section.get("rdma_dev_addr_list")
        mount_table = section.get("mount_table")
        if addr_list and mount_table:
            entries.append(_make_synthetic_entry(
                f"fs.{fs_name}.rdma_dev_addr_list/fs.{fs_name}.mount_table",
                "WARN",
                "Per-filesystem RDMA addresses and per-mount RDMA table are both configured",
                "Use fs.<name>.rdma_dev_addr_list for a single mount, or fs.<name>.mount_table for multiple mounts. The GDS documentation treats configuring both as a config error.",
                {"rdma_dev_addr_list": addr_list, "mount_table": mount_table},
            ))
        if isinstance(mount_table, dict):
            for mount, mount_cfg in mount_table.items():
                if not isinstance(mount_cfg, dict):
                    entries.append(_make_synthetic_entry(
                        f"fs.{fs_name}.mount_table.{mount}",
                        "WARN",
                        "Mount table entry must be an object",
                        "Use {'rdma_dev_addr_list': ['<client-rdma-ip>']} for each mount path.",
                        mount_cfg,
                    ))
                    continue
                mount_addrs = mount_cfg.get("rdma_dev_addr_list")
                if not isinstance(mount_addrs, list):
                    entries.append(_make_synthetic_entry(
                        f"fs.{fs_name}.mount_table.{mount}.rdma_dev_addr_list",
                        "WARN",
                        "Mount table rdma_dev_addr_list must be a list",
                        "Set rdma_dev_addr_list to a list of client-side RDMA IPv4 addresses for this mount.",
                        mount_addrs,
                    ))

    if props.get("rdma_dynamic_routing") is True:
        has_route_ips = bool(props.get("rdma_dev_addr_list"))
        for section in fs_cfg.values():
            if isinstance(section, dict):
                has_route_ips = has_route_ips or bool(section.get("rdma_dev_addr_list")) or bool(section.get("mount_table"))
        if not has_route_ips:
            entries.append(_make_synthetic_entry(
                "properties.rdma_dynamic_routing",
                "WARN",
                "Dynamic routing is enabled without RDMA address configuration",
                "Provide client RDMA IPs globally in properties.rdma_dev_addr_list or in the filesystem mount_table so cuFile can map mounts to NICs.",
                True,
            ))

    misc = config.get("miscellaneous", {}) if isinstance(config.get("miscellaneous"), dict) else {}
    sparse = config.get("sparse", {}) if isinstance(config.get("sparse"), dict) else {}
    static_routing_enabled = misc.get("enable_static_routing", sparse.get("enable_static_routing"))
    static_routing_filepath = misc.get("static_routing_filepath", sparse.get("static_routing_filepath"))
    static_routing_path_name = (
        "miscellaneous.static_routing_filepath"
        if "static_routing_filepath" in misc or "enable_static_routing" in misc
        else "sparse.static_routing_filepath"
    )
    static_routing_enable_name = (
        "miscellaneous.enable_static_routing"
        if "enable_static_routing" in misc
        else "sparse.enable_static_routing"
    )
    if static_routing_enabled is True:
        filepath = static_routing_filepath
        if not isinstance(filepath, str) or not filepath:
            entries.append(_make_synthetic_entry(
                static_routing_path_name,
                "WARN",
                "Static routing is enabled without a valid topology file path",
                "Set miscellaneous.static_routing_filepath to the topology JSON path, for example /etc/topology.json.",
                filepath,
            ))
        else:
            invalid_reason = ""
            try:
                if not os.path.isfile(filepath):
                    invalid_reason = "Static routing topology file is missing or not a regular file"
                elif os.path.getsize(filepath) <= 0:
                    invalid_reason = "Static routing topology file is empty"
                elif not os.access(filepath, os.R_OK):
                    invalid_reason = "Static routing topology file is unreadable"
            except OSError:
                invalid_reason = "Static routing topology file is missing, empty, or unreadable"
            if invalid_reason:
                entries.append(_make_synthetic_entry(
                    static_routing_path_name,
                    "WARN",
                    invalid_reason,
                    f"Create a readable, non-empty topology JSON file at the configured path or disable {static_routing_enable_name}.",
                    filepath,
                ))


def _get_dotted(config: Optional[dict], dotted: str) -> tuple[bool, Any]:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _rdma_list_has_entries(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _rdma_list_is_empty(value: Any) -> bool:
    return value is None or not _rdma_list_has_entries(value)


def _profile_has_rdma_address_source(config: Optional[dict], profile: Optional[str]) -> bool:
    if not isinstance(config, dict) or "_parse_error" in config:
        return False

    props = config.get("properties", {}) if isinstance(config.get("properties"), dict) else {}
    if _rdma_list_has_entries(props.get("rdma_dev_addr_list")):
        return True

    fs_name = {
        "gpfs": "gpfs",
        "lustre": "lustre",
        "nfs-rdma": "nfs",
        "wekafs": "weka",
    }.get(profile or "")
    if not fs_name:
        return False

    fs_cfg = config.get("fs", {}) if isinstance(config.get("fs"), dict) else {}
    section = fs_cfg.get(fs_name, {}) if isinstance(fs_cfg.get(fs_name), dict) else {}
    return _rdma_list_has_entries(section.get("rdma_dev_addr_list")) or bool(section.get("mount_table"))


def _gpu_memory_totals_kb() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    totals: list[int] = []
    for line in result.stdout.splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        try:
            totals.append(int(token) * 1024)
        except ValueError:
            continue
    return totals


def _normalize_gdscheck_schema_value(item: dict[str, Any], value: Any, source: str) -> Any:
    if source != "gdscheck":
        return value

    value_type = item.get("type")
    if value_type == "bool":
        if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip() in ("0", "1"):
            return value.strip() == "1"

    if value_type == "list" and isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [part.strip().strip("'\"") for part in re.split(r"[\s,]+", text) if part.strip()]

    return value


def _value_from_schema(config: Optional[dict], entry: dict[str, Any], source_label: str = "file") -> tuple[str, Any, str]:
    """Return (source, value, resolved_key) for a schema entry."""
    if config is None or "_parse_error" in (config or {}):
        return "default", entry.get("default"), entry["path"]

    for key in [entry["path"], *entry.get("aliases", [])]:
        found, value = _get_dotted(config, key)
        if found:
            return source_label, value, key

    return "default", entry.get("default"), entry["path"]


def _env_override_map() -> dict[str, tuple[Any, str]]:
    overrides: dict[str, tuple[Any, str]] = {}
    for env, (path, typ) in ENV_OVERRIDES.items():
        if env in os.environ:
            overrides[path] = (_coerce_env_value(os.environ[env], typ), env)
    return overrides


def audit_config(
    profile: Optional[str] = None,
    config_path: Optional[str] = None,
    apply_env: bool = True,
    prefer_gdscheck: bool = True,
) -> dict[str, Any]:
    """
    Return a structured, effective cuFile configuration audit.

    This is intentionally data-first so subcommands can render text or JSON
    without re-parsing /etc/cufile.json.
    """
    source_label = "file"
    config_source = "file"
    config_source_detail = "cufile.json search path"
    gdscheck_path = None
    gdscheck_error = None
    gdscheck_fallback = False
    file_fallback_config = None
    file_fallback_path = None
    file_fallback_error = None

    if config_path:
        config, path = _load_cufile_json_from_path(config_path)
        config_source_detail = "--config file"
    else:
        if prefer_gdscheck:
            try:
                config, gdscheck_path, gdscheck_error = _load_cufile_config_from_gdscheck(apply_env=apply_env)
            except TypeError:
                # Unit tests and small embedding shims may monkeypatch the loader
                # with the previous zero-argument signature.
                config, gdscheck_path, gdscheck_error = _load_cufile_config_from_gdscheck()
            if config is not None:
                path = None
                source_label = "gdscheck"
                config_source = "gdscheck"
                config_source_detail = "gdscheck -p CUFILE CONFIGURATION"
                file_fallback_config, file_fallback_path = _load_cufile_json_with_path()
                if isinstance(file_fallback_config, dict) and "_parse_error" in file_fallback_config:
                    file_fallback_error = file_fallback_config.get("_parse_error") or "parse error"
            else:
                gdscheck_fallback = True
                config, path = _load_cufile_json_with_path()
                config_source = "file" if path else "defaults"
                config_source_detail = "cufile.json fallback after gdscheck"
        else:
            config, path = _load_cufile_json_with_path()
            config_source = "file" if path else "defaults"
    env_overrides = _env_override_map() if apply_env else {}
    entries: list[dict[str, Any]] = []
    known_paths = _schema_known_paths()
    gpu_memory_totals_kb: Optional[list[int]] = None
    effective_config = (
        _deep_merge_config(file_fallback_config, config)
        if config_source == "gdscheck"
        else copy.deepcopy(config) if isinstance(config, dict) else config
    )

    for item in CONFIG_SCHEMA:
        profiles = item.get("profiles", ["all"])
        if profile and "all" not in profiles and profile not in profiles:
            continue

        source, value, resolved_key = _value_from_schema(config, item, source_label=source_label)
        if source == "default" and config_source == "gdscheck" and file_fallback_config is not None:
            fallback_source, fallback_value, fallback_resolved_key = _value_from_schema(
                file_fallback_config,
                item,
                source_label="file fallback",
            )
            if fallback_source != "default":
                source = fallback_source
                value = fallback_value
                resolved_key = fallback_resolved_key
        value = _normalize_gdscheck_schema_value(item, value, source)
        if item["path"] in env_overrides:
            value, env_name = env_overrides[item["path"]]
            source = f"env:{env_name}"
            if isinstance(effective_config, dict) and "_parse_error" not in effective_config:
                _set_dotted(effective_config, item["path"], value)

        status = "OK"
        recommendation = ""
        risk = ""
        detail = ""

        def mark(new_status: str, new_risk: str, new_recommendation: str = "", new_detail: str = "") -> None:
            nonlocal status, risk, recommendation, detail
            if _STATUS_ORDER.get(new_status, 0) > _STATUS_ORDER.get(status, 0):
                status = new_status
            risk = _append_unique(risk, new_risk)
            recommendation = _append_unique(recommendation, new_recommendation)
            detail = _append_unique(detail, new_detail)

        if source not in ("default",) or value is not None:
            for issue, action in _generic_value_issues(item, value):
                mark("WARN", issue, action)

        if item["path"] == "properties.force_compat_mode" and value is True:
            mark("WARN", "GDS acceleration disabled", "Unset CUFILE_FORCE_COMPAT_MODE or set properties.force_compat_mode=false.")
        elif item["path"] == "properties.allow_compat_mode" and value is False:
            mark("WARN", "No CPU fallback if direct GDS is unavailable", "Set allow_compat_mode/use_compat_mode=true unless failures are intentional.")
        elif item["path"] == "properties.use_pci_p2pdma" and value is True and profile in (
            "lustre", "beegfs", "scatefs", "wekafs", "gpfs", "nfs-rdma",
        ):
            mark("WARN", "P2PDMA/C2C preference is enabled, but this profile is not a GDS library-supported direct P2P route", "Supported direct P2P routes are NVMe, NVMe-oF, virtiofs, and RAID0.")
        elif item["path"].endswith("use_pci_p2pdma") and value is True and item["path"] in (
            "fs.lustre.use_pci_p2pdma", "fs.beegfs.use_pci_p2pdma",
            "fs.scatefs.use_pci_p2pdma", "fs.gpfs.use_pci_p2pdma",
        ):
            mark("WARN", "JSON enables a P2PDMA/C2C-looking key, but the GDS library does not support direct P2P for this filesystem", "Use one of the supported direct P2P routes: NVMe, NVMe-oF, virtiofs, or RAID0.")
        elif item["path"] == "fs.weka.rdma_write_support" and value is False and profile == "wekafs":
            mark("INFO", "WekaFS writes use POSIX fallback", "Enable only if the WekaFS deployment supports RDMA writes.")
        elif item["path"] == "fs.gpfs.gds_write_support" and value is False and profile == "gpfs":
            mark(
                "INFO",
                "GPFS GDS writes are disabled",
                "Set fs.gpfs.gds_write_support=true if this GPFS deployment supports GDS writes; otherwise writes may use POSIX/compat fallback.",
            )
        elif item["path"] == "fs.gpfs.gds_async_support" and value is False and profile == "gpfs":
            mark(
                "INFO",
                "GPFS async GDS support is disabled",
                "Set fs.gpfs.gds_async_support=true if applications use cuFile async APIs and the GPFS stack supports async GDS.",
            )
        elif item["path"] == "properties.max_direct_io_size_kb" and isinstance(value, int) and value < 16384:
            mark(
                "INFO",
                "Maximum direct IO size is below 16 MiB and may limit large-IO throughput",
                "Use at least 16384 KB unless a workload-specific reason requires a smaller cap. If you expected the shipped cufile.json default, verify the intended config file is being loaded.",
            )
        elif item["path"] == "properties.max_device_pinned_mem_size_kb" and isinstance(value, int):
            if gpu_memory_totals_kb is None:
                gpu_memory_totals_kb = _gpu_memory_totals_kb()
            larger_gpus = [total for total in gpu_memory_totals_kb if value < total]
            if larger_gpus:
                max_gpu_kb = max(larger_gpus)
                mark(
                    "INFO",
                    "Maximum device pinned-memory cap is below available GPU memory",
                    (
                        "Increase properties.max_device_pinned_mem_size_kb if workloads "
                        "need to register buffers larger than the current cap."
                    ),
                    f"Configured cap {value} KB; largest observed GPU memory {max_gpu_kb} KB.",
                )
        elif item["path"] == "execution.parallel_io" and value is False:
            mark("WARN", "cuFile threadpool parallel IO is disabled", "Set execution.parallel_io=true for parallel request processing unless intentionally debugging or limiting concurrency.")
        elif item["path"] == "execution.max_request_parallelism" and value == 0:
            mark("WARN", "cuFile request parallelism is disabled", "Set execution.max_request_parallelism to a positive value, commonly 4, to allow request-level parallelism.")
        elif item["path"].endswith("rdma_dev_addr_list") and profile in ("lustre", "wekafs", "gpfs", "nfs-rdma"):
            if _rdma_list_is_empty(value):
                profile_rdma_path = {
                    "gpfs": "fs.gpfs.rdma_dev_addr_list",
                    "wekafs": "fs.weka.rdma_dev_addr_list",
                }.get(profile or "")
                has_rdma_source = (
                    _profile_has_rdma_address_source(effective_config, profile)
                )
                if profile in ("wekafs", "gpfs"):
                    if item["path"] == "properties.rdma_dev_addr_list" and source != "default":
                        risk = (
                            "properties.rdma_dev_addr_list is empty; this profile is relying on a per-filesystem or per-mount RDMA address source"
                            if has_rdma_source
                            else "properties.rdma_dev_addr_list is empty; cuFile has no global RDMA client address source"
                        )
                        mark(
                            "INFO",
                            risk,
                            "Set RDMA client IPv4 addresses globally, or keep the per-filesystem rdma_dev_addr_list/mount_table populated for this profile.",
                        )
                    elif item["path"] == profile_rdma_path and not has_rdma_source:
                        mark(
                            "INFO",
                            f"{profile_rdma_path} is empty; cuFile has no explicit RDMA client address source for this profile",
                            "Set RDMA client IPv4 addresses in the per-filesystem rdma_dev_addr_list or mount_table so cuFile can map storage mounts to NICs.",
                        )
            else:
                try:
                    from .rdma import validate_rdma_client_addresses
                    rdma_addr_check = validate_rdma_client_addresses(value)
                    rdma_issues = []
                    if rdma_addr_check["invalid"]:
                        rdma_issues.append(f"invalid/non-IPv4 entries: {', '.join(rdma_addr_check['invalid'])}")
                    if rdma_addr_check["nonlocal"]:
                        rdma_issues.append(f"not local to this client: {', '.join(rdma_addr_check['nonlocal'])}")
                    if rdma_addr_check["non_rdma_iface"]:
                        rdma_issues.append(f"local but not on an RDMA netdev: {', '.join(rdma_addr_check['non_rdma_iface'])}")
                    if rdma_issues:
                        mark("WARN",
                            "rdma_dev_addr_list should contain client-side RDMA NIC IPv4 addresses",
                            "Use IPs assigned to this host's IB/RoCE interfaces, not storage server IPs: "
                            + "; ".join(rdma_issues)
                        )
                    elif not rdma_addr_check["verification_available"]:
                        mark("INFO", "Unable to verify configured RDMA IPs against local client interfaces", "Verify manually with: ip -o -4 addr show")
                    else:
                        verified_detail = (
                            "Verified client-local RDMA IP(s): "
                            f"{', '.join(rdma_addr_check['matched'])}"
                        )
                        if rdma_addr_check.get("rdma_netdevs"):
                            verified_detail += f"; RDMA netdevs: {', '.join(rdma_addr_check['rdma_netdevs'])}"
                        mark("OK", "", "", verified_detail)
                except Exception:
                    pass
        elif item["path"] == "logging.level" and str(value).upper() in ("ERROR", "WARN"):
            mark("INFO", "Production logging level", "ERROR/WARN is appropriate for production use. Consider CUFILE_LOGGING_LEVEL=DEBUG or TRACE when troubleshooting.")

        entries.append({
            "path": item["path"],
            "resolved_key": resolved_key,
            "value": value,
            "default": item.get("default"),
            "source": source,
            "status": status,
            "risk": risk,
            "recommendation": recommendation,
            "detail": detail,
            "scope": item.get("scope"),
            "description": item.get("description"),
        })

    if file_fallback_error:
        entries.append(_make_synthetic_entry(
            "file_fallback.cufile_json",
            "WARN",
            "File fallback cufile.json could not be parsed",
            "Fix the fallback cufile.json syntax so keys omitted from gdscheck can be audited from the file source.",
            file_fallback_path,
            file_fallback_error,
        ))

    unknown_source_label = (
        "effective"
        if config_source == "gdscheck" and isinstance(file_fallback_config, dict) and "_parse_error" not in file_fallback_config
        else source_label
    )
    entries.extend(_unknown_config_entries(effective_config, known_paths, source_label=unknown_source_label))
    _append_cross_field_entries(effective_config, entries)

    # Cross-field P2PDMA/C2C consistency. NVIDIA requires both the global
    # properties.use_pci_p2pdma preference and the per-transport/per-FS key.
    # The key names say "pci_p2pdma", but on GH/GB ARM platforms the same
    # knobs enable the coherent C2C direct path and gdscheck reports "c2c".
    by_path = {entry["path"]: entry for entry in entries}
    p2p_profile_key = {
        "local-nvme": "block.nvme.use_pci_p2pdma",
        "nvmeof": "block.nvmeof.use_pci_p2pdma",
        "virtiofs": "fs.virtiofs.use_pci_p2pdma",
        "raid0": "block.raid.use_pci_p2pdma",
    }.get(profile or "")
    if p2p_profile_key:
        global_entry = by_path.get("properties.use_pci_p2pdma")
        scoped_entry = by_path.get(p2p_profile_key)
        if global_entry and scoped_entry:
            def merge_entry(entry: dict[str, Any], new_status: str, new_risk: str, new_recommendation: str) -> None:
                if _STATUS_ORDER.get(new_status, 0) > _STATUS_ORDER.get(entry.get("status", "OK"), 0):
                    entry["status"] = new_status
                entry["risk"] = _append_unique(entry.get("risk", ""), new_risk)
                entry["recommendation"] = _append_unique(entry.get("recommendation", ""), new_recommendation)

            global_on = global_entry["value"] is True
            scoped_on = scoped_entry["value"] is True
            route_note = {
                "local-nvme": "Local NVMe P2PDMA/C2C also requires NVMe multipath to be disabled unless the host has a specialized multipath patch.",
                "nvmeof": "NVMe-oF P2PDMA/C2C uses block.nvmeof.use_pci_p2pdma and must be confirmed with gdscheck or workload evidence.",
                "nfs-rdma": "NFS P2PDMA/C2C is route-limited; confirm the active route with gdscheck or workload evidence.",
                "virtiofs": "virtiofs direct GDS depends on the P2PDMA/C2C route, so confirm with gdscheck or workload evidence.",
                "raid0": "RAID0 P2PDMA/C2C is Grace-configured support and must be confirmed with gdscheck or workload evidence.",
            }.get(profile or "", "Confirm active P2PDMA/C2C with gdscheck or workload evidence.")
            if not global_on and not scoped_on:
                merge_entry(
                    global_entry,
                    "INFO",
                    "P2PDMA/C2C is not globally enabled for this direct-route-capable profile",
                    (
                        "Set properties.use_pci_p2pdma=true or CUFILE_USE_PCIP2PDMA=true "
                        f"if direct P2P mode is intended. {route_note} {P2P_C2C_DIRECT_PATH_NOTE}"
                    ),
                )
                merge_entry(
                    scoped_entry,
                    "INFO",
                    f"P2PDMA/C2C is not enabled for {p2p_profile_key}",
                    (
                        f"Set {p2p_profile_key}=true if direct P2P mode is intended. "
                        "Otherwise GDS may use nvidia-fs/nvfs when available, or compat mode when enabled. "
                        f"{P2P_C2C_DIRECT_PATH_NOTE}"
                    ),
                )
            elif global_on and not scoped_on:
                merge_entry(
                    scoped_entry,
                    "WARN",
                    "P2PDMA/C2C globally preferred but not enabled for this storage path",
                    (
                        f"Set {p2p_profile_key}=true if direct P2P mode is intended. "
                        f"Otherwise GDS will use the nvidia-fs/nvfs path when available. {route_note} "
                        f"{P2P_C2C_DIRECT_PATH_NOTE}"
                    ),
                )
            elif scoped_on and not global_on:
                merge_entry(
                    global_entry,
                    "WARN",
                    "Per-path P2PDMA/C2C is enabled but the global direct P2P preference is off",
                    (
                        "Set properties.use_pci_p2pdma=true or CUFILE_USE_PCIP2PDMA=true "
                        f"if direct P2P mode is intended. {route_note} {P2P_C2C_DIRECT_PATH_NOTE}"
                    ),
                )

    if profile == "raid0":
        global_entry = by_path.get("properties.use_pci_p2pdma")
        raid_entry = by_path.get("block.raid.use_pci_p2pdma")
        raid_p2p_intended = any(
            entry and entry["value"] is True for entry in (global_entry, raid_entry)
        )
        if raid_entry and raid_p2p_intended:
            release_risk = "Release-note limitation for NVMe P2PDMA/C2C with RAID0 or multipath NVMe"
            release_recommendation = (
                "Do not rely on JSON settings alone for RAID0 P2PDMA/C2C. "
                "RAID0 P2PDMA requires NVIDIA Grace or Linux kernel >= 7.1. "
                "Confirm the route with gdscheck/runtime testing; use nvidia-fs/nvfs "
                "or compat mode when NVMe P2PDMA/C2C is unavailable for RAID0."
            )
            if raid_entry["status"] == "OK":
                raid_entry["status"] = "WARN"
                raid_entry["risk"] = release_risk
                raid_entry["recommendation"] = release_recommendation
            else:
                if release_risk not in raid_entry["risk"]:
                    raid_entry["risk"] = (
                        f"{raid_entry['risk']}; {release_risk}"
                        if raid_entry["risk"] else release_risk
                    )
                if release_recommendation not in raid_entry["recommendation"]:
                    raid_entry["recommendation"] = (
                        f"{raid_entry['recommendation']} {release_recommendation}"
                        if raid_entry["recommendation"] else release_recommendation
                    )

    return {
        "config_path": path,
        "config_source": config_source,
        "config_source_detail": config_source_detail,
        "requested_config_path": config_path,
        "env_config_path": os.environ.get("CUFILE_ENV_PATH_JSON"),
        "env_overrides_applied": apply_env,
        "gdscheck_path": gdscheck_path,
        "gdscheck_error": gdscheck_error,
        "gdscheck_fallback": gdscheck_fallback,
        "file_fallback_path": file_fallback_path,
        "file_fallback_error": file_fallback_error,
        "parse_error": config.get("_parse_error") if isinstance(config, dict) and "_parse_error" in config else None,
        "profile": profile,
        "entries": entries,
    }


def check_force_compat_mode(config: dict) -> Optional[CheckResult]:
    """
    force_compat_mode: true means GDS acceleration is bypassed — all I/O goes through
    CPU bounce buffers. Compat mode itself IS active (not broken), but Native/P2PDMA
    are effectively disabled even if the hardware is fully capable.
    Reported as a WARN on the GDS acceleration modes, not the compat mode.
    """
    # Environment variable takes precedence over the JSON file
    env_val = os.environ.get("CUFILE_FORCE_COMPAT_MODE", "").lower()
    if env_val in ("1", "true", "yes"):
        return CheckResult(
            check="CUFILE_FORCE_COMPAT_MODE env var", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                "CUFILE_FORCE_COMPAT_MODE is set in the environment. "
                "All I/O is routed through CPU bounce buffers — GDS acceleration disabled."
            ),
            mitigation="Unset the variable: unset CUFILE_FORCE_COMPAT_MODE",
            evidence=f"CUFILE_FORCE_COMPAT_MODE={os.environ.get('CUFILE_FORCE_COMPAT_MODE')}",
        )

    props = config.get("properties", {})
    if props.get("force_compat_mode", False) is True:
        return CheckResult(
            check="cufile.json force_compat_mode", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                "'properties.force_compat_mode': true in cufile.json. "
                "All I/O is routed through CPU bounce buffers — GDS acceleration is disabled "
                "even if all hardware requirements are met."
            ),
            mitigation=(
                "Set 'force_compat_mode': false in /etc/cufile.json (or remove the key — "
                "default is false) to re-enable GDS acceleration.\n"
                "Also check: CUFILE_FORCE_COMPAT_MODE env var (overrides the file)."
            ),
            evidence="cufile.json properties.force_compat_mode = true",
        )
    return None


def check_allow_compat_mode(config: Optional[dict]) -> CheckResult:
    """
    allow_compat_mode: false disables the CPU bounce buffer fallback entirely.
    When all GDS modes are unavailable (e.g. tmpfs, overlayfs, unsupported FS),
    this means ALL I/O via cuFile will fail — there is no fallback path.
    """
    # Environment variable takes precedence
    env_val = os.environ.get("CUFILE_FORCE_COMPAT_MODE", "").lower()
    if env_val in ("1", "true", "yes"):
        # force_compat_mode env var also implies compat is allowed
        return CheckResult(
            check="Compat mode availability", mode=GDSMode.COMPAT, status=Status.PASS,
            why="CUFILE_FORCE_COMPAT_MODE is set — compat mode is active (forced by env var).",
        )

    allow_env = os.environ.get("CUFILE_ALLOW_COMPAT_MODE", "").lower()
    if allow_env in ("1", "true", "yes"):
        return CheckResult(
            check="Compat mode availability",
            mode=GDSMode.COMPAT,
            status=Status.PASS,
            why="CUFILE_ALLOW_COMPAT_MODE enables compatibility fallback.",
            evidence=f"CUFILE_ALLOW_COMPAT_MODE={os.environ.get('CUFILE_ALLOW_COMPAT_MODE')}",
        )
    if allow_env in ("0", "false", "no"):
        return CheckResult(
            check="Compat mode availability",
            mode=GDSMode.COMPAT,
            status=Status.WARN,
            why=(
                "CUFILE_ALLOW_COMPAT_MODE disables compatibility fallback. This is often set "
                "intentionally so a missing GDS/RDMA path fails explicitly instead of silently "
                "falling back. If GDS modes are unavailable for this filesystem (e.g. tmpfs, "
                "overlayfs), cuFile I/O will return an error instead of falling back."
            ),
            mitigation="Unset CUFILE_ALLOW_COMPAT_MODE or set it to true if CPU/POSIX fallback should be available.",
            evidence=f"CUFILE_ALLOW_COMPAT_MODE={os.environ.get('CUFILE_ALLOW_COMPAT_MODE')}",
        )

    if config is None:
        # Current NVIDIA cufile.json documents allow_compat_mode defaulting to false.
        return CheckResult(
            check="Compat mode availability", mode=GDSMode.COMPAT, status=Status.WARN,
            why="No cufile.json found; cannot verify whether compatibility fallback is enabled.",
            mitigation=(
                "Install or point CUFILE_ENV_PATH_JSON at a cufile.json and set "
                "'properties.allow_compat_mode': true when CPU/POSIX fallback is desired."
            ),
        )

    if "_parse_error" in config:
        return CheckResult(
            check="Compat mode availability", mode=GDSMode.COMPAT, status=Status.WARN,
            why="cufile.json has a parse error — cannot verify allow_compat_mode setting.",
            mitigation="Fix the JSON parse error in /etc/cufile.json.",
        )

    props = config.get("properties", {})
    allowed = props.get("allow_compat_mode", props.get("use_compat_mode", False))

    if allowed is False:
        return CheckResult(
            check="cufile.json compat mode", mode=GDSMode.COMPAT, status=Status.WARN,
            why=(
                "'properties.allow_compat_mode' / 'properties.use_compat_mode' is false in cufile.json, "
                "so cuFile will not fall back to the CPU bounce buffer. This is often set intentionally "
                "so a missing GDS/RDMA path fails explicitly instead of silently falling back. "
                "If GDS modes are unavailable for this filesystem (e.g. tmpfs, overlayfs), "
                "cuFile I/O will return an error instead of falling back."
            ),
            mitigation=(
                "Set 'allow_compat_mode' or 'use_compat_mode': true in /etc/cufile.json:\n"
                '  "properties": { "allow_compat_mode": true }\n\n'
                "If compat mode must stay disabled, you cannot use cuFile on filesystems "
                "that lack native GDS or RDMA support (tmpfs, NFS without RDMA, overlay, etc.)."
            ),
            evidence="cufile.json compat mode setting = false",
        )

    return CheckResult(
        check="Compat mode availability", mode=GDSMode.COMPAT, status=Status.PASS,
        why=f"compat mode is {'true' if allowed is True else 'enabled'} — "
            "CPU bounce buffer fallback is available.",
    )


def check_weka_write_support(config: Optional[dict] = None) -> CheckResult:
    """
    WekaFS GDS write support is gated by 'fs.weka.rdma_write_support' in
    cufile.json, which defaults to false. It is a config-gated capability,
    not an unconditional architectural limitation of WekaFS — leave it
    disabled and writes use the POSIX path; enable it (with a WekaFS
    deployment/client that supports RDMA writes) and writes can use GDS.
    """
    if config is None:
        config = _load_cufile_json()

    found, raw_value = _get_dotted(config, "fs.weka.rdma_write_support")
    value = raw_value if found else False

    if value is True:
        return CheckResult(
            check="WekaFS write path", mode=GDSMode.NATIVE, status=Status.INFO,
            why=(
                "'fs.weka.rdma_write_support' is true in cufile.json, so WekaFS writes "
                "may use GDS/RDMA instead of the POSIX fallback. Confirm your WekaFS "
                "deployment and client version actually support RDMA writes — "
                "verify with gdscheck or nvidia-fs write stats rather than assuming."
            ),
            evidence="fs.weka.rdma_write_support = true",
        )

    return CheckResult(
        check="WekaFS write path", mode=GDSMode.NATIVE, status=Status.WARN,
        why=(
            "'fs.weka.rdma_write_support' is false (the cufile.json default) for WekaFS, "
            "so writes use the POSIX fallback path rather than GDS/RDMA. This is the "
            "current default behavior, not a fixed WekaFS architectural limit — reads are "
            "unaffected and continue to use GDS/RDMA."
        ),
        mitigation=(
            "If your WekaFS deployment and client version support RDMA writes, set "
            "'fs.weka.rdma_write_support': true under the \"fs\": { \"weka\": { ... } } "
            "block in /etc/cufile.json, then verify with gdscheck/nvidia-fs write stats."
        ),
        evidence="fs.weka.rdma_write_support = false (default)",
    )


def check_p2pdma_config_for_fs(fs_type: str, config: dict) -> CheckResult:
    """
    For filesystems where P2PDMA/C2C requires an explicit filesystem-specific
    config key (currently NFS and virtiofs), check whether it is enabled.
    """
    from .fs_matrix import P2PDMA_CONFIG_KEY

    config_key = P2PDMA_CONFIG_KEY.get(fs_type)
    if config_key is None:
        return CheckResult(
            check="P2PDMA config key", mode=GDSMode.P2PDMA, status=Status.PASS,
            why=f"P2PDMA/C2C for {fs_type} does not require a filesystem-specific cufile.json key.",
        )

    # Navigate dotted key like "fs.lustre.use_pci_p2pdma"
    parts = config_key.split(".")
    node: Any = config
    for part in parts:
        if not isinstance(node, dict):
            node = None
            break
        node = node.get(part)

    if node is True:
        return CheckResult(
            check="P2PDMA config key", mode=GDSMode.P2PDMA, status=Status.PASS,
            why=f"'{config_key}': true found in cufile.json — P2PDMA/C2C enabled for {fs_type}.",
            evidence=f"{config_key} = {node}",
        )

    return CheckResult(
        check="P2PDMA config key", mode=GDSMode.P2PDMA, status=Status.FAIL,
        why=(
            f"P2PDMA/C2C for {fs_type} requires '{config_key}': true in /etc/cufile.json, "
            f"but it is currently {'missing' if node is None else repr(node)}. "
            "Without this, the direct P2P path is not attempted for this filesystem."
        ),
        mitigation=(
            f"Add or update /etc/cufile.json:\n"
            f'{{\n'
            f'  "fs": {{\n'
            f'    "{fs_type}": {{\n'
            f'      "use_pci_p2pdma": true\n'
            f'    }}\n'
            f'  }}\n'
            f'}}\n\n'
            f"Also ensure the hard direct-P2P requirements are met (IOMMU passthrough/off "
            f"where required for x86 PCIe P2PDMA, ACS redirect disabled for PCIe paths, "
            f"and close topology for performance). {P2P_C2C_DIRECT_PATH_NOTE}"
        ),
        evidence=f"{config_key} = {node!r}",
    )


def check_logging_level(config: dict) -> Optional[CheckResult]:
    """Suggest enabling DEBUG/TRACE logging if there are failures."""
    level = config.get("logging", {}).get("level", "ERROR")
    if level in ("ERROR", "WARN"):
        return CheckResult(
            check="cufile.json logging level", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                f"GDS logging level is '{level}'. If GDS issues are suspected, "
                "detailed logs won't be captured at this level."
            ),
            mitigation=(
                'Set logging level in /etc/cufile.json for diagnosis:\n'
                '  "logging": { "level": "DEBUG" }\n'
                "For maximum detail: \"TRACE\"\n"
                "Log file location: ./cufile.log (current dir by default) or set CUFILE_LOGFILE_PATH env var."
            ),
        )
    return None


def _check_nvme_p2pdma_settings(config: dict, block_key: str = "nvme") -> list[CheckResult]:
    """
    Check the two cufile.json settings required for P2PDMA/C2C on NVMe-backed filesystems:
      - properties.use_pci_p2pdma   (global enable)
      - block.<block_key>.use_pci_p2pdma  (per-transport enable; nvme or nvmeof)
    Both must be true. If either is absent or false, the direct P2P path will not activate.
    """
    props = config.get("properties", {})
    block_cfg = config.get("block", {}).get(block_key, {})

    prop_val  = props.get("use_pci_p2pdma", False)
    block_val = block_cfg.get("use_pci_p2pdma", False)

    issues = []
    if prop_val is not True:
        issues.append(f"properties.use_pci_p2pdma = {prop_val!r}")
    if block_val is not True:
        issues.append(f"block.{block_key}.use_pci_p2pdma = {block_val!r}")

    if issues:
        return [CheckResult(
            check="cufile.json P2PDMA settings",
            mode=GDSMode.P2PDMA,
            status=Status.FAIL,
            why=(
                f"P2PDMA/C2C is disabled in /etc/cufile.json: {', '.join(issues)}. "
                "Both settings must be true for the direct P2P path to activate."
            ),
            mitigation=(
                "Set both to true in /etc/cufile.json:\n"
                '  "properties": { "use_pci_p2pdma": true }\n'
                f'  "block": {{ "{block_key}": {{ "use_pci_p2pdma": true }} }}\n'
                f"{P2P_C2C_DIRECT_PATH_NOTE}"
            ),
            evidence="; ".join(issues),
        )]

    return [CheckResult(
        check="cufile.json P2PDMA settings",
        mode=GDSMode.P2PDMA,
        status=Status.PASS,
        why=(
            f"P2PDMA/C2C enabled in cufile.json: "
            f"properties.use_pci_p2pdma=true, block.{block_key}.use_pci_p2pdma=true. "
            f"{P2P_C2C_DIRECT_PATH_NOTE}"
        ),
    )]


def run_all(fs_type: str, p2pdma_block_key: Optional[str] = None) -> list[CheckResult]:
    """
    Checks for Native GDS / P2PDMA/C2C modes — cufile.json settings that affect acceleration.
    Does NOT include compat mode checks; use check_allow_compat_mode() for that.
    """
    config = _load_cufile_json()
    results: list[CheckResult] = []

    if config is None:
        results.append(CheckResult(
            check="cufile.json", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                f"cufile.json not found at {CUFILE_JSON_PATH}. "
                "GDS will use compiled-in defaults. "
                "For filesystems requiring a P2PDMA/C2C config key, direct P2P will be disabled."
            ),
            mitigation=(
                "A cufile.json template is installed with the GDS tools/libcufile packages:\n"
                "  cp /usr/local/cuda/gds/cufile.json /etc/cufile.json\n"
                "Then edit to enable P2PDMA/C2C for your filesystem if applicable."
            ),
        ))
        return results

    if "_parse_error" in config:
        results.append(CheckResult(
            check="cufile.json", mode=GDSMode.NATIVE, status=Status.FAIL,
            why=f"cufile.json at {config.get('_path')} has a JSON parse error: {config['_parse_error']}",
            mitigation="Validate the file: python3 -m json.tool /etc/cufile.json",
        ))
        return results

    # force_compat_mode bypasses GDS acceleration (WARN — compat IS active, acceleration is not)
    fc = check_force_compat_mode(config)
    if fc:
        results.append(fc)

    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES
    normalized_fs = FS_ALIASES.get(fs_type, fs_type)
    caps = FS_CAPABILITIES.get(normalized_fs, {})

    if p2pdma_block_key:
        results.extend(_check_nvme_p2pdma_settings(config, block_key=p2pdma_block_key))

    elif normalized_fs == "nvme-of":
        # NVMe-oF uses a separate block.nvmeof key from local NVMe.
        results.extend(_check_nvme_p2pdma_settings(config, block_key="nvmeof"))

    elif normalized_fs == "raid0":
        # RAID0 uses a separate block.raid key.
        results.extend(_check_nvme_p2pdma_settings(config, block_key="raid"))

    elif caps.get("p2pdma") is True:
        # Local NVMe-backed FSes (ext4, xfs): need both global and per-transport settings.
        results.extend(_check_nvme_p2pdma_settings(config, block_key="nvme"))

    elif caps.get("p2pdma") == "config":
        # FSes with a per-FS config key in cufile.json (virtiofs, etc.)
        # The key is looked up from P2PDMA_CONFIG_KEY in fs_matrix.py.
        results.append(check_p2pdma_config_for_fs(normalized_fs, config))

    elif caps.get("p2pdma") is False:
        # P2PDMA/C2C is not a GDS library-supported route for this filesystem.
        # If the global use_pci_p2pdma=true is set, gdscheck may show a p2pdma/c2c token in
        # DRIVER CONFIGURATION — but this reflects the global cufile.json setting, NOT an active
        # capability. Warn so the user is not misled by gdscheck output.
        prop_val = config.get("properties", {}).get("use_pci_p2pdma", False)
        if prop_val is True:
            results.append(CheckResult(
                check="cufile.json P2PDMA settings",
                mode=GDSMode.P2PDMA,
                status=Status.WARN,
                why=(
                    f"properties.use_pci_p2pdma=true is set globally in cufile.json, but "
                    f"{normalized_fs} is not a GDS library-supported direct P2P route. "
                    f"gdscheck may show 'p2pdma' or 'c2c' in the DRIVER CONFIGURATION line for this "
                    f"filesystem, but the JSON setting does not create library support. "
                    f"Supported direct P2P routes are NVMe, NVMe-oF, virtiofs, and RAID0."
                ),
                mitigation=None,
            ))

    return results


def run_compat_checks() -> list[CheckResult]:
    """
    Checks specifically for compat (CPU bounce buffer) mode availability.
    Call this when building the compat ModeReport.
    """
    config = _load_cufile_json()
    return [check_allow_compat_mode(config)]
