# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RDMA stack checks for GDS network-storage modes.

Required for: Lustre RDMA, NFSoRDMA, WekaFS, GPFS/SpectrumScale.

Stack requirements:
  1. MLNX_OFED or DOCA installed (provides Mellanox IB drivers)
  2. nvidia_peermem module loaded (enables RDMA↔GPU direct transfers)
     - CUDA 11.5.1+: modprobe nvidia_peermem
     - Persistence: /etc/modules-load.d/nvidia-peermem.conf
  3. IB/RoCE devices up and connected
  4. Peer distance between GPU and NIC should be low (same PCIe switch preferred)

Source: https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
"""
from __future__ import annotations

import os
import ipaddress
import re
import socket
import subprocess
from collections import Counter
from typing import Any, Optional

from .result import CheckResult, GDSMode, Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lsmod_has(module: str) -> bool:
    try:
        result = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        return any(
            re.match(rf"^{re.escape(module)}\b", line)
            for line in result.stdout.splitlines()
        )
    except Exception:
        return False


def _run(*cmd, timeout=10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return -1, "", str(exc)


def _usable_ipv4(value: str) -> Optional[str]:
    try:
        ip = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None
    if ip.version != 4:
        return None
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return None
    return str(ip)


def _local_ipv4_map() -> dict[str, Optional[str]]:
    """Return local non-loopback IPv4 addresses mapped to interface names when known."""
    addresses: dict[str, Optional[str]] = {}

    rc, out, _ = _run("ip", "-o", "-4", "addr", "show", timeout=5)
    if rc == 0:
        for line in out.splitlines():
            m = re.match(r"\d+:\s+(\S+)\s+inet\s+([0-9.]+)/\d+", line)
            if not m:
                continue
            ip = _usable_ipv4(m.group(2))
            if ip:
                addresses[ip] = m.group(1).split("@", 1)[0]

    if not addresses:
        try:
            current: Optional[str] = None
            with open("/proc/net/fib_trie") as fh:
                for line in fh:
                    m = re.search(r"\|\--\s+([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", line)
                    if m:
                        current = m.group(1)
                        continue
                    if "host LOCAL" in line and current:
                        ip = _usable_ipv4(current)
                        if ip:
                            addresses[ip] = None
        except OSError:
            pass

    if not addresses:
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
                ip = _usable_ipv4(item[4][0])
                if ip:
                    addresses[ip] = None
        except OSError:
            pass

    return addresses


def _rdma_netdevs() -> set[str]:
    netdevs: set[str] = set()
    root = "/sys/class/infiniband"
    try:
        for hca in os.listdir(root):
            net_root = os.path.join(root, hca, "device", "net")
            try:
                netdevs.update(os.listdir(net_root))
            except OSError:
                continue
    except OSError:
        pass
    return netdevs


def _flatten_address_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_flatten_address_values(nested))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for nested in value:
            values.extend(_flatten_address_values(nested))
        return values
    return [str(value)]


def _rdma_addresses_for_fs(config: dict[str, Any], fs_type: str = "") -> tuple[list[str], list[str]]:
    props = config.get("properties", {}) if isinstance(config, dict) else {}
    addresses: list[str] = []
    sources: list[str] = []

    global_addrs = _flatten_address_values(props.get("rdma_dev_addr_list", []))
    if global_addrs:
        addresses.extend(global_addrs)
        sources.append("properties.rdma_dev_addr_list")

    fs_key = {
        "lustre": "lustre",
        "nfs": "nfs",
        "nfs-rdma": "nfs",
        "gpfs": "gpfs",
        "mmfs": "gpfs",
        "wekafs": "weka",
        "weka": "weka",
    }.get((fs_type or "").lower())

    fs_cfg = config.get("fs", {}).get(fs_key, {}) if fs_key else {}
    if isinstance(fs_cfg, dict):
        fs_addrs = _flatten_address_values(fs_cfg.get("rdma_dev_addr_list", []))
        if fs_addrs:
            addresses.extend(fs_addrs)
            sources.append(f"fs.{fs_key}.rdma_dev_addr_list")

        mount_addrs = _flatten_address_values(fs_cfg.get("mount_table", {}))
        if mount_addrs:
            addresses.extend(mount_addrs)
            sources.append(f"fs.{fs_key}.mount_table")

    deduped = list(dict.fromkeys(str(addr).strip() for addr in addresses if str(addr).strip()))
    return deduped, sources


def validate_rdma_client_addresses(
    addresses: Any,
    local_ip_map: Optional[dict[str, Optional[str]]] = None,
    rdma_netdevs: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Validate cufile.json RDMA addresses against local client IPv4 addresses.

    `rdma_dev_addr_list` is expected to contain client-side RDMA NIC IPs, not
    storage server addresses. When interface mapping is available, flag local
    IPs that are not on a netdev associated with an RDMA HCA.
    """
    configured = list(dict.fromkeys(
        str(addr).strip() for addr in _flatten_address_values(addresses)
        if str(addr).strip()
    ))
    local_ip_map = _local_ipv4_map() if local_ip_map is None else local_ip_map
    rdma_netdevs = _rdma_netdevs() if rdma_netdevs is None else rdma_netdevs

    invalid: list[str] = []
    nonlocal_ips: list[str] = []
    matched: list[str] = []
    non_rdma_iface: list[str] = []

    for raw in configured:
        try:
            parsed = ipaddress.ip_address(raw)
        except ValueError:
            invalid.append(raw)
            continue
        if parsed.version != 4:
            invalid.append(raw)
            continue
        ip = str(parsed)
        if not local_ip_map:
            continue
        if ip not in local_ip_map:
            nonlocal_ips.append(ip)
            continue
        matched.append(ip)
        iface = local_ip_map.get(ip)
        if iface and rdma_netdevs and iface not in rdma_netdevs:
            non_rdma_iface.append(f"{ip} ({iface})")

    return {
        "configured": configured,
        "local_ips": sorted(local_ip_map),
        "rdma_netdevs": sorted(rdma_netdevs),
        "invalid": invalid,
        "nonlocal": nonlocal_ips,
        "matched": matched,
        "non_rdma_iface": non_rdma_iface,
        "verification_available": bool(local_ip_map),
    }


RDMA_POLICY_FS_TYPES = {"gpfs", "mmfs", "wekafs"}
RDMA_POLICY_DEFAULT = "RoundRobin"
RDMA_TOPN_DEFAULT = 1
RDMA_DYNAMIC_ROUTING_ORDER_DEFAULT = ["GPU_MEM_NVLINKS", "GPU_MEM", "SYS_MEM", "P2P"]
RDMA_LOAD_BALANCING_POLICIES = {
    "FirstFit",
    "MaxMinFit",
    "RoundRobin",
    "RoundRobinMaxMin",
    "Randomized",
}


def _run_nvidia_smi_topo() -> tuple[str, Optional[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return "", msg or f"nvidia-smi topo exited with code {result.returncode}"
        return result.stdout, None
    except FileNotFoundError:
        return "", "nvidia-smi not found in PATH"
    except Exception as exc:
        return "", f"nvidia-smi topo -m failed: {exc}"


def _device_sort_key(label: str) -> tuple[str, int, str]:
    m = re.match(r"^([A-Za-z]+)(\d+)$", label)
    if m:
        return (m.group(1).upper(), int(m.group(2)), label)
    return (label.upper(), -1, label)


def _topo_output_preview(output: str, max_lines: int = 12) -> str:
    from . import pcie

    lines = [line.rstrip() for line in pcie._strip_ansi(output).splitlines() if line.strip()]
    if not lines:
        return "<empty output>"

    for idx, line in enumerate(lines):
        tokens = [tok.rstrip(":=") for tok in line.split()]
        if any(pcie._is_nic_label(tok) for tok in tokens):
            start = max(0, idx - 2)
            end = min(len(lines), idx + max_lines)
            return "\n".join(lines[start:end])

    return "\n".join(lines[:max_lines])


def parse_gpu_nic_topology(output: str) -> dict[str, Any]:
    """
    Parse `nvidia-smi topo -m` GPU/NIC relationships.

    The cuFile RDMA load balancer builds its rank table from platform topology,
    then applies the chosen K-nearest policy. This parser keeps the same user
    visible labels (`GPU0`, `NIC0`) so recommendations can map back to the
    nvidia-smi table and NIC legend.
    """
    from . import pcie

    columns: list[str] = []
    gpu_to_nic: dict[str, dict[str, str]] = {}
    nic_details: dict[str, str] = {}

    for raw in output.splitlines():
        line = pcie._strip_ansi(raw).rstrip()
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens:
            continue

        legend_match = re.match(r"^\s*(NIC\d+)\s*:\s*(.+?)\s*$", line, re.I)
        if legend_match:
            nic_details[legend_match.group(1)] = legend_match.group(2)
            continue

        header_columns = pcie._topo_header_columns(line, tokens)
        if header_columns and any(pcie._is_nic_label(col) for col in header_columns):
            columns = header_columns
            continue

        if not columns:
            continue

        row_label = tokens[0].rstrip(":=")
        values = tokens[1:1 + len(columns)]
        if len(values) < len(columns):
            continue

        if pcie._is_gpu_label(row_label):
            for col, val in zip(columns, values):
                if pcie._is_nic_label(col) and pcie._is_topo_relation(val):
                    gpu_to_nic.setdefault(row_label, {})[col] = val
        elif pcie._is_nic_label(row_label):
            for col, val in zip(columns, values):
                if pcie._is_gpu_label(col) and pcie._is_topo_relation(val):
                    gpu_to_nic.setdefault(col, {})[row_label] = val

    gpus = sorted(gpu_to_nic, key=_device_sort_key)
    nics = sorted({nic for relations in gpu_to_nic.values() for nic in relations}, key=_device_sort_key)
    return {
        "gpus": gpus,
        "nics": nics,
        "gpu_to_nic": gpu_to_nic,
        "nic_details": nic_details,
    }


def _normalize_fs_type(fs_type: str) -> str:
    try:
        from .fs_matrix import FS_ALIASES
        return FS_ALIASES.get(fs_type, fs_type).lower()
    except Exception:
        return (fs_type or "").lower()


def _rdma_config_snapshot(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if config is None:
        try:
            from .cufile_config import _load_cufile_json
            config = _load_cufile_json()
        except Exception:
            config = None

    props = config.get("properties", {}) if isinstance(config, dict) and "_parse_error" not in config else {}
    return {
        "policy": props.get(
            "rdma_load_balancing_policy",
            props.get("rdma_peer_affinity_policy", RDMA_POLICY_DEFAULT),
        ),
        "topn": props.get("rdma_topN_ranks", RDMA_TOPN_DEFAULT),
        "dynamic_routing": props.get("rdma_dynamic_routing", False),
        "dynamic_routing_order": props.get(
            "rdma_dynamic_routing_order",
            RDMA_DYNAMIC_ROUTING_ORDER_DEFAULT,
        ),
        "rdma_dev_addr_list": props.get("rdma_dev_addr_list", []),
    }


def _best_nics_for_gpu(relations: dict[str, str]) -> tuple[list[str], str, int]:
    from . import pcie

    ranked = sorted(relations.items(), key=lambda item: (pcie._topo_relation_rank(item[1]), _device_sort_key(item[0])))
    if not ranked:
        return [], "", 99
    best_rank = pcie._topo_relation_rank(ranked[0][1])
    best_relation = ranked[0][1]
    return [nic for nic, rel in ranked if pcie._topo_relation_rank(rel) == best_rank], best_relation, best_rank


def _best_gpus_for_nic(parsed: dict[str, Any], nic: str) -> list[str]:
    from . import pcie

    choices: list[tuple[str, str]] = []
    for gpu, relations in parsed.get("gpu_to_nic", {}).items():
        if isinstance(relations, dict) and nic in relations:
            choices.append((gpu, relations[nic]))
    if not choices:
        return []
    best_rank = min(pcie._topo_relation_rank(rel) for _, rel in choices)
    return sorted(
        [gpu for gpu, rel in choices if pcie._topo_relation_rank(rel) == best_rank],
        key=_device_sort_key,
    )


def _format_nic(nic: str, details: dict[str, str]) -> str:
    detail = details.get(nic)
    return f"{nic}/{detail}" if detail else nic


def _format_best_summary(parsed: dict[str, Any], best_by_gpu: dict[str, tuple[list[str], str, int]]) -> str:
    details = parsed.get("nic_details", {})
    parts = []
    for gpu in parsed.get("gpus", []):
        best_nics, relation, _ = best_by_gpu.get(gpu, ([], "", 99))
        if not best_nics:
            continue
        rendered = ", ".join(_format_nic(nic, details) for nic in best_nics)
        parts.append(f"{gpu} -> {rendered} ({relation})")
    return "; ".join(parts)


def _recommend_policy(
    gpu_count: int,
    nic_count: int,
    best_by_gpu: dict[str, tuple[list[str], str, int]],
) -> tuple[str, str]:
    if gpu_count <= 1 and nic_count <= 1:
        return (
            "RoundRobin",
            "single GPU or single NIC topology; every policy resolves to the same nearest NIC, so keep the cuFile default",
        )
    if gpu_count <= 1:
        return (
            "RoundRobin",
            "single GPU with multiple visible NICs; RoundRobin rotates across equivalent K-nearest NICs while keeping the cuFile default",
        )
    if nic_count <= 1:
        return (
            "RoundRobin",
            "multiple GPUs share one visible NIC; the load-balancing policy cannot choose another NIC, so keep the cuFile default",
        )

    shared_best = Counter(
        nic
        for best_nics, _, _ in best_by_gpu.values()
        for nic in best_nics[:1]
    )
    max_sharing = max(shared_best.values()) if shared_best else 0
    tied_gpu_count = sum(1 for best_nics, _, _ in best_by_gpu.values() if len(best_nics) > 1)

    if max_sharing > 1 or gpu_count != nic_count:
        return (
            "RoundRobinMaxMin",
            "multiple GPUs/NICs with shared nearest peers or an uneven GPU:NIC count; cuFile first minimizes NIC sharing within K-nearest ranks, then round-robins",
        )
    if tied_gpu_count:
        return (
            "RoundRobinMaxMin",
            "multiple GPUs/NICs with tied nearest NICs; the MaxMin table avoids accidental hot spots while round-robin still uses equivalent paths",
        )
    return (
        "RoundRobinMaxMin",
        "multiple GPUs and multiple NICs; prefer the least-shared K-nearest table for WekaFS/GPFS instead of first-fit selection",
    )


def _dynamic_routing_recommendation(
    parsed: dict[str, Any],
    config_snapshot: dict[str, Any],
) -> str:
    gpus = parsed.get("gpus", [])
    nics = parsed.get("nics", [])
    current = bool(config_snapshot.get("dynamic_routing"))
    order = config_snapshot.get("dynamic_routing_order") or RDMA_DYNAMIC_ROUTING_ORDER_DEFAULT
    addr_list = config_snapshot.get("rdma_dev_addr_list") or []

    if len(gpus) <= 1:
        state = "enabled" if current else "disabled"
        return f"Dynamic routing: not needed for a single visible GPU; current properties.rdma_dynamic_routing is {state}."

    route_candidates = {nic: _best_gpus_for_nic(parsed, nic) for nic in nics}
    useful = any(best and len(best) < len(gpus) for best in route_candidates.values())
    if useful:
        nic_parts = []
        details = parsed.get("nic_details", {})
        for nic, best_gpus in route_candidates.items():
            if best_gpus and len(best_gpus) < len(gpus):
                nic_parts.append(f"{_format_nic(nic, details)} best via {', '.join(best_gpus)}")
        action = "keep enabled" if current else "enable"
        rdma_addr_note = ""
        if not addr_list:
            rdma_addr_note = " Also set properties.rdma_dev_addr_list or the Weka/GPFS mount_table so cuFile can map RDMA IPs to NICs."
        return (
            f"Dynamic routing: {action} properties.rdma_dynamic_routing=true with "
            f"rdma_dynamic_routing_order={order!r}; {'; '.join(nic_parts)}."
            f"{rdma_addr_note}"
        )

    state = "enabled" if current else "disabled"
    return (
        f"Dynamic routing: optional for this visible topology because each NIC has all GPUs at the same best distance; "
        f"current properties.rdma_dynamic_routing is {state}."
    )


def rdma_policy_recommendations_for_topology(
    fs_type: str,
    topo_output: str,
    config: Optional[dict[str, Any]] = None,
) -> list[str]:
    """
    Return WekaFS/GPFS RDMA load-balancing and dynamic-routing recommendations
    from `nvidia-smi topo -m` output.
    """
    normalized = _normalize_fs_type(fs_type)
    if normalized not in RDMA_POLICY_FS_TYPES:
        return []

    parsed = parse_gpu_nic_topology(topo_output)
    gpu_to_nic = parsed.get("gpu_to_nic", {})
    if not isinstance(gpu_to_nic, dict) or not gpu_to_nic:
        return [
            "GPU/NIC topology unavailable from nvidia-smi topo -m: "
            "no parsed GPU/NIC relationships. Output preview:\n" + _topo_output_preview(topo_output)
        ]

    gpus = parsed.get("gpus", [])
    nics = parsed.get("nics", [])
    best_by_gpu = {
        gpu: _best_nics_for_gpu(gpu_to_nic.get(gpu, {}))
        for gpu in gpus
    }
    config_snapshot = _rdma_config_snapshot(config)
    recommended_policy, reason = _recommend_policy(len(gpus), len(nics), best_by_gpu)
    current_policy = config_snapshot["policy"]
    current_topn = config_snapshot["topn"]

    recs = [
        (
            f"WekaFS/GPFS RDMA topology: {len(gpus)} GPU(s), {len(nics)} NIC(s); "
            f"closest NICs by GPU: {_format_best_summary(parsed, best_by_gpu)}"
        )
    ]

    if current_policy not in RDMA_LOAD_BALANCING_POLICIES:
        recs.append(
            f"Recommended RDMA policy: properties.rdma_load_balancing_policy='{recommended_policy}', "
            f"properties.rdma_topN_ranks={RDMA_TOPN_DEFAULT}. Current policy {current_policy!r} is not a known cuFile policy; {reason}."
        )
    elif current_policy != recommended_policy:
        recs.append(
            f"Recommended RDMA policy: set properties.rdma_load_balancing_policy='{recommended_policy}' "
            f"(current: {current_policy!r}) and keep properties.rdma_topN_ranks={RDMA_TOPN_DEFAULT} unless benchmarking justifies adding farther NIC ranks; {reason}."
        )
    else:
        recs.append(
            f"RDMA policy: current properties.rdma_load_balancing_policy='{current_policy}' matches the recommendation; "
            f"properties.rdma_topN_ranks={current_topn}. Reason: {reason}."
        )

    try:
        if int(current_topn) < 1:
            recs.append("properties.rdma_topN_ranks must be >= 1; cuFile's default is 1.")
    except (TypeError, ValueError):
        recs.append(f"properties.rdma_topN_ranks should be an integer >= 1; current value is {current_topn!r}.")

    best_ranks = [rank for _, _, rank in best_by_gpu.values() if rank != 99]
    if best_ranks:
        from . import pcie
        worst_best_rank = max(best_ranks)
        if worst_best_rank >= pcie._topo_relation_rank("SYS"):
            recs.append(
                "Topology caution: at least one closest GPU/NIC path is SYS; policy can balance choices but cannot remove cross-socket traffic."
            )
        elif worst_best_rank >= pcie._topo_relation_rank("NODE"):
            recs.append(
                "Topology note: the closest GPU/NIC path is NODE for at least one GPU; same-switch PIX/PXB placement would usually perform better."
            )

    recs.append(_dynamic_routing_recommendation(parsed, config_snapshot))
    return recs


def rdma_policy_recommendations(fs_type: str) -> list[str]:
    normalized = _normalize_fs_type(fs_type)
    if normalized not in RDMA_POLICY_FS_TYPES:
        return []

    output, err = _run_nvidia_smi_topo()
    if err:
        return [f"GPU/NIC topology unavailable from nvidia-smi topo -m: {err}"]
    return rdma_policy_recommendations_for_topology(normalized, output)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_nvidia_peermem() -> CheckResult:
    """nvidia_peermem enables RDMA devices to DMA directly to/from GPU memory."""
    loaded = _lsmod_has("nvidia_peermem")
    if loaded:
        return CheckResult(
            check="nvidia_peermem module", mode=GDSMode.RDMA, status=Status.PASS,
            why="nvidia_peermem is loaded — RDMA↔GPU memory access enabled.",
        )

    # Check if it's available but not loaded
    rc, out, _ = _run("modinfo", "nvidia_peermem")
    if rc == 0:
        return CheckResult(
            check="nvidia_peermem module", mode=GDSMode.RDMA, status=Status.FAIL,
            why=(
                "nvidia_peermem is installed but NOT loaded. "
                "Without it, RDMA devices cannot directly access GPU memory — "
                "RDMA GDS mode will fail."
            ),
            mitigation=(
                "Load immediately:\n"
                "  sudo modprobe nvidia_peermem\n\n"
                "Persist across reboots:\n"
                "  echo 'nvidia_peermem' | sudo tee /etc/modules-load.d/nvidia-peermem.conf\n\n"
                "Requires CUDA 11.5.1+. Verify with:\n"
                "  lsmod | grep nvidia_peermem"
            ),
        )

    return CheckResult(
        check="nvidia_peermem module", mode=GDSMode.RDMA, status=Status.FAIL,
        why=(
            "nvidia_peermem module not found. "
            "This module is required for all RDMA-based GDS paths (Lustre, NFS, WekaFS, GPFS)."
        ),
        mitigation=(
            "Install CUDA 11.5.1+ and the matching NVIDIA driver/GDS RDMA stack.\n"
            "On packaged CUDA systems, verify GDS packages with:\n"
            "  rpm -qa | grep gds\n"
            "  dpkg -l | grep gds\n\n"
            "Alternatively, install MLNX_OFED/DOCA where required by your filesystem stack.\n"
            "After install:\n"
            "  sudo modprobe nvidia_peermem\n"
            "  echo 'nvidia_peermem' | sudo tee /etc/modules-load.d/nvidia-peermem.conf"
        ),
    )


def check_ofed() -> CheckResult:
    """Check MLNX_OFED or DOCA installation."""
    # Try ofed_info first (MLNX_OFED)
    rc, out, _ = _run("ofed_info", "-s")
    if rc == 0 and out.strip():
        version = out.strip().splitlines()[0]
        return CheckResult(
            check="MLNX_OFED / DOCA", mode=GDSMode.RDMA, status=Status.PASS,
            why=f"MLNX_OFED detected: {version}",
            evidence=version,
        )

    # Try DOCA
    rc2, out2, _ = _run("doca_version")
    if rc2 == 0 and out2.strip():
        return CheckResult(
            check="MLNX_OFED / DOCA", mode=GDSMode.RDMA, status=Status.PASS,
            why=f"NVIDIA DOCA detected: {out2.strip()}",
            evidence=out2.strip(),
        )

    # Check if inbox RDMA drivers are present (may work for some setups)
    inbox_modules = ["mlx5_ib", "mlx4_ib", "rdma_ucm"]
    loaded_inbox = [m for m in inbox_modules if _lsmod_has(m)]
    if loaded_inbox:
        return CheckResult(
            check="MLNX_OFED / DOCA", mode=GDSMode.RDMA, status=Status.WARN,
            why=(
                f"Inbox RDMA drivers found ({', '.join(loaded_inbox)}) but MLNX_OFED/DOCA not detected. "
                "Inbox drivers may work for basic RDMA but NVIDIA recommends MLNX_OFED for GDS."
            ),
            mitigation=(
                "Install MLNX_OFED from https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/\n"
                "Or install DOCA from https://developer.nvidia.com/doca\n"
                "MLNX_OFED is required for optimal GDS RDMA performance and full feature support."
            ),
            evidence=f"Inbox modules loaded: {', '.join(loaded_inbox)}",
        )

    return CheckResult(
        check="MLNX_OFED / DOCA", mode=GDSMode.RDMA, status=Status.FAIL,
        why=(
            "Neither MLNX_OFED nor DOCA found, and no inbox RDMA drivers are loaded. "
            "RDMA GDS path requires Mellanox/NVIDIA ConnectX InfiniBand or RoCE hardware + drivers."
        ),
        mitigation=(
            "1. Verify you have a Mellanox/NVIDIA ConnectX NIC: lspci | grep -i mellanox\n"
            "2. Install MLNX_OFED:\n"
            "   Download from https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/\n"
            "   Run: sudo ./mlnxofedinstall --with-nvmf --with-nfsrdma\n"
            "3. Or install DOCA for newer BlueField/ConnectX-7 hardware."
        ),
    )


def check_ofed_preinstall() -> CheckResult:
    """
    Pre-install advisory for MLNX_OFED / DOCA.

    Missing MLNX_OFED/DOCA is not a universal GDS blocker: a host may still use
    supported upstream PCI P2PDMA or compat paths. It is, however, important
    enough to warn before installation because GPFS/WekaFS userspace RDMA,
    Lustre/NFS RDMA, and NVMe/NVMe-oF nvfs paths that need GDS storage-stack
    patches depend on MLNX_OFED or DOCA/DOCA-OFED in many deployments.
    """
    result = check_ofed()
    if result.status == Status.PASS:
        return result

    mitigation = (
        "Install MLNX_OFED or DOCA/DOCA-OFED if this host will use GPFS/WekaFS "
        "userspace RDMA, Lustre/NFS-RDMA, or NVMe/NVMe-oF nvidia-fs/nvfs mode "
        "with GDS storage-stack patches.\n"
        "GDS DOCA requirements:\n"
        "  https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#doca-requirements-and-installation\n"
        "DOCA storage installation:\n"
        "  https://docs.nvidia.com/doca/sdk/doca-host-installation-and-upgrade/index.html#storage-installation\n"
        "Verify after install:\n"
        "  ofed_info -s || doca_version\n"
        "If you only plan to use a confirmed upstream PCI P2PDMA route or compat mode, "
        "this warning may be informational."
    )

    return CheckResult(
        check=result.check,
        mode=result.mode,
        status=Status.WARN,
        why=(
            f"{result.why} Pre-install advisory: MLNX_OFED/DOCA is recommended "
            "for GDS RDMA paths and for NVMe/NVMe-oF nvfs deployments that rely "
            "on NVIDIA-provided storage-stack patches."
        ),
        mitigation=mitigation,
        evidence=result.evidence,
    )


def check_ib_devices() -> CheckResult:
    """Check that InfiniBand/RoCE devices are present and active."""
    rc, out, _ = _run("ibv_devinfo")
    if rc != 0 or not out.strip():
        # Try rdma link show as alternative
        rc2, out2, _ = _run("rdma", "link", "show")
        if rc2 == 0 and out2.strip():
            active = [l for l in out2.splitlines() if "ACTIVE" in l.upper()]
            if active:
                return CheckResult(
                    check="RDMA devices", mode=GDSMode.RDMA, status=Status.PASS,
                    why=f"RDMA link(s) active:\n" + "\n".join(active),
                    evidence=out2[:300],
                )

        return CheckResult(
            check="RDMA devices", mode=GDSMode.RDMA, status=Status.FAIL,
            why=(
                "ibv_devinfo returned no devices. No InfiniBand/RoCE ports visible. "
                "RDMA GDS path requires at least one active RDMA-capable port."
            ),
            mitigation=(
                "1. Check hardware: lspci | grep -i 'infiniband\\|mellanox\\|connectx'\n"
                "2. Check port state: ibstat\n"
                "3. Bring up ports: sudo /etc/init.d/openibd restart\n"
                "4. Verify subnet manager is running (for IB): systemctl status opensmd\n"
                "5. For RoCE: ensure the ethernet port is up and configured"
            ),
        )

    # Parse device count and PORT_ACTIVE status
    device_blocks = re.split(r"(?=^hca_id:)", out, flags=re.MULTILINE)
    active_ports: list[str] = []
    inactive_ports: list[str] = []

    for block in device_blocks:
        if not block.strip():
            continue
        hca_m = re.search(r"hca_id:\s*(\S+)", block)
        hca = hca_m.group(1) if hca_m else "unknown"
        port_states = re.findall(r"(?:port_state|state):\s*(PORT_\S+)", block)
        for state in port_states:
            if state.upper() == "PORT_ACTIVE":
                active_ports.append(f"{hca}:{state}")
            else:
                inactive_ports.append(f"{hca}:{state}")

    if active_ports:
        return CheckResult(
            check="RDMA devices", mode=GDSMode.RDMA, status=Status.PASS,
            why=f"Active RDMA port(s): {', '.join(active_ports)}",
            evidence=out[:400],
        )

    if inactive_ports:
        return CheckResult(
            check="RDMA devices", mode=GDSMode.RDMA, status=Status.WARN,
            why=(
                f"RDMA device(s) found but ports not in PORT_ACTIVE state: {', '.join(inactive_ports)}. "
                "RDMA GDS transfers will fail until ports are up."
            ),
            mitigation=(
                "Check cable connections and switch configuration.\n"
                "For InfiniBand: ensure subnet manager is running: systemctl status opensmd\n"
                "Check port state: ibstat\n"
                "Restart OpenIB stack: sudo /etc/init.d/openibd restart"
            ),
            evidence=out[:400],
        )

    return CheckResult(
        check="RDMA devices", mode=GDSMode.RDMA, status=Status.WARN,
        why="ibv_devinfo ran but could not parse device/port state.",
        evidence=out[:400],
    )


def check_peer_distance() -> CheckResult:
    """
    Check GPU↔NIC peer distance via /proc/driver/nvidia-fs/peer_distance if available.
    A last-column value of 1 indicates traffic goes via root complex (suboptimal).
    """
    peer_dist_path = "/proc/driver/nvidia-fs/peer_distance"
    try:
        with open(peer_dist_path) as fh:
            content = fh.read()
    except FileNotFoundError:
        return CheckResult(
            check="GPU↔NIC peer distance", mode=GDSMode.RDMA, status=Status.WARN,
            why=(
                f"{peer_dist_path} not found. "
                "nvidia_fs module must be loaded to expose peer distance info. "
                "Cannot verify GPU↔NIC PCIe proximity."
            ),
            mitigation="Load nvidia_fs and check: cat /proc/driver/nvidia-fs/peer_distance",
        )
    except PermissionError:
        return CheckResult(
            check="GPU↔NIC peer distance", mode=GDSMode.RDMA, status=Status.WARN,
            why=f"Cannot read {peer_dist_path} — requires elevated privileges.",
            mitigation=f"Run as root: sudo cat {peer_dist_path}",
        )

    suboptimal: list[str] = []
    for line in content.splitlines()[1:]:  # skip header
        parts = line.split()
        if parts and parts[-1] == "1":     # last column = 1 → via root complex
            suboptimal.append(line.strip())

    if suboptimal:
        return CheckResult(
            check="GPU↔NIC peer distance", mode=GDSMode.RDMA, status=Status.WARN,
            why=(
                "Some GPU↔NIC pairs route through the PCIe root complex (suboptimal). "
                "This increases latency but does not prevent GDS RDMA — it reduces throughput.\n"
                + "\n".join(f"  {l}" for l in suboptimal)
            ),
            mitigation=(
                "For best RDMA GDS performance, GPU and NIC should be on the same PCIe switch. "
                "Enable dynamic routing in /etc/cufile.json:\n"
                '  "rdma_dynamic_routing": true\n'
                '  "rdma_dynamic_routing_order": ["GPU_MEM_NVLINKS"]\n'
                "This lets GDS route via NVLink when PCIe path is suboptimal."
            ),
            evidence=content[:400],
        )

    return CheckResult(
        check="GPU↔NIC peer distance", mode=GDSMode.RDMA, status=Status.PASS,
        why="All GPU↔NIC pairs have optimal PCIe proximity (no root-complex hops).",
        evidence=content[:400],
    )


def check_rdma_peer_type_config() -> CheckResult:
    """
    Check if rdma_peer_type is set in cufile.json.
    When Mellanox PeerDirect (nvidia_peermem) is not in use, DmaBuf is the
    alternative RDMA path. It requires 'rdma_peer_type': 'dmabuf' in cufile.json
    and MLNX_OFED >= 5.6 or DOCA.

    This check surfaces the setting so operators know which path is active.
    """
    from .cufile_config import _load_cufile_json
    cfg = _load_cufile_json()

    if cfg is None or "_parse_error" in (cfg or {}):
        return CheckResult(
            check="cufile.json rdma_peer_type", mode=GDSMode.RDMA, status=Status.WARN,
            why="Cannot read /etc/cufile.json to check rdma_peer_type setting.",
        )

    peer_type = cfg.get("properties", {}).get("rdma_peer_type")

    if peer_type == "dmabuf":
        return CheckResult(
            check="cufile.json rdma_peer_type", mode=GDSMode.RDMA, status=Status.PASS,
            why="rdma_peer_type = 'dmabuf' — using DmaBuf RDMA path (no nvidia_peermem required).",
            evidence=f"properties.rdma_peer_type = {peer_type}",
        )
    if peer_type == "peer_mem":
        return CheckResult(
            check="cufile.json rdma_peer_type", mode=GDSMode.RDMA, status=Status.PASS,
            why="rdma_peer_type = 'peer_mem' — using PeerDirect path (nvidia_peermem required).",
            evidence=f"properties.rdma_peer_type = {peer_type}",
        )
    # Not set — default is PeerDirect (nvidia_peermem)
    return CheckResult(
        check="cufile.json rdma_peer_type", mode=GDSMode.RDMA, status=Status.WARN,
        why=(
            "properties.rdma_peer_type not set in cufile.json — defaulting to PeerDirect path "
            "(nvidia_peermem required). If nvidia_peermem is not loaded and DmaBuf is available, "
            "consider switching to the DmaBuf path."
        ),
        mitigation=(
            "To use DmaBuf RDMA (no nvidia_peermem needed; requires MLNX_OFED >= 5.6):\n"
            '  Add to /etc/cufile.json under "properties":\n'
            '    "rdma_peer_type": "dmabuf"\n\n'
            "To explicitly use PeerDirect:\n"
            '    "rdma_peer_type": "peer_mem"\n'
            "  And ensure nvidia_peermem is loaded: sudo modprobe nvidia_peermem"
        ),
        evidence="properties.rdma_peer_type = (not set)",
    )


def _format_rdma_addr_evidence(validation: dict[str, Any], sources: Optional[list[str]] = None) -> str:
    lines = []
    if sources:
        lines.append(f"sources: {', '.join(sources)}")
    lines.append(f"configured: {', '.join(validation['configured']) or '(none)'}")
    lines.append(f"local client IPv4s: {', '.join(validation['local_ips']) or '(none discovered)'}")
    if validation.get("rdma_netdevs"):
        lines.append(f"RDMA netdevs: {', '.join(validation['rdma_netdevs'])}")
    return "\n".join(lines)


def check_rdma_dev_addr_list(fs_type: str = "") -> CheckResult:
    """
    Check if RDMA device IP addresses are configured in cufile.json.
    Without rdma_dev_addr_list, GDS attempts auto-discovery of RDMA devices,
    which may not work in all environments.
    """
    from .cufile_config import _load_cufile_json
    cfg = _load_cufile_json()

    if cfg is None or "_parse_error" in (cfg or {}):
        return CheckResult(
            check="cufile.json rdma_dev_addr_list", mode=GDSMode.RDMA, status=Status.WARN,
            why="Cannot read /etc/cufile.json to check rdma_dev_addr_list.",
        )

    addr_list, sources = _rdma_addresses_for_fs(cfg, fs_type)

    if addr_list:
        validation = validate_rdma_client_addresses(addr_list)
        issues = []
        if validation["invalid"]:
            issues.append(f"invalid/non-IPv4 entries: {', '.join(validation['invalid'])}")
        if validation["nonlocal"]:
            issues.append(f"not local to this client: {', '.join(validation['nonlocal'])}")
        if validation["non_rdma_iface"]:
            issues.append(f"local but not on an RDMA netdev: {', '.join(validation['non_rdma_iface'])}")

        if issues:
            return CheckResult(
                check="cufile.json rdma_dev_addr_list", mode=GDSMode.RDMA, status=Status.WARN,
                why=(
                    "rdma_dev_addr_list is configured, but it should contain client-side "
                    "RDMA NIC IPv4 addresses. Found " + "; ".join(issues) + "."
                ),
                mitigation=(
                    "Update /etc/cufile.json so properties.rdma_dev_addr_list or the "
                    "per-filesystem mount_table contains IPs assigned to this host's "
                    "IB/RoCE client interfaces. Verify with:\n"
                    "  ip -o -4 addr show\n"
                    "  ls /sys/class/infiniband/*/device/net"
                ),
                evidence=_format_rdma_addr_evidence(validation, sources),
            )

        if not validation["verification_available"]:
            return CheckResult(
                check="cufile.json rdma_dev_addr_list", mode=GDSMode.RDMA, status=Status.WARN,
                why=(
                    "rdma_dev_addr_list is configured, but local client IPv4 addresses "
                    "could not be discovered, so the tool cannot verify that the entries "
                    "belong to this host."
                ),
                mitigation="Verify manually with: ip -o -4 addr show",
                evidence=_format_rdma_addr_evidence(validation, sources),
            )

        return CheckResult(
            check="cufile.json rdma_dev_addr_list", mode=GDSMode.RDMA, status=Status.PASS,
            why=(
                f"rdma_dev_addr_list configured with {len(addr_list)} client-local "
                f"address(es): {', '.join(str(a) for a in addr_list)}"
            ),
            evidence=_format_rdma_addr_evidence(validation, sources),
        )

    return CheckResult(
        check="cufile.json rdma_dev_addr_list", mode=GDSMode.RDMA, status=Status.WARN,
        why=(
            "properties.rdma_dev_addr_list is not set in cufile.json. "
            "GDS will attempt auto-discovery of RDMA devices, which may fail in "
            "multi-NIC or complex network configurations."
        ),
        mitigation=(
            "Add RDMA NIC IP addresses to /etc/cufile.json:\n"
            '  "properties": {\n'
            '    "rdma_dev_addr_list": ["<rdma-nic-ip-1>", "<rdma-nic-ip-2>"]\n'
            '  }\n\n'
            "Find your RDMA NIC IPs:\n"
            "  ip addr show   (look for IB/RoCE interface)\n"
            "  ibstat | grep -A5 'CA '   (for InfiniBand)"
        ),
    )


def check_peermem_or_dmabuf() -> CheckResult:
    """
    For GPFS/WekaFS (userspace RDMA): either nvidia_peermem must be loaded
    OR rdma_peer_type=dmabuf must be set in cufile.json.
    Both are valid paths — the operator chooses one.
    """
    from .cufile_config import _load_cufile_json
    cfg = _load_cufile_json() or {}
    peer_type = cfg.get("properties", {}).get("rdma_peer_type") if cfg else None
    using_dmabuf = peer_type == "dmabuf"

    peermem_loaded = _lsmod_has("nvidia_peermem")

    if peermem_loaded:
        return CheckResult(
            check="Userspace RDMA peer transport", mode=GDSMode.RDMA, status=Status.PASS,
            why="nvidia_peermem is loaded — PeerDirect path active for userspace RDMA.",
        )
    if using_dmabuf:
        return CheckResult(
            check="Userspace RDMA peer transport", mode=GDSMode.RDMA, status=Status.PASS,
            why="rdma_peer_type=dmabuf in cufile.json — DmaBuf path active for userspace RDMA.",
            evidence="properties.rdma_peer_type = dmabuf",
        )

    # Neither present — show both options
    rc, _, _ = _run("modinfo", "nvidia_peermem")
    peermem_installed = (rc == 0)

    return CheckResult(
        check="Userspace RDMA peer transport", mode=GDSMode.RDMA, status=Status.FAIL,
        why=(
            "Neither nvidia_peermem (PeerDirect) nor rdma_peer_type=dmabuf (DmaBuf) is active. "
            "GPFS/WekaFS require one of these for userspace RDMA GDS transfers."
        ),
        mitigation=(
            "Option 1 — Load nvidia_peermem (PeerDirect path):\n"
            + (
                "  sudo modprobe nvidia_peermem\n"
                "  echo 'nvidia_peermem' | sudo tee /etc/modules-load.d/nvidia-peermem.conf\n"
                if peermem_installed else
                "  nvidia_peermem is not installed. Install cuda-toolkit (≥ 11.5.1) or MLNX_OFED.\n"
            ) +
            "\nOption 2 — Use DmaBuf path (no nvidia_peermem needed; requires MLNX_OFED ≥ 5.6):\n"
            '  Add to /etc/cufile.json under "properties":\n'
            '    "rdma_peer_type": "dmabuf"\n'
            "  Then verify: gdscheck -p | grep -A3 'Userspace RDMA'"
        ),
    )


def run_all(fs_type: str = "") -> list[CheckResult]:
    """
    Run RDMA checks appropriate for the given filesystem type.

    GPFS/WekaFS        — userspace RDMA: need OFED + (nvidia_peermem OR dmabuf) + IB devices.
    Lustre/BeeGFS/NFS  — kernel RDMA (nvidia-fs orchestrates it): need OFED + IB devices;
                         nvidia_peermem NOT required. NFS/NFS4 additionally needs the rdma
                         mount option, checked separately via _check_nfs_rdma_mount().
    """
    from .fs_matrix import FS_CAPABILITIES, FS_ALIASES
    normalized = FS_ALIASES.get(fs_type, fs_type)
    caps = FS_CAPABILITIES.get(normalized, {})
    rdma_type = caps.get("rdma_type", "userspace")  # default: assume userspace for unknown FS

    results = [check_ofed(), check_ib_devices()]

    if rdma_type == "userspace":
        # GPFS/WekaFS: userspace RDMA — need nvidia_peermem or DmaBuf
        results.append(check_peermem_or_dmabuf())
        results.append(check_rdma_peer_type_config())
        results.append(check_rdma_dev_addr_list(normalized))
        results.append(check_peer_distance())
    elif rdma_type == "kernel":
        # Lustre/BeeGFS/NFS: kernel-level RDMA — nvidia_peermem not required
        results.append(check_peer_distance())

    return results
