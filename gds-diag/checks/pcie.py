# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PCIe topology and ACS (Access Control Services) checks for GDS P2PDMA.

P2PDMA topology guidance:
  1. GPU and NVMe on a close PCIe path (same root complex or P2P-capable switch)
     is optimal. Cross-root-port paths can still work, but are typically slower.
  2. ACS P2P Request Redirect (bit 2 of ACSCtl) NOT enabled on any bridge in the path.
     When set, P2P transactions are rerouted through the root complex, destroying direct DMA.
  3. IOMMU not in strict mode (covered in iommu.py).

ACS bits in ACSCtl that matter:
  ReqRedir (bit 2) — redirects P2P requests upstream → blocks P2P DMA
  CmpltRedir (bit 3) — redirects completions upstream

Detection approach:
  - GPU BDFs: nvidia-smi
  - NVMe BDFs: /sys/block/nvme*/device symlink
  - PCIe tree: /sys/bus/pci/devices/<bdf>/ parent path analysis
  - ACS: lspci -vvv output parsing
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from .result import CheckResult, GDSMode, Status

_BDF4_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", re.I)
_BDF8_RE = re.compile(r"^([0-9a-f]{8}):([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])$", re.I)
_BDF_SHORT_RE = re.compile(r"^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", re.I)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    """Remove terminal styling sequences that nvidia-smi may emit in tables."""
    return _ANSI_RE.sub("", text)


def _normalize_bdf(value: str) -> Optional[str]:
    """
    Normalize PCI BDF strings to Linux sysfs form: dddd:bb:dd.f.

    nvidia-smi commonly prints GPU PCI bus IDs with an 8-hex-digit domain
    (00000000:65:00.0), while Linux sysfs uses a 4-hex-digit domain
    (0000:65:00.0). Keep the low 16 bits of the domain so the value can be
    compared with /sys/bus/pci/devices entries.
    """
    bdf = re.sub(r"^gpu-", "", value.strip().lower())
    m8 = _BDF8_RE.match(bdf)
    if m8:
        return f"{m8.group(1)[-4:]}:{m8.group(2)}"
    if _BDF4_RE.match(bdf):
        return bdf
    if _BDF_SHORT_RE.match(bdf):
        return f"0000:{bdf}"
    return None


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

def _query_gpu_bdfs() -> tuple[list[str], Optional[str]]:
    """
    Return (GPU PCI BDFs, error) via nvidia-smi.

    nvidia-smi can exist but still fail to communicate with the running NVIDIA
    driver. Keep that error text so callers can explain the real discovery
    failure instead of implying the binary is missing.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return [], msg or f"nvidia-smi exited with code {result.returncode}"
        bdfs = []
        for line in result.stdout.splitlines():
            bdf = _normalize_bdf(line)
            if bdf:
                bdfs.append(bdf)
        if not bdfs:
            return [], "nvidia-smi returned no GPU PCI bus IDs"
        return bdfs, None
    except FileNotFoundError:
        return [], "nvidia-smi not found in PATH"
    except Exception as exc:
        return [], f"nvidia-smi GPU query failed: {exc}"


def get_gpu_bdfs() -> list[str]:
    """Return list of GPU PCI BDFs (e.g. '0000:03:00.0') via nvidia-smi."""
    bdfs, _ = _query_gpu_bdfs()
    return bdfs


def get_gpu_bdfs_with_error() -> tuple[list[str], Optional[str]]:
    """Return GPU PCI BDFs plus a human-readable discovery error, if any."""
    return _query_gpu_bdfs()


def get_nvme_bdfs() -> list[str]:
    """Return list of NVMe controller PCI BDFs from /sys/block/nvme*."""
    bdfs = []
    try:
        for name in os.listdir("/sys/block"):
            if not name.startswith("nvme"):
                continue
            link = f"/sys/block/{name}/device"
            if os.path.islink(link):
                # Use realpath — readlink returns a relative symlink (e.g. ../../nvme0)
                # that doesn't contain the PCI BDF. realpath resolves to the full
                # /sys/devices/pciXXXX:YY/.../0000:ZZ:00.0/nvme/nvme0 path.
                target = os.path.realpath(link)
                parts = target.split("/")
                for part in reversed(parts):
                    if re.match(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", part):
                        bdfs.append(part)
                        break
    except FileNotFoundError:
        pass
    return list(set(bdfs))


def get_nvme_bdf_for_device(device: str) -> Optional[str]:
    """Return the PCI BDF backing a Linux NVMe namespace device, if available."""
    name = os.path.basename(device)
    m = re.match(r"^(nvme\d+n\d+)(p\d+)?$", name)
    block = m.group(1) if m else name
    link = f"/sys/block/{block}/device"
    if not os.path.exists(link):
        return None

    try:
        target = os.path.realpath(link)
    except OSError:
        return None

    for part in reversed(target.split("/")):
        bdf = _normalize_bdf(part)
        if bdf:
            return bdf
    return None


# ---------------------------------------------------------------------------
# nvidia-smi topo -m -nvme parsing
# ---------------------------------------------------------------------------

_TOPO_RANK = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
    "SOC": 5,
}


def _run_nvidia_smi_topo_nvme() -> tuple[str, Optional[str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m", "-nvme"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            return "", msg or f"nvidia-smi topo exited with code {result.returncode}"
        return result.stdout, None
    except FileNotFoundError:
        return "", "nvidia-smi not found in PATH"
    except Exception as exc:
        return "", f"nvidia-smi topo -m -nvme failed: {exc}"


def _is_gpu_label(label: str) -> bool:
    return bool(re.match(r"^GPU\d+$", label.strip(), re.I))


def _is_nvme_label(label: str) -> bool:
    normalized = label.strip().lower()
    return bool(re.match(r"^nvme\d+$", normalized))


def _is_nic_label(label: str) -> bool:
    return bool(re.match(r"^NIC\d+$", label.strip(), re.I))


def _is_topo_relation(label: str) -> bool:
    rel = label.strip().upper()
    return rel in {"X", "NV#"} or rel in _TOPO_RANK or bool(re.match(r"^NV\d+$", rel))


def _topo_header_columns(line: str, tokens: list[str]) -> list[str]:
    """
    Return topology matrix columns for a header row, or [] for data/legend rows.

    Driver versions differ here. Some print one matrix with GPU/NVMe columns;
    others print a GPU/NIC matrix first, then a second NVMe-only matrix:

        GPU0    NIC0    CPU Affinity ...
    GPU0 ...
    ...
        NVMe0   NVMe1
    GPU0    PHB     PHB

    Keep updating the active columns whenever a new indented device-header row
    appears so the second NVMe matrix is parsed too.
    """
    columns: list[str] = []
    for tok in tokens:
        label = tok.rstrip(":=")
        if label.upper() in {"CPU", "NUMA", "GPU"}:
            break
        if _is_topo_relation(label):
            return []
        if _is_gpu_label(label) or _is_nvme_label(label) or _is_nic_label(label):
            columns.append(label)
            continue
        return []

    return columns


def _topo_relation_rank(relation: str) -> int:
    rel = relation.strip().upper()
    if rel == "X":
        return -1
    return _TOPO_RANK.get(rel, 99)


def _parse_nvme_split_tables(output: str) -> dict[str, dict[str, str]]:
    """
    Parse split NVMe tables independent of the main rolling-header parser.

    Some nvidia-smi versions print GPU/NIC topology, a legend, then a compact
    NVMe matrix:

        NVMe0   NVMe1
    GPU0        PHB     PHB
    Legend:
    ...
    NVMe Legend:
      NVMe0: nvme0n1

    This fallback looks for an all-NVMe header and nearby GPU rows directly.
    """
    lines = output.splitlines()
    gpu_to_nvme: dict[str, dict[str, str]] = {}

    for idx, raw in enumerate(lines):
        tokens = raw.split()
        if not tokens:
            continue

        header = [tok.rstrip(":=") for tok in tokens]
        if not header or not all(_is_nvme_label(tok) for tok in header):
            continue

        for row_raw in lines[idx + 1: idx + 12]:
            row_tokens = row_raw.split()
            if not row_tokens:
                continue

            row_label = row_tokens[0].rstrip(":=")
            if row_label.lower().startswith("legend") or row_label.lower() == "nic":
                break
            if not _is_gpu_label(row_label):
                continue

            values = row_tokens[1:1 + len(header)]
            if len(values) < len(header):
                continue
            relations = {
                nvme: rel
                for nvme, rel in zip(header, values)
                if _is_topo_relation(rel)
            }
            if relations:
                gpu_to_nvme.setdefault(row_label, {}).update(relations)

    return gpu_to_nvme


def parse_topo_nvme(output: str) -> dict[str, object]:
    """
    Parse `nvidia-smi topo -m -nvme` output.

    The exact NVMe labels vary across driver versions, so this parser keeps the
    labels as reported by nvidia-smi instead of assuming a specific NVMe0/nvme0n1
    mapping. It extracts the GPU rows and GPU->NVMe relationship cells.
    """
    columns: list[str] = []
    gpu_to_nvme: dict[str, dict[str, str]] = {}
    nvme_details: dict[str, dict[str, object]] = {}

    for raw in output.splitlines():
        line = _strip_ansi(raw).rstrip()
        if not line.strip():
            continue
        tokens = line.split()
        if not tokens:
            continue

        label = tokens[0].rstrip(":=")
        if _is_nvme_label(label):
            detail = line[len(tokens[0]):].strip(" \t:=")
            bdf = None
            for match in re.finditer(
                r"\b(?:"
                r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]|"
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]|"
                r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]"
                r")\b",
                detail,
                re.I,
            ):
                bdf = _normalize_bdf(match.group(0))
                if bdf:
                    break
            devices = re.findall(r"(?:/dev/)?(nvme\d+n\d+(?:p\d+)?)\b", detail, re.I)
            if bdf or devices:
                nvme_details.setdefault(label, {})
                if bdf:
                    nvme_details[label]["bdf"] = bdf
                if devices:
                    nvme_details[label]["devices"] = [d.lower() for d in devices]
                nvme_details[label]["detail"] = detail

        header_columns = _topo_header_columns(line, tokens)
        if header_columns:
            columns = header_columns
            continue

        row_label = tokens[0]
        if not _is_gpu_label(row_label) or not columns:
            continue

        values = tokens[1:1 + len(columns)]
        if len(values) < len(columns):
            continue

        relations: dict[str, str] = {}
        for col, val in zip(columns, values):
            if _is_nvme_label(col):
                relations[col] = val
        if relations:
            gpu_to_nvme[row_label] = relations

    if not gpu_to_nvme:
        gpu_to_nvme = _parse_nvme_split_tables(output)

    nvme_labels = sorted({nvme for relations in gpu_to_nvme.values() for nvme in relations})
    return {
        "columns": columns,
        "gpus": sorted(gpu_to_nvme),
        "nvmes": nvme_labels,
        "nvme_details": nvme_details,
        "gpu_to_nvme": gpu_to_nvme,
    }


def _topo_output_preview(output: str, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in _strip_ansi(output).splitlines() if line.strip()]
    if not lines:
        return "<empty output>"

    for idx, line in enumerate(lines):
        tokens = line.split()
        if any(_is_nvme_label(tok.rstrip(":=")) for tok in tokens):
            start = max(0, idx - 3)
            end = min(len(lines), idx + max_lines)
            return "\n".join(lines[start:end])

    return "\n".join(lines[:max_lines])


def _target_nvme_labels(parsed: dict[str, object], target_nvme: Optional[str]) -> list[str]:
    nvmes = list(parsed.get("nvmes", []))
    if not target_nvme:
        return []

    target = os.path.basename(target_nvme).lower()
    m = re.match(r"^(nvme\d+n\d+)(p\d+)?$", target)
    target_base = m.group(1) if m else re.sub(r"p?\d+$", "", target)
    target_controller = None
    if m:
        ctrl_m = re.match(r"^(nvme\d+)n\d+", target_base)
        if ctrl_m:
            target_controller = ctrl_m.group(1)
    target_bdf = get_nvme_bdf_for_device(target_nvme)
    details = parsed.get("nvme_details", {})
    matches = []
    for label in nvmes:
        label_lower = label.lower()
        info = details.get(label, {}) if isinstance(details, dict) else {}
        devices = info.get("devices", []) if isinstance(info, dict) else []
        detail_bdf = info.get("bdf") if isinstance(info, dict) else None
        if (
            target in label_lower
            or target_base in label_lower
            or label_lower in {target, target_base}
            or (target_controller and label_lower == target_controller)
            or target in devices
            or target_base in devices
            or (target_bdf and detail_bdf == target_bdf)
        ):
            matches.append(label)
    return matches


def topo_nvme_recommendations(target_nvme: Optional[str] = None) -> tuple[list[str], Optional[str]]:
    output, err = _run_nvidia_smi_topo_nvme()
    if err:
        return [], err

    parsed = parse_topo_nvme(output)
    gpu_to_nvme = parsed.get("gpu_to_nvme", {})
    if not isinstance(gpu_to_nvme, dict) or not gpu_to_nvme:
        return [], (
            "nvidia-smi topo -m -nvme returned no parsed GPU/NVMe relationships. "
            "Output preview:\n" + _topo_output_preview(output)
        )

    lines: list[str] = []
    target_labels = _target_nvme_labels(parsed, target_nvme)

    if target_nvme and target_labels:
        for nvme in target_labels:
            gpu_choices = []
            for gpu, relations in gpu_to_nvme.items():
                if isinstance(relations, dict) and nvme in relations:
                    gpu_choices.append((gpu, relations[nvme]))
            gpu_choices.sort(key=lambda item: (_topo_relation_rank(item[1]), item[0]))
            if gpu_choices:
                rendered = ", ".join(f"{gpu} ({rel})" for gpu, rel in gpu_choices[:4])
                lines.append(f"For {nvme}, prefer GPU(s): {rendered}")
    elif target_nvme:
        lines.append(
            f"nvidia-smi topo did not expose a label matching {target_nvme}; "
            "showing best NVMe choice per GPU instead."
        )

    best_pairs = []
    for gpu, relations in sorted(gpu_to_nvme.items()):
        if not isinstance(relations, dict) or not relations:
            continue
        ranked = sorted(relations.items(), key=lambda item: (_topo_relation_rank(item[1]), item[0]))
        best_rank = _topo_relation_rank(ranked[0][1])
        best_rel = ranked[0][1]
        best_nvmes = [nvme for nvme, rel in ranked if _topo_relation_rank(rel) == best_rank]
        best_pairs.append(f"{gpu} -> {', '.join(best_nvmes)} ({best_rel})")
    if best_pairs:
        lines.append("Best NVMe per GPU: " + "; ".join(best_pairs[:8]))

    return lines, None


def check_topo_nvme(target_nvme: Optional[str] = None) -> CheckResult:
    recs, err = topo_nvme_recommendations(target_nvme)
    if err:
        return CheckResult(
            check="nvidia-smi topo -m -nvme",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why=f"Could not read nvidia-smi NVMe topology: {err}",
            mitigation=(
                "Ensure the NVIDIA driver is healthy and run:\n"
                "  nvidia-smi topo -m -nvme"
            ),
        )

    return CheckResult(
        check="nvidia-smi topo -m -nvme",
        mode=GDSMode.P2PDMA,
        status=Status.PASS,
        why="nvidia-smi reports GPU/NVMe topology. Prefer the closest NVMe path for each GPU.",
        evidence="\n".join(recs),
    )


# ---------------------------------------------------------------------------
# PCIe topology analysis via sysfs
# ---------------------------------------------------------------------------

def _sysfs_ancestors(bdf: str) -> list[str]:
    """
    Walk /sys/bus/pci/devices/<bdf> up through its parents,
    returning BDFs of all ancestor PCIe devices up to the root.
    """
    ancestors = []
    path = f"/sys/bus/pci/devices/{bdf}"
    if not os.path.exists(path):
        path = f"/sys/bus/pci/devices/0000:{bdf}"
    try:
        current = os.path.realpath(path)
        while True:
            parent = os.path.dirname(current)
            parent_name = os.path.basename(parent)
            if re.match(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", parent_name):
                ancestors.append(parent_name)
                current = parent
            else:
                break
    except Exception:
        pass
    return ancestors


def find_common_ancestor(bdf_a: str, bdf_b: str) -> Optional[str]:
    """
    Return the BDF of the common PCIe ancestor of two devices,
    or None if they share only the host bridge (different root complexes).
    """
    ancestors_a = set(_sysfs_ancestors(bdf_a))
    ancestors_b = set(_sysfs_ancestors(bdf_b))
    common = ancestors_a & ancestors_b
    if not common:
        return None
    # Return the deepest common ancestor (longest BDF path = most specific)
    # In practice all BDFs are same length; sort by depth proxy
    return sorted(common)[-1]


def same_root_complex(bdf_a: str, bdf_b: str) -> tuple[Optional[bool], str]:
    """
    Return (same_complex, evidence).
    same_complex is True/False when topology could be determined via sysfs, or
    None when it could not (caller must not treat that as a confirmed match).
    Two devices are on the same root complex if they share a root port ancestor.
    """
    anc_a = _sysfs_ancestors(bdf_a)
    anc_b = _sysfs_ancestors(bdf_b)

    if not anc_a or not anc_b:
        return None, f"Could not determine topology for {bdf_a} ↔ {bdf_b} via sysfs"

    common = set(anc_a) & set(anc_b)
    if common:
        shared = sorted(common)[0]
        return True, f"{bdf_a} and {bdf_b} share ancestor {shared}"
    else:
        return False, (
            f"{bdf_a} ancestors: {anc_a}\n"
            f"{bdf_b} ancestors: {anc_b}\n"
            "No common ancestor — likely different root complexes (different CPU sockets)"
        )


# ---------------------------------------------------------------------------
# ACS detection via lspci
# ---------------------------------------------------------------------------

def _run_lspci_vvv() -> str:
    try:
        result = subprocess.run(
            ["lspci", "-vvv"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _parse_acs_redirect_devices(lspci_output: str) -> list[tuple[str, str]]:
    """
    Parse lspci -vvv output and return list of (bdf, acs_ctl_line) tuples
    where ACS P2P Request Redirect (ReqRedir+) is set.
    """
    offenders: list[tuple[str, str]] = []
    current_bdf = None

    for line in lspci_output.splitlines():
        # Device header: "03:00.0 ..."
        m = re.match(r"^([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])\s", line)
        if m:
            current_bdf = m.group(1)

        # ACSCtl line
        if "ACSCtl:" in line and current_bdf:
            # ReqRedir+ means bit 2 is set (redirect P2P requests) — blocks P2P DMA
            if "ReqRedir+" in line:
                offenders.append((current_bdf, line.strip()))

    return offenders


def check_acs() -> CheckResult:
    """Check whether any PCIe bridge has ACS P2P Request Redirect enabled."""
    lspci_out = _run_lspci_vvv()

    if not lspci_out:
        return CheckResult(
            check="PCIe ACS redirect", mode=GDSMode.P2PDMA, status=Status.WARN,
            why=(
                "lspci -vvv is unavailable or returned no output. "
                "Cannot determine whether ACS P2P Request Redirect is blocking P2PDMA."
            ),
            mitigation=(
                "Install pciutils: apt-get install pciutils / yum install pciutils\n"
                "Then run: sudo lspci -vvv | grep -A5 ACSCtl\n"
                "Look for 'ReqRedir+' — if present on any bridge, P2PDMA is blocked."
            ),
        )

    offenders = _parse_acs_redirect_devices(lspci_out)

    if not offenders:
        return CheckResult(
            check="PCIe ACS redirect", mode=GDSMode.P2PDMA, status=Status.PASS,
            why="No PCIe bridges have ACS P2P Request Redirect (ReqRedir+) enabled.",
        )

    bdf_list = ", ".join(bdf for bdf, _ in offenders)
    evidence = "\n".join(f"  {bdf}: {ctl}" for bdf, ctl in offenders)

    return CheckResult(
        check="PCIe ACS redirect", mode=GDSMode.P2PDMA, status=Status.FAIL,
        why=(
            f"ACS P2P Request Redirect (ReqRedir+) is set on PCIe bridge(s): {bdf_list}. "
            "This forces all P2P DMA transactions to be rerouted through the root complex, "
            "preventing direct GPU↔NVMe transfers."
        ),
        mitigation=(
            "Option 1 (recommended): Add 'pci=noacs' to GRUB_CMDLINE_LINUX in /etc/default/grub.\n"
            "  Note: this disables ACS globally, reducing PCIe device isolation (security tradeoff).\n"
            "Option 2: Selectively disable ACS on the offending bridge(s) using setpci:\n"
            f"  For each bridge in [{bdf_list}]:\n"
            "  sudo setpci -s <bdf> ECAP_ACS+6.w=0  # clears ACSCtl bits\n"
            "  (This resets on reboot; add to a systemd service for persistence.)\n"
            "Option 3: Check BIOS/firmware for a 'PCIe ACS Override' or 'Peer-to-Peer' setting.\n"
            "Option 4: Physically restructure so GPU and NVMe do not share an ACS-enabled bridge."
        ),
        evidence=evidence,
    )


def check_pcie_topology() -> list[CheckResult]:
    """
    Check PCIe topology between all GPU/NVMe pairs.
    Returns one result summarising compatibility.
    """
    results = []
    gpu_bdfs, gpu_error = _query_gpu_bdfs()
    nvme_bdfs = get_nvme_bdfs()

    if not gpu_bdfs:
        topo_result = check_topo_nvme()
        if topo_result.status == Status.PASS:
            topo_result.why = (
                "GPU PCI bus IDs were not available from nvidia-smi query, "
                "so using nvidia-smi topo -m -nvme for GPU/NVMe topology instead."
            )
            if gpu_error:
                topo_result.evidence = (
                    f"GPU BDF query detail: {gpu_error}\n"
                    + (topo_result.evidence or "")
                )
            results.append(topo_result)
            return results

        why = "No GPU topology could be discovered via nvidia-smi."
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
            why = f"{why} Detail: {'; '.join(details)}"
        results.append(CheckResult(
            check="PCIe topology (GPU discovery)", mode=GDSMode.P2PDMA, status=Status.WARN,
            why=why,
            mitigation=(
                "Ensure the NVIDIA kernel driver is loaded and healthy, then verify:\n"
                "  nvidia-smi -L\n"
                "If running in a container, ensure GPU devices and driver libraries are exposed."
            ),
        ))
        return results

    if not nvme_bdfs:
        results.append(CheckResult(
            check="PCIe topology (NVMe discovery)", mode=GDSMode.P2PDMA, status=Status.WARN,
            why=(
                "No NVMe block devices found in /sys/block. "
                "P2PDMA only applies to NVMe (local) storage — "
                "if using a network filesystem, this is expected and P2PDMA is not relevant."
            ),
        ))
        return results

    suboptimal_pairs: list[str] = []
    optimal_pairs: list[str] = []
    unknown_pairs: list[str] = []

    for gpu in gpu_bdfs:
        for nvme in nvme_bdfs:
            same, ev = same_root_complex(gpu, nvme)
            label = f"GPU {gpu} ↔ NVMe {nvme}"
            if same is None:
                unknown_pairs.append(f"{label}: {ev}")
            elif same:
                optimal_pairs.append(f"{label} (shared ancestor)")
            else:
                suboptimal_pairs.append(f"{label}: {ev}")

    if unknown_pairs:
        results.append(CheckResult(
            check="PCIe topology",
            mode=GDSMode.P2PDMA,
            status=Status.INFO,
            why=(
                f"PCIe topology could not be determined via sysfs for "
                f"{len(unknown_pairs)} GPU↔NVMe pair(s). Performance relative to "
                "PCIe placement cannot be assessed for these pairs."
            ),
            evidence="\n".join(unknown_pairs),
        ))

    if suboptimal_pairs:
        partial = bool(optimal_pairs)
        total_pairs = len(suboptimal_pairs) + len(optimal_pairs)
        results.append(CheckResult(
            check="PCIe topology",
            mode=GDSMode.P2PDMA,
            status=Status.INFO,
            why=(
                f"{len(suboptimal_pairs)} of {total_pairs} visible GPU↔NVMe pair(s) "
                "are on different PCIe root complexes or have no shared PCIe ancestor. "
                "GDS can still operate across root ports, but expect lower performance "
                "than a same-root-port or same-switch path. "
                "Rerun mount-check with --verbose to view the affected pair list."
                if not partial else
                f"{len(suboptimal_pairs)} of {total_pairs} visible GPU↔NVMe pair(s) "
                "are farther apart in the PCIe topology. GDS can operate across root "
                "ports, but these pairs are expected to be less performant than closer "
                "pairs. Rerun mount-check with --verbose to view the affected pair list."
            ),
            evidence="\n".join(suboptimal_pairs),
        ))

    if optimal_pairs:
        results.append(CheckResult(
            check="PCIe topology", mode=GDSMode.P2PDMA, status=Status.PASS,
            why=(
                "Closest GPU↔NVMe pairs found on the same PCIe root complex:\n" +
                "\n".join(f"  {p}" for p in optimal_pairs)
            ),
        ))

    topo_result = check_topo_nvme()
    if topo_result.status == Status.PASS:
        results.append(topo_result)

    return results


def run_all() -> list[CheckResult]:
    results = []
    results.extend(check_pcie_topology())
    results.append(check_acs())
    return results
