# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Container observability checks for GDS diagnostics.

These helpers answer a narrow question: can this process see enough host
state for gds-diag subcommands to produce meaningful results from inside a
container? They intentionally do not decide whether the host itself is GDS
capable.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable

from .result import CheckResult, GDSMode, Status


@dataclass(frozen=True)
class ContainerContext:
    in_container: bool
    runtime: str
    evidence: str


def _exists_any(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def _read_text(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def detect_container_context() -> ContainerContext:
    evidence: list[str] = []
    runtime = "unknown"

    if os.path.exists("/.dockerenv"):
        runtime = "docker"
        evidence.append("/.dockerenv exists")
    env_keys = [k for k in os.environ if k.startswith("ENROOT_")]
    if env_keys:
        runtime = "enroot"
        evidence.append("container environment variables: " + ", ".join(sorted(env_keys)[:6]))

    cgroup = _read_text("/proc/self/cgroup")
    cgroup_markers = [
        marker for marker in ("docker", "kubepods", "containerd", "enroot")
        if marker in cgroup.lower()
    ]
    if cgroup_markers:
        if runtime == "unknown":
            runtime = cgroup_markers[0]
        evidence.append("/proc/self/cgroup contains: " + ", ".join(cgroup_markers))

    # On a cgroup v2 host, current OCI runtimes (runc, crun) default to giving
    # each container its own private cgroup namespace. From inside such a
    # container "/proc/self/cgroup" always reads "0::/", with no runtime name
    # in the path -- this is not specific to any one orchestrator or runtime
    # (containerd, CRI-O, Podman, and modern Docker all do it). Fall back to
    # markers that don't depend on that path being visible.
    kubernetes_markers = []
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        kubernetes_markers.append("KUBERNETES_SERVICE_HOST environment variable is set")
    if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount"):
        kubernetes_markers.append("/var/run/secrets/kubernetes.io/serviceaccount is mounted")
    if kubernetes_markers:
        if runtime == "unknown":
            runtime = "kubepods"
        evidence.append("Kubernetes pod markers: " + ", ".join(kubernetes_markers))

    mountinfo = _read_text("/proc/self/mountinfo")
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        mount_point = fields[4]
        if mount_point != "/":
            continue
        if "/enroot/" in line:
            if runtime == "unknown":
                runtime = "enroot"
            evidence.append("/proc/self/mountinfo root mount references Enroot data path")
            break
        if "containerd" in line.lower():
            if runtime == "unknown":
                runtime = "containerd"
            evidence.append("/proc/self/mountinfo root mount references a containerd snapshot")
            break

    return ContainerContext(
        in_container=bool(evidence),
        runtime=runtime if evidence else "host",
        evidence="\n".join(evidence) if evidence else "No common container markers found.",
    )


def find_gdscheck() -> str | None:
    candidates = (
        _exists_any((
            "/usr/local/cuda*/gds/tools/gdscheck",
            "/usr/local/cuda/gds/tools/gdscheck",
        ))
        + ([shutil.which("gdscheck")] if shutil.which("gdscheck") else [])
    )
    return candidates[0] if candidates else None


def find_libcufile() -> list[str]:
    return _exists_any((
        "/usr/local/cuda*/lib64/libcufile.so*",
        "/usr/lib*/libcufile.so*",
        "/lib*/libcufile.so*",
    ))


def _command_works(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return False, f"{command[0]} not found"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return True, output.splitlines()[0] if output else "command succeeded"
    return False, output.splitlines()[0] if output else f"exit {result.returncode}"


def _result(
    check: str,
    status: Status,
    why: str,
    mitigation: str | None = None,
    evidence: str | None = None,
    mode: GDSMode = GDSMode.NATIVE,
) -> CheckResult:
    return CheckResult(check=check, mode=mode, status=status, why=why, mitigation=mitigation, evidence=evidence)


def _runtime_family(runtime: str) -> str:
    runtime = runtime.lower()
    if runtime in {"docker", "containerd", "kubepods"}:
        return "docker"
    if runtime == "enroot":
        return "enroot"
    if runtime == "host":
        return "host"
    return "unknown"


def _runtime_mitigation(runtime: str, issue: str) -> str:
    family = _runtime_family(runtime)
    messages = {
        "nvidia_smi_missing": {
            "docker": (
                "Launch with Docker `--gpus=all` and the NVIDIA container runtime so host "
                "driver tools and NVML libraries are injected for diagnostics."
            ),
            "enroot": (
                "Add nvidia-smi only if this Enroot diagnostic container must confirm GPU state "
                "with NVIDIA tools; verify NVIDIA hooks mount host userspace tools if needed."
            ),
            "generic": (
                "Install or bind nvidia-smi only if this diagnostic container must confirm GPU "
                "state with NVIDIA tools; it is not by itself a GDS runtime requirement."
            ),
        },
        "nvidia_smi_unusable": {
            "docker": (
                "Launch with Docker `--gpus=all` and the NVIDIA container runtime; verify the "
                "host driver with `nvidia-smi -L` on the host."
            ),
            "enroot": (
                "Check Enroot NVIDIA hook configuration, usually /etc/enroot/hooks.d/98-nvidia.sh, "
                "and verify nvidia-container-cli plus host `nvidia-smi -L` work."
            ),
            "generic": (
                "Expose GPU devices with the container runtime and verify the host driver with "
                "`nvidia-smi -L` on the host."
            ),
        },
        "gpu_devices_missing": {
            "docker": "Launch with Docker `--gpus=all` and the NVIDIA container runtime.",
            "enroot": (
                "Check Enroot NVIDIA hook configuration, usually /etc/enroot/hooks.d/98-nvidia.sh, "
                "and verify ENROOT_RESTRICT_DEV is not hiding NVIDIA devices. If site hooks do not "
                "expose host devices, add `--mount /dev:/dev:none:rbind,ro` to the Enroot start command."
            ),
            "generic": "Expose NVIDIA GPU character devices through the container runtime.",
        },
        "gdscheck_missing": {
            "docker": "Install gds-tools or bind host CUDA/GDS tools only if this Docker container must collect live GDS route tokens.",
            "enroot": "Install gds-tools or add an Enroot mount for host CUDA/GDS tools only if live GDS route tokens are needed.",
            "generic": "Install gds-tools or bind host CUDA/GDS tools only if live GDS route tokens are needed.",
        },
        "libcufile_missing": {
            "docker": "Use a CUDA/GDS-capable image or bind host CUDA libraries into the Docker container.",
            "enroot": "Use a CUDA/GDS-capable image or add an Enroot mount for host CUDA libraries.",
            "generic": "Use a CUDA/GDS-capable image or bind the host CUDA libraries into the container.",
        },
        "cufile_config_missing": {
            "docker": "Bind /etc/cufile.json into Docker or set CUFILE_ENV_PATH_JSON to a visible config file.",
            "enroot": "Mount /etc/cufile.json with Enroot or set CUFILE_ENV_PATH_JSON to a visible config file.",
            "generic": "Provide /etc/cufile.json or set CUFILE_ENV_PATH_JSON to a config file visible inside the container.",
        },
        "nvidia_fs_missing": {
            "docker": (
                "Map each host /dev/nvidia-fs* node with Docker `--device`. "
                "Use Docker `--privileged` only as a broader diagnostic shortcut."
            ),
            "enroot": (
                "Expose host /dev/nvidia-fs* nodes with Enroot device hooks or site policy. If those "
                "nodes are still filtered, add `--mount /dev:/dev:none:rbind,ro` for validation."
            ),
            "generic": "Expose /dev/nvidia-fs* device nodes when validating native nvidia-fs/nvfs paths.",
        },
        "proc_driver_missing": {
            "docker": "Do not bind-mount over /proc/driver/nvidia directly; rely on procfs/NVIDIA runtime exposure.",
            "enroot": "Check Enroot NVIDIA hook/procfs handling if kernel-driver metadata is needed.",
            "generic": "Ensure /proc/driver/nvidia is visible if kernel-driver metadata is needed.",
        },
        "sysfs_missing": {
            "docker": "Bind /sys read-only into the Docker diagnostic container.",
            "enroot": "Mount /sys read-only with Enroot if the default site configuration does not expose it.",
            "generic": "Expose /sys read-only when running topology-sensitive diagnostics.",
        },
        "udev_missing": {
            "docker": "Bind /run/udev:/run/udev:ro into Docker containers used for GDS workloads or validation.",
            "enroot": "Add `-m /run/udev:/run/udev:none:x-create=dir,rbind,ro:0:0` to the Enroot start command.",
            "generic": "Expose /run/udev read-only so GDS can discover block-device metadata from inside the container.",
        },
        "rdma_cm_missing": {
            "docker": "Expose /dev/infiniband/rdma_cm and consider host networking for RDMA-backed Docker workloads.",
            "enroot": (
                "Check Enroot Mellanox hook configuration, usually /etc/enroot/hooks.d/99-mellanox.sh. "
                "If RDMA devices remain hidden, add `--mount /dev:/dev:none:rbind,ro` for validation."
            ),
            "generic": "Expose /dev/infiniband/rdma_cm for RDMA-backed routes.",
        },
        "uverbs_missing": {
            "docker": "Expose /dev/infiniband/uverbs* for WekaFS, GPFS, NFS/RDMA, or NVMe-oF validation in Docker.",
            "enroot": (
                "Check Enroot Mellanox hook configuration and site device policy for /dev/infiniband/uverbs*. "
                "If RDMA devices remain hidden, add `--mount /dev:/dev:none:rbind,ro` for validation."
            ),
            "generic": "Expose /dev/infiniband/uverbs* for WekaFS, GPFS, NFS/RDMA, or NVMe-oF validation.",
        },
        "post_partial": {
            "docker": "Expose GPUs, gdscheck, libcufile, /etc/cufile.json, and GDS device nodes before treating Docker post-install output as host truth.",
            "enroot": "Verify Enroot NVIDIA/Mellanox hooks plus CUDA/GDS mounts before treating Enroot post-install output as host truth.",
            "generic": "Expose GPUs, gdscheck, libcufile, /etc/cufile.json, and GDS device nodes before treating post-install output as host truth.",
        },
        "gdscheck_live": {
            "docker": "Bind or install gdscheck in Docker to collect live driver/client mode tokens.",
            "enroot": "Install gdscheck in the image or mount host CUDA/GDS tools with Enroot.",
            "generic": "Bind or install gdscheck to collect live driver/client mode tokens.",
        },
        "mount_check": {
            "docker": "Bind the target path, /sys, /run/udev, /dev/nvidia-fs*, and RDMA devices into Docker as needed.",
            "enroot": (
                "Mount the target path, CUDA/GDS tools, /run/udev, and required devices with Enroot. "
                "Use `--mount /dev:/dev:none:rbind,ro` when site hooks do not expose the needed device nodes."
            ),
            "generic": "Bind the target path, /sys, /run/udev, /dev/nvidia-fs*, and RDMA devices as needed.",
        },
    }
    values = messages[issue]
    return values.get(family, values["generic"])


def collect_sections(context: ContainerContext | None = None) -> dict[str, list[CheckResult]]:
    context = context or detect_container_context()
    runtime = context.runtime
    forced_runtime = context.evidence.startswith("Runtime forced by --runtime ")
    if not context.in_container:
        context_finding = "No common container markers were detected."
    elif forced_runtime:
        article = "an" if context.runtime == "enroot" else "a"
        context_finding = f"Assuming {article} {context.runtime} container because --runtime {context.runtime} was supplied."
    else:
        context_finding = f"Running inside a likely {context.runtime} container."
    sections: dict[str, list[CheckResult]] = {
        "Container Context": [
            _result(
                "Container runtime",
                Status.INFO if context.in_container else Status.PASS,
                context_finding,
                mitigation=(
                    "Use this subcommand to decide whether post-install, mount-check, "
                    "and support-matrix can validate GDS from the current container."
                    if context.in_container
                    else None
                ),
                evidence=context.evidence,
            )
        ],
        "Diagnostic Tools": [],
        "GDS Runtime Files": [],
        "Runtime Devices": [],
        "Host Metadata For Diagnostics": [],
        "Command Readiness": [],
    }

    nvidia_devices = _exists_any(("/dev/nvidia[0-9]*", "/dev/nvidiactl", "/dev/nvidia-uvm"))
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        nvidia_smi_ok, nvidia_smi_evidence = _command_works(["nvidia-smi", "-L"])
        nvidia_smi_why = (
            "nvidia-smi can list GPU devices for diagnostics."
            if nvidia_smi_ok
            else "nvidia-smi is installed, but it cannot list GPUs from this container. This limits diagnostics; it is not by itself a GDS runtime blocker."
        )
        nvidia_smi_mitigation = (
            "No action needed."
            if nvidia_smi_ok
            else _runtime_mitigation(runtime, "nvidia_smi_unusable")
        )
    else:
        nvidia_smi_ok = False
        nvidia_smi_evidence = "nvidia-smi not found in PATH"
        nvidia_smi_why = "nvidia-smi is not installed or not present in this container PATH. This limits diagnostics; it is not by itself a GDS runtime blocker."
        nvidia_smi_mitigation = _runtime_mitigation(runtime, "nvidia_smi_missing")
    sections["Diagnostic Tools"].append(
        _result(
            "nvidia-smi",
            Status.PASS if nvidia_smi_ok else Status.INFO,
            nvidia_smi_why,
            mitigation=nvidia_smi_mitigation,
            evidence=nvidia_smi_evidence,
        )
    )

    gdscheck = find_gdscheck()
    sections["Diagnostic Tools"].append(
        _result(
            "gdscheck",
            Status.PASS if gdscheck else Status.INFO,
            (
                "gdscheck is visible for live GDS route/token diagnostics."
                if gdscheck
                else "gdscheck is not visible. Live GDS route/token inspection is unavailable; GDS workloads may still work if libcufile, configuration, and required devices are present."
            ),
            mitigation="No action needed." if gdscheck else _runtime_mitigation(runtime, "gdscheck_missing"),
            evidence=gdscheck,
        )
    )

    sections["Runtime Devices"].extend([
        _result(
            "NVIDIA GPU devices",
            Status.PASS if nvidia_devices else Status.WARN,
            (
                "NVIDIA GPU character devices are visible for GPU-memory GDS."
                if nvidia_devices
                else "No /dev/nvidia* GPU character devices are visible. GPU-memory GDS workloads cannot use GPUs from this container until GPU devices are exposed."
            ),
            mitigation="No action needed." if nvidia_devices else _runtime_mitigation(runtime, "gpu_devices_missing"),
            evidence="\n".join(nvidia_devices[:12]) if nvidia_devices else None,
        ),
    ])

    libcufile = find_libcufile()
    nvidia_fs_devices = _exists_any(("/dev/nvidia-fs*",))
    proc_driver = os.path.exists("/proc/driver/nvidia")
    config_path = os.environ.get("CUFILE_ENV_PATH_JSON") or "/etc/cufile.json"
    config_exists = os.path.exists(config_path)
    sections["GDS Runtime Files"].extend([
        _result(
            "libcufile",
            Status.PASS if libcufile else Status.WARN,
            "libcufile is visible for GDS workloads." if libcufile else "libcufile was not found in common container library paths.",
            mitigation="No action needed." if libcufile else _runtime_mitigation(runtime, "libcufile_missing"),
            evidence="\n".join(libcufile[:8]) if libcufile else None,
        ),
        _result(
            "cuFile config",
            Status.PASS if config_exists else Status.WARN,
            f"cuFile configuration is visible at {config_path}." if config_exists else f"cuFile configuration is not visible at {config_path}.",
            mitigation="No action needed." if config_exists else _runtime_mitigation(runtime, "cufile_config_missing"),
            evidence=config_path if config_exists else None,
        ),
    ])
    sections["Runtime Devices"].append(
        _result(
            "nvidia-fs devices",
            Status.PASS if nvidia_fs_devices else Status.WARN,
            (
                "nvidia-fs character devices are visible for native nvfs paths."
                if nvidia_fs_devices
                else (
                    "No /dev/nvidia-fs* devices are visible. Native nvidia-fs/nvfs "
                    "paths will not be available, but P2PDMA/C2C or compat routes "
                    "may still work when otherwise supported and configured."
                )
            ),
            mitigation="No action needed." if nvidia_fs_devices else _runtime_mitigation(runtime, "nvidia_fs_missing"),
            evidence="\n".join(nvidia_fs_devices[:16]) if nvidia_fs_devices else None,
        )
    )
    sections["Host Metadata For Diagnostics"].append(
        _result(
            "NVIDIA procfs",
            Status.PASS if proc_driver else Status.INFO,
            "/proc/driver/nvidia is visible." if proc_driver else "/proc/driver/nvidia is not visible; kernel-driver metadata may be incomplete.",
            mitigation="No action needed." if proc_driver else _runtime_mitigation(runtime, "proc_driver_missing"),
        )
    )

    sys_pci = os.path.isdir("/sys/bus/pci/devices")
    run_udev = False
    udev_evidence = None
    if os.path.isdir("/run/udev"):
        try:
            run_udev = bool(os.listdir("/run/udev"))
        except OSError as exc:
            udev_evidence = f"Unable to list /run/udev: {type(exc).__name__}: {exc}"
    mountinfo_exists = os.path.exists("/proc/self/mountinfo")
    sections["Runtime Devices"].append(
        _result(
            "udev database",
            Status.PASS if run_udev else Status.WARN,
            (
                "/run/udev is visible and non-empty for GDS device discovery."
                if run_udev
                else (
                    "/run/udev is missing or empty. GDS may be unable to resolve "
                    "block-device metadata correctly from inside this container."
                )
            ),
            mitigation="No action needed." if run_udev else _runtime_mitigation(runtime, "udev_missing"),
            evidence=udev_evidence,
        )
    )
    sections["Host Metadata For Diagnostics"].extend([
        _result(
            "sysfs PCI topology",
            Status.PASS if sys_pci else Status.WARN,
            "/sys/bus/pci/devices is visible for topology and ACS checks." if sys_pci else "PCI sysfs is not visible; topology and ACS checks will be incomplete.",
            mitigation="No action needed." if sys_pci else _runtime_mitigation(runtime, "sysfs_missing"),
        ),
        _result(
            "mountinfo",
            Status.PASS if mountinfo_exists else Status.FAIL,
            "/proc/self/mountinfo is visible for mount attribution." if mountinfo_exists else "/proc/self/mountinfo is missing.",
            mitigation="Run in a normal Linux container namespace with procfs mounted.",
        ),
    ])

    rdma_cm = os.path.exists("/dev/infiniband/rdma_cm")
    uverbs = _exists_any(("/dev/infiniband/uverbs*",))
    sections["Runtime Devices"].extend([
        _result(
            "RDMA connection manager",
            Status.PASS if rdma_cm else Status.INFO,
            (
                "/dev/infiniband/rdma_cm is visible for RDMA-backed route diagnostics."
                if rdma_cm
                else (
                    "/dev/infiniband/rdma_cm is not visible. This only matters for "
                    "RDMA-backed GDS routes such as NFS/RDMA, NVMe-oF, WekaFS, or GPFS."
                )
            ),
            mitigation="No action needed." if rdma_cm else _runtime_mitigation(runtime, "rdma_cm_missing"),
            mode=GDSMode.RDMA,
        ),
        _result(
            "RDMA uverbs devices",
            Status.PASS if uverbs else Status.INFO,
            (
                "RDMA uverbs devices are visible for RDMA-backed route diagnostics."
                if uverbs
                else (
                    "No /dev/infiniband/uverbs* devices are visible. This only matters "
                    "for RDMA-backed GDS routes such as NFS/RDMA, NVMe-oF, WekaFS, or GPFS."
                )
            ),
            mitigation="No action needed." if uverbs else _runtime_mitigation(runtime, "uverbs_missing"),
            evidence="\n".join(uverbs[:12]) if uverbs else None,
            mode=GDSMode.RDMA,
        ),
    ])

    support_matrix_ready = Status.PASS if gdscheck else Status.INFO
    config_ready = Status.PASS if (gdscheck or config_exists) else Status.WARN
    mount_ready = Status.PASS if (
        mountinfo_exists and sys_pci and run_udev and (nvidia_devices or nvidia_smi_ok)
    ) else Status.WARN
    runtime_ready = bool(nvidia_devices and libcufile and config_exists)
    post_ready = Status.PASS if (nvidia_smi_ok and gdscheck and config_exists) else (
        Status.WARN if not runtime_ready else Status.INFO
    )
    sections["Command Readiness"].extend([
        _result(
            "pre-install",
            Status.INFO if context.in_container else Status.PASS,
            "pre-install is a host-readiness command; container output may reflect container visibility rather than host install state.",
            mitigation="Prefer running pre-install on the host. Use container-check inside containers.",
        ),
        _result(
            "post-install",
            post_ready,
            (
                "post-install has enough diagnostic visibility."
                if post_ready == Status.PASS
                else "post-install may be partial because runtime inputs are missing from this container."
                if post_ready == Status.WARN
                else "post-install can inspect core runtime inputs, but diagnostic tools are missing so live validation may be partial."
            ),
            mitigation="No action needed." if post_ready == Status.PASS else _runtime_mitigation(runtime, "post_partial"),
        ),
        _result(
            "support-matrix --live",
            support_matrix_ready,
            "support-matrix can use live gdscheck evidence." if gdscheck else "support-matrix will fall back to the documentation/version matrix without gdscheck.",
            mitigation="No action needed." if gdscheck else _runtime_mitigation(runtime, "gdscheck_live"),
        ),
        _result(
            "config-audit",
            config_ready,
            "config-audit can inspect gdscheck or a visible cuFile config." if config_ready == Status.PASS else "config-audit cannot see gdscheck or a cuFile config file.",
            mitigation="No action needed." if config_ready == Status.PASS else _runtime_mitigation(runtime, "cufile_config_missing"),
        ),
        _result(
            "mount-check",
            mount_ready,
            (
                "mount-check has the main host-observability inputs."
                if mount_ready == Status.PASS
                else "mount-check may be incomplete from this container."
            ),
            mitigation="No action needed." if mount_ready == Status.PASS else _runtime_mitigation(runtime, "mount_check"),
        ),
    ])

    return sections


