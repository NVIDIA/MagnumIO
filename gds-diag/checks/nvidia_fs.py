# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
nvidia-fs kernel module checks.

nvidia-fs.ko is the GDS kernel driver installed by NVIDIA GDS packages
(normally nvidia-gds, with nvidia-fs-dkms providing the kernel module).
It must be loaded, version-matched to the NVIDIA display driver, and
its dmesg output analysed to understand runtime mode decisions.

Also checks:
  - GPU compute capability (requires >= 6.0 for GDS, i.e. Pascal+)
  - gdscheck tool availability and output
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Optional

from .result import CheckResult, GDSMode, Status
from . import cufile_version, kmsg

# Minimum driver version for GDS support
MIN_DRIVER_VERSION = (460, 0)
CUDA_DOWNLOADS_URL = "https://developer.nvidia.com/cuda-downloads"
CUDA_LINUX_INSTALL_GUIDE_URL = (
    "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html"
)
REBUILD_INITRAMFS_CMDS = (
    "  sudo update-initramfs -u -k all   # Ubuntu/Debian\n"
    "  sudo dracut -f --regenerate-all   # RHEL/Rocky\n"
    "  sudo reboot"
)


def _driver_install_mitigation() -> str:
    return (
        "Install the NVIDIA driver from the CUDA Downloads page:\n"
        f"{CUDA_DOWNLOADS_URL}\n"
        "Select the Open Kernel Module / NVIDIA Open Driver option for GDS."
    )


def _nvidia_fs_package_mitigation(intro: str) -> str:
    return (
        f"{intro}\n"
        "Official CUDA package-manager guidance installs nvidia-gds after the "
        "NVIDIA driver and CUDA Toolkit are fully installed.\n"
        f"CUDA Linux installation guide: {CUDA_LINUX_INSTALL_GUIDE_URL}\n"
        "Find exact package names in the configured CUDA repository:\n"
        "  Ubuntu/Debian: apt-cache search nvidia-gds\n"
        "                 apt-cache search nvidia-fs\n"
        "  RHEL/Rocky:    sudo dnf list --available 'nvidia-gds*' 'nvidia-fs*'\n"
        "Install all GDS packages:\n"
        "  Ubuntu/Debian: sudo apt-get install nvidia-gds\n"
        "  RHEL/Rocky:    sudo dnf install nvidia-gds\n"
        "Use a versioned package only when your CUDA repo exposes one and you "
        "must pin to the installed toolkit, for example nvidia-gds-13-2.\n"
        "If only the nvidia_fs kernel module package is missing:\n"
        "  Ubuntu/Debian: sudo apt-get install nvidia-fs-dkms\n"
        "  RHEL/Rocky:    sudo dnf install nvidia-fs-dkms\n"
        "After install: sudo modprobe nvidia_fs"
    )


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


def _module_sysfs_attr(module: str, attr: str) -> Optional[str]:
    path = os.path.join("/sys/module", module.replace("-", "_"), attr)
    try:
        with open(path) as fh:
            value = fh.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    return value or None


def _kernel_module_loaded(module: str) -> bool:
    return _lsmod_has(module) or os.path.isdir(
        os.path.join("/sys/module", module.replace("-", "_"))
    )


def _modinfo(module: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["modinfo", module], capture_output=True, text=True, timeout=5
        )
        info: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
        return info
    except Exception:
        return {}


def _kernel_module_state(module: str) -> dict[str, object]:
    """Return loaded/installed/version state for a kernel module.

    A loaded module can be absent from modinfo's search path when it was loaded
    from an out-of-tree or kernel-version-specific location. Prefer the active
    sysfs module version when the module is already resident.
    """
    loaded = _kernel_module_loaded(module)
    if loaded:
        version = _module_sysfs_attr(module, "version")
        if version:
            return {
                "found": True,
                "loaded": True,
                "version": version,
                "source": f"/sys/module/{module.replace('-', '_')}/version",
                "detail": "kernel module is loaded; version read from sysfs.",
            }

    info = _modinfo(module)
    version = info.get("version")
    if version:
        return {
            "found": True,
            "loaded": loaded,
            "version": version,
            "source": f"modinfo {module}",
            "detail": (
                "kernel module is loaded; version read from modinfo."
                if loaded else
                "kernel module package is installed."
            ),
        }
    if loaded:
        return {
            "found": True,
            "loaded": True,
            "version": None,
            "source": "lsmod/sysfs",
            "detail": "kernel module is loaded, but no version field was found.",
        }
    if info:
        return {
            "found": True,
            "loaded": False,
            "version": None,
            "source": f"modinfo {module}",
            "detail": "kernel module package is installed, but no version field was found.",
        }
    return {
        "found": False,
        "loaded": False,
        "version": None,
        "source": None,
        "detail": "kernel module package not detected.",
    }


def _nvidia_driver_version() -> Optional[tuple[int, int]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        m = re.match(r"(\d+)\.(\d+)", line)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def _nvidia_driver_version_text() -> tuple[Optional[str], Optional[str]]:
    """Return a displayable driver version and the source used to find it."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                version = line.strip()
                if version:
                    return version, "nvidia-smi"
    except Exception:
        pass

    info = _modinfo("nvidia")
    version = info.get("version")
    if version:
        return version, "modinfo nvidia"
    return None, None


def _cuda_version_from_dir(path: str) -> Optional[str]:
    candidates = [
        os.path.join(path, "version.json"),
        os.path.join(path, "version.txt"),
    ]
    for candidate in candidates:
        try:
            with open(candidate) as fh:
                text = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if candidate.endswith(".json"):
            try:
                import json as _json

                payload = _json.loads(text)
                cuda_entry = payload.get("cuda") if isinstance(payload, dict) else None
                if isinstance(cuda_entry, dict) and cuda_entry.get("version"):
                    return str(cuda_entry["version"])
                if isinstance(cuda_entry, str):
                    return cuda_entry
                if isinstance(payload, dict) and payload.get("version"):
                    return str(payload["version"])
            except Exception:
                pass
        match = re.search(r'"cuda"\s*:\s*"([^"]+)"', text)
        if match:
            return match.group(1)
        match = re.search(r"CUDA Version\s+([0-9.]+)", text)
        if match:
            return match.group(1)
    return None


def _cuda_toolkit_info() -> dict:
    """Find CUDA Toolkit version/path details for checks and display."""
    import glob as _glob
    import shutil

    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = sorted(_glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)
        if candidates:
            nvcc = candidates[0]

    if nvcc:
        version = "unknown"
        detail = ""
        try:
            result = subprocess.run(
                [nvcc, "--version"], capture_output=True, text=True, timeout=5
            )
            detail = result.stdout.strip()
            match = re.search(r"release\s+(\S+),", result.stdout)
            if match:
                version = match.group(1).rstrip(",")
        except Exception as exc:
            detail = str(exc)
        return {
            "found": True,
            "complete": True,
            "version": version,
            "nvcc": nvcc,
            "cuda_dir": os.path.dirname(os.path.dirname(nvcc)),
            "source": f"nvcc: {nvcc}",
            "detail": detail,
        }

    cuda_dirs = sorted(_glob.glob("/usr/local/cuda*"), reverse=True)
    if cuda_dirs:
        return {
            "found": True,
            "complete": False,
            "version": _cuda_version_from_dir(cuda_dirs[0]) or "unknown",
            "nvcc": None,
            "cuda_dir": cuda_dirs[0],
            "source": cuda_dirs[0],
            "detail": "CUDA directory found, but nvcc was not found.",
        }

    return {
        "found": False,
        "complete": False,
        "version": None,
        "nvcc": None,
        "cuda_dir": None,
        "source": None,
        "detail": "CUDA Toolkit not found.",
    }


def collect_version_context(
    *,
    include_runtime_driver: bool = True,
    gdscheck_output: Optional[str] = None,
) -> list[dict[str, object]]:
    """Collect display-friendly kernel, CUDA, driver, nvidia-fs, and libcufile versions."""
    import platform as _platform

    rows: list[dict[str, object]] = []

    kernel_release = _platform.release() or "unknown"
    rows.append({
        "component": "Linux kernel",
        "found": kernel_release != "unknown",
        "version": kernel_release,
        "source": "uname -r",
        "detail": f"platform.release() = {kernel_release}",
    })

    cuda = _cuda_toolkit_info()
    rows.append({
        "component": "CUDA Toolkit",
        "found": bool(cuda.get("found")),
        "version": cuda.get("version") or "not found",
        "source": cuda.get("source"),
        "detail": cuda.get("detail"),
    })

    if include_runtime_driver:
        driver_version, driver_source = _nvidia_driver_version_text()
        driver_detail = (
            "Runtime driver version from nvidia-smi when available; modinfo fallback otherwise."
            if driver_version else
            "NVIDIA driver not found via nvidia-smi or modinfo nvidia."
        )
    else:
        info = _modinfo("nvidia")
        driver_version = info.get("version")
        driver_source = "modinfo nvidia" if driver_version else None
        driver_detail = (
            "pre-install intentionally avoids nvidia-smi; runtime driver checks run in post-install."
            if driver_version else
            "NVIDIA kernel driver not found by modinfo. pre-install intentionally avoids nvidia-smi."
        )

    rows.append({
        "component": "NVIDIA driver",
        "found": bool(driver_version),
        "version": driver_version or "not found",
        "source": driver_source,
        "detail": driver_detail,
    })

    nvidia_fs_state = _kernel_module_state("nvidia_fs")
    nvidia_fs_version = nvidia_fs_state.get("version")
    nvidia_fs_display_version = (
        nvidia_fs_version
        or ("loaded, version unknown" if nvidia_fs_state.get("loaded") else None)
        or ("installed, version unknown" if nvidia_fs_state.get("found") else None)
        or "not found"
    )
    rows.append({
        "component": "nvidia-fs",
        "found": bool(nvidia_fs_state.get("found")),
        "version": nvidia_fs_display_version,
        "source": nvidia_fs_state.get("source"),
        "detail": nvidia_fs_state.get("detail"),
    })

    libcufile = cufile_version.detect_libcufile(gdscheck_output)
    if libcufile.get("found"):
        version = (
            libcufile.get("file_version")
            or libcufile.get("gds_release_version")
            or libcufile.get("api_version")
            or "found, version unknown"
        )
        rows.append({
            "component": "libcufile",
            "found": True,
            "version": version,
            "source": libcufile.get("path"),
            "detail": f"realpath={libcufile.get('realpath')}",
        })
    else:
        rows.append({
            "component": "libcufile",
            "found": False,
            "version": "not found",
            "source": None,
            "detail": "GDS user-space library not detected.",
        })

    return rows


def _gpu_compute_caps() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        return [
            line.strip()
            for line in result.stdout.splitlines()
            if re.match(r"^\d+\.\d+$", line.strip())
        ]
    except Exception:
        return []


def _kmsg_nvidia_fs() -> kmsg.KernelLogResult:
    return kmsg.grep_kmsg_result(r"nvidia[_\s-]fs|nvidia_peermem")


# ---------------------------------------------------------------------------
# Module presence and version
# ---------------------------------------------------------------------------

def check_driver_installed() -> CheckResult:
    """Check the root driver prerequisite before version/type dependent probes."""
    ver = _nvidia_driver_version()
    info = _modinfo("nvidia")

    if ver or info:
        evidence = []
        if ver:
            evidence.append(f"nvidia-smi driver_version → {ver[0]}.{ver[1]}")
        if info:
            evidence.append(
                "modinfo nvidia: "
                f"version={info.get('version', 'unknown')}, "
                f"license={info.get('license', 'unknown')}"
            )
        return CheckResult(
            check="NVIDIA driver installed",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why="NVIDIA driver is installed; version and Open Driver checks can run.",
            evidence="\n".join(evidence) if evidence else None,
        )

    return CheckResult(
        check="NVIDIA driver installed",
        mode=GDSMode.NATIVE,
        status=Status.FAIL,
        why=(
            "NVIDIA driver is not installed or not loaded. "
            "nvidia-smi returned no driver version and modinfo nvidia returned nothing."
        ),
        mitigation=_driver_install_mitigation(),
    )


def check_open_driver_preinstall() -> CheckResult:
    """
    Pre-install NVIDIA driver check that does not use nvidia-smi.

    pre-install must work before the runtime driver stack is healthy, so it
    uses only modinfo and leaves nvidia-smi version/compute validation to
    post-install.
    """
    info = _modinfo("nvidia")
    if not info:
        return CheckResult(
            check="NVIDIA Open Driver install",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                "NVIDIA kernel driver is not installed. Install the NVIDIA driver "
                "before installing or validating GDS."
            ),
            mitigation=_driver_install_mitigation(),
        )

    version = info.get("version", "unknown")
    license_ = info.get("license", "")
    license_lower = license_.lower()

    if "mit" in license_lower or "gpl" in license_lower:
        return CheckResult(
            check="NVIDIA Open Driver install",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=(
                f"NVIDIA Open Kernel Driver {version} is installed. "
                "post-install will verify runtime driver version and GPU compute capability."
            ),
            evidence=f"modinfo nvidia: version={version}, license={license_}",
        )

    if "nvidia" in license_lower:
        return CheckResult(
            check="NVIDIA Open Driver install",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                f"Proprietary NVIDIA driver {version} is installed. GDS direct "
                "modes require the NVIDIA Open Kernel Driver."
            ),
            mitigation=_driver_install_mitigation(),
            evidence=f"modinfo nvidia: version={version}, license={license_}",
        )

    return CheckResult(
        check="NVIDIA Open Driver install",
        mode=GDSMode.NATIVE,
        status=Status.WARN,
        why=(
            f"NVIDIA driver {version} is installed but the license is unrecognised "
            f"({license_}). GDS requires the NVIDIA Open Kernel Driver."
        ),
        mitigation=_driver_install_mitigation(),
        evidence=f"modinfo nvidia: version={version}, license={license_}",
    )


def check_nvidia_fs_loaded() -> CheckResult:
    module_state = _kernel_module_state("nvidia_fs")
    if module_state.get("loaded"):
        ver = module_state.get("version") or "unknown"
        source = module_state.get("source") or "lsmod/sysfs"
        return CheckResult(
            check="nvidia-fs module", mode=GDSMode.NATIVE, status=Status.PASS,
            why=f"nvidia_fs module is loaded (version {ver}).",
            evidence=f"nvidia_fs loaded; version={ver}; source={source}",
        )

    # Not loaded — check if the .ko exists but just isn't inserted
    if module_state.get("found"):
        ver = module_state.get("version") or "unknown"
        return CheckResult(
            check="nvidia-fs module", mode=GDSMode.NATIVE, status=Status.WARN,
            why=(
                f"nvidia_fs module (version {ver}) is installed but NOT loaded. "
                "The nvidia-fs/nvfs direct route requires this module to be active, "
                "but this is not a global GDS blocker. Direct P2P "
                "(upstream PCI P2PDMA on x86 or C2C on supported GH/GB ARM platforms), "
                "userspace RDMA, or compat may still be valid depending on the "
                "filesystem, block device, topology, cufile.json, and gdscheck "
                "active routes."
            ),
            mitigation=(
                "sudo modprobe nvidia_fs\n"
                "For persistence across reboots:\n"
                "  echo 'nvidia_fs' | sudo tee /etc/modules-load.d/nvidia_fs.conf\n"
                "If modprobe fails, check the kernel log for load errors:\n"
                "  sudo journalctl -k -b | grep nvidia_fs\n"
                "  sudo dmesg | grep nvidia_fs\n"
                "If you intend to use direct P2P instead, review the "
                "post-install P2PDMA/C2C Direct Routes section."
            ),
            evidence=f"{module_state.get('source') or 'module metadata'} found but module is not loaded",
        )

    return CheckResult(
        check="nvidia-fs module", mode=GDSMode.NATIVE, status=Status.WARN,
        why=(
            "nvidia_fs module is not installed. "
            "The nvfs kernel route requires the nvidia-fs kernel module package: "
            "normally installed through nvidia-gds, with nvidia-fs-dkms providing "
            "the module. This is not a global GDS blocker: "
            "direct P2P (upstream PCI P2PDMA on x86 or C2C on supported GH/GB ARM platforms), "
            "userspace RDMA, or compat may still be valid "
            "depending on the filesystem, block device, topology, cufile.json, "
            "and gdscheck active routes."
        ),
        mitigation=(
            _nvidia_fs_package_mitigation(
                "Install the matching GDS package for this CUDA/driver stack."
            )
            + "\n"
            "Verify packages:\n"
            "  rpm -qa | grep -E 'nvidia-gds|nvidia-fs|gds'\n"
            "  dpkg -l | grep -E 'nvidia-gds|nvidia-fs|gds'\n"
            "For NVMe/NVMe-oF nvfs mode, also ensure the NVMe stack has the "
            "GDS patches supplied by MLNX_OFED/DOCA when you are not using a "
            "confirmed P2PDMA/C2C direct route.\n"
        ),
    )


def check_libcufile_version(gdscheck_output: Optional[str] = None) -> CheckResult:
    info = cufile_version.detect_libcufile(gdscheck_output)
    if not info.get("found"):
        return CheckResult(
            check="libcufile library",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why="libcufile was not found. GDS user-space APIs require the libcufile package.",
            mitigation=(
                "Install the matching libcufile package for the installed CUDA Toolkit.\n"
                "Verify packages:\n"
                "  rpm -qa | grep libcufile\n"
                "  dpkg -l | grep libcufile\n"
                "Then verify library paths:\n"
                "  ls /usr/local/cuda*/lib64/libcufile.so*"
            ),
            evidence="\n".join(info.get("probe_errors") or []) or None,
        )

    pieces = []
    if info.get("api_version"):
        pieces.append(f"cuFileGetVersion API={info['api_version']}")
    elif info.get("probe_errors"):
        # Library found and loaded, but this older GDS release doesn't
        # export cuFileGetVersion (e.g. GDS 1.7.x, shipped with CUDA 12.2).
        pieces.append("cuFileGetVersion unavailable")
    if info.get("file_version"):
        pieces.append(f"file={info['file_version']}")
    if info.get("gds_release_version"):
        pieces.append(f"gdscheck release={info['gds_release_version']}")
    evidence_lines = [
        f"path={info.get('path')}",
        f"realpath={info.get('realpath')}",
        f"cuFileGetVersion={info.get('api_version_int')}",
    ]
    if not info.get("api_version") and info.get("probe_errors"):
        evidence_lines.append(f"probe_error={info['probe_errors'][0]}")
    return CheckResult(
        check="libcufile library",
        mode=GDSMode.NATIVE,
        status=Status.PASS,
        why=(
            "libcufile detected"
            + (f" ({', '.join(pieces)})" if pieces else ".")
        ),
        evidence="\n".join(evidence_lines),
    )


def check_driver_version() -> CheckResult:
    ver = _nvidia_driver_version()
    if ver is None:
        info = _modinfo("nvidia")
        if info:
            version = info.get("version", "unknown")
            return CheckResult(
                check="NVIDIA driver version",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why=(
                    f"NVIDIA driver module is installed (version {version}), but "
                    "nvidia-smi did not return a runtime driver version. This "
                    "usually means no NVIDIA GPU is visible to the runtime or the "
                    "runtime driver stack is not usable."
                ),
                mitigation=(
                    "For GPU-memory GDS validation, expose/install an NVIDIA GPU "
                    "and verify: nvidia-smi -L\n"
                    "If this is a no-GPU host, libcufile can only be validated for "
                    "system-memory buffers."
                ),
                evidence=(
                    "modinfo nvidia: "
                    f"version={version}, license={info.get('license', 'unknown')}"
                ),
            )
        return CheckResult(
            check="NVIDIA driver version", mode=GDSMode.NATIVE, status=Status.FAIL,
            why=(
                "nvidia-smi not found or returned no driver version, and the "
                "NVIDIA kernel module was not found. Cannot verify GDS driver "
                "compatibility."
            ),
            mitigation=_driver_install_mitigation(),
        )

    major, minor = ver
    ver_str = f"{major}.{minor}"

    if (major, minor) >= MIN_DRIVER_VERSION:
        return CheckResult(
            check="NVIDIA driver version", mode=GDSMode.NATIVE, status=Status.PASS,
            why=f"Driver {ver_str} >= minimum {MIN_DRIVER_VERSION[0]} for GDS support.",
            evidence=f"nvidia-smi driver_version → {ver_str}",
        )

    return CheckResult(
        check="NVIDIA driver version", mode=GDSMode.NATIVE, status=Status.FAIL,
        why=(
            f"NVIDIA driver {ver_str} is below the minimum {MIN_DRIVER_VERSION[0]}.x "
            "required for GDS support."
        ),
        mitigation=_driver_install_mitigation(),
        evidence=f"nvidia-smi driver_version → {ver_str}",
    )


def check_gpu_compute_capability() -> CheckResult:
    caps = _gpu_compute_caps()
    if not caps:
        return CheckResult(
            check="GPU compute capability", mode=GDSMode.NATIVE, status=Status.FAIL,
            why="Cannot determine GPU compute capability — nvidia-smi unavailable.",
            mitigation=_driver_install_mitigation(),
        )

    unsupported: list[str] = []
    for cap in caps:
        try:
            major, minor = cap.split(".")
            if int(major) < 6:
                unsupported.append(cap)
        except ValueError:
            pass

    if unsupported:
        return CheckResult(
            check="GPU compute capability", mode=GDSMode.NATIVE, status=Status.FAIL,
            why=(
                f"GPU(s) with compute capability {', '.join(unsupported)} found. "
                "GDS requires compute capability >= 6.0 (Volta V100 or newer). "
                "Pascal (6.x) is the minimum; Volta+ is recommended."
            ),
            mitigation=(
                "GDS is not available on pre-Pascal GPUs. "
                "Use a V100, T4, A100, H100, or any Volta/Turing/Ampere/Hopper GPU."
            ),
            evidence=f"nvidia-smi compute_cap → {', '.join(caps)}",
        )

    return CheckResult(
        check="GPU compute capability", mode=GDSMode.NATIVE, status=Status.PASS,
        why=f"GPU compute capability {', '.join(caps)} supports GDS (>= 6.0 required).",
        evidence=f"nvidia-smi compute_cap → {', '.join(caps)}",
    )


# ---------------------------------------------------------------------------
# dmesg analysis — surface nvidia-fs P2PDMA rejection reasons
# ---------------------------------------------------------------------------

# Maps dmesg patterns from nvidia-fs to human explanations
DMESG_PATTERNS: list[tuple[re.Pattern, Status, str, Optional[str]]] = [
    (
        re.compile(r"iommu.*not.*passthrough|iommu.*strict|p2pdma.*iommu", re.I),
        Status.FAIL,
        "nvidia-fs detected IOMMU not in passthrough mode — P2PDMA disabled.",
        "Set iommu=pt or intel_iommu=off / amd_iommu=off in kernel cmdline.",
    ),
    (
        re.compile(r"ACS.*enabled|acs.*req.?redir", re.I),
        Status.FAIL,
        "nvidia-fs detected ACS P2P Request Redirect enabled — P2PDMA blocked.",
        "Add pci=noacs to kernel cmdline or disable ACS in BIOS.",
    ),
    (
        re.compile(r"distance.*fail|different.*root|root complex", re.I),
        Status.WARN,
        (
            "nvidia-fs reports a higher-distance GPU↔NVMe topology. GDS can "
            "operate across root ports, but performance may be lower than a "
            "same-root-port or same-switch path."
        ),
        (
            "Prefer the closest GPU/NVMe pair from nvidia-smi topo -m -nvme, "
            "use NUMA binding near that path, and verify active routes with "
            "gdscheck or a workload."
        ),
    ),
    (
        re.compile(r"p2p.*not supported|p2pdma.*unavailable", re.I),
        Status.FAIL,
        "nvidia-fs reports P2PDMA unavailable for this GPU↔NVMe pair.",
        (
            "Check ACS, IOMMU, cufile.json P2PDMA settings, kernel PCI_P2PDMA "
            "support, and gdscheck active routes. Cross-root-port topology alone "
            "should be treated as a performance warning, not a blocker."
        ),
    ),
    (
        re.compile(r"CONFIG_PCI_P2PDMA.*not.*set|p2pdma.*compiled.*out", re.I),
        Status.FAIL,
        "nvidia-fs detected CONFIG_PCI_P2PDMA not compiled into this kernel.",
        "Use a kernel with CONFIG_PCI_P2PDMA=y.",
    ),
    (
        re.compile(r"using p2pdma|p2pdma.*enabled", re.I),
        Status.PASS,
        "nvidia-fs confirmed P2PDMA is active for at least one GPU↔NVMe pair.",
        None,
    ),
]


def check_kmsg_nvidia_fs() -> list[CheckResult]:
    """Scan the kernel log for nvidia-fs runtime messages.

    Uses the kmsg helper (journalctl preferred, dmesg fallback) to read
    kernel-log lines and matches them against DMESG_PATTERNS to surface
    runtime decisions and rejections from the nvidia-fs module.
    """
    log_result = _kmsg_nvidia_fs()
    lines = list(log_result.lines)
    if not log_result.available:
        detail = (
            "noninteractive sudo could not read journalctl or dmesg"
            if log_result.permission_denied
            else "journalctl and dmesg did not return readable kernel logs"
        )
        return [CheckResult(
            check="nvidia-fs kernel log messages", mode=GDSMode.P2PDMA, status=Status.WARN,
            why=(
                "Could not inspect nvidia-fs kernel log messages; "
                f"{detail}. This log-based rejection check was not performed."
            ),
            mitigation=kmsg.privileged_validation_mitigation(
                (
                    "sudo journalctl -k -b | grep nvidia_fs",
                    "sudo dmesg | grep nvidia_fs",
                )
            ),
            evidence=log_result.error,
        )]

    if not lines:
        source = log_result.source or "unknown"
        if "dmesg" in source:
            evidence_parts = [f"source={source}"]
            if log_result.error:
                evidence_parts.append(f"fallback_reason={log_result.error}")
            return [CheckResult(
                check="nvidia-fs kernel log messages", mode=GDSMode.P2PDMA, status=Status.WARN,
                why=(
                    "Only the dmesg kernel ring buffer was readable, and it "
                    "contained no nvidia-fs messages. No log-based rejection "
                    "evidence is available, but dmesg can wrap during long "
                    "boots, so older nvidia-fs messages may have been lost."
                ),
                mitigation=(
                    "Inspect the current boot journal with privileges:\n"
                    "  sudo journalctl -k -b | grep -iE 'nvidia[_ -]fs|nvidia_peermem'\n"
                    "If journalctl is unavailable, reload nvidia_fs or reproduce "
                    "the workload, then immediately check:\n"
                    "  sudo dmesg | grep -iE 'nvidia[_ -]fs|nvidia_peermem'"
                ),
                evidence="\n".join(evidence_parts),
            )]
        return [CheckResult(
            check="nvidia-fs kernel log messages", mode=GDSMode.P2PDMA, status=Status.INFO,
            why=(
                "Kernel logs were readable, but no nvidia-fs messages were found. "
                "No log-based rejection evidence is available."
            ),
            evidence=f"source={log_result.source}",
        )]

    results: list[CheckResult] = []

    for pattern, status, why, mitigation in DMESG_PATTERNS:
        matched = [l for l in lines if pattern.search(l)]
        if matched:
            results.append(CheckResult(
                check="nvidia-fs kernel log", mode=GDSMode.P2PDMA, status=status,
                why=why,
                mitigation=mitigation,
                evidence="\n".join(matched[:3]),
            ))

    if not results:
        results.append(CheckResult(
            check="nvidia-fs kernel log messages", mode=GDSMode.P2PDMA, status=Status.PASS,
            why="nvidia-fs kernel-log output shows no P2PDMA rejection messages.",
            evidence="\n".join(lines[:5]),
        ))

    return results


# ---------------------------------------------------------------------------
# gdscheck
# ---------------------------------------------------------------------------

def check_gdscheck() -> list[CheckResult]:
    """
    Verify gdscheck binary is available. Mode-line parsing is handled by
    checks.gds_report._gdscheck_parse(), which understands the DRIVER
    CONFIGURATION format (e.g. "NVMe : nvfs, compat"). This check just
    confirms the tool is present.
    """
    import glob as glob_mod

    # gdscheck doesn't have a --version flag; probe by file existence instead.
    candidates = glob_mod.glob("/usr/local/cuda*/gds/tools/gdscheck")
    candidates += ["/usr/local/cuda/gds/tools/gdscheck"]

    for c in candidates:
        if os.path.isfile(c):
            return [CheckResult(
                check="gdscheck", mode=GDSMode.NATIVE, status=Status.PASS,
                why=f"gdscheck binary found at {c}.",
                evidence=c,
            )]

    return [CheckResult(
        check="gdscheck", mode=GDSMode.NATIVE, status=Status.WARN,
        why=(
            "gdscheck tool not found at /usr/local/cuda/gds/tools/gdscheck. "
            "It is installed by the matching GDS tools package."
        ),
        mitigation=(
            "Verify CUDA Toolkit is installed, then install gds-tools for the same CUDA version.\n"
            "Verify the path exists:\n"
            "  ls /usr/local/cuda/gds/tools/gdscheck\n"
            "Run manually:\n"
            "  /usr/local/cuda/gds/tools/gdscheck -p"
        ),
    )]


def check_open_driver(gdscheck_output: str, mode: GDSMode = GDSMode.NATIVE) -> CheckResult:
    """
    Verify the NVIDIA Open Kernel Driver is installed.

    GDS (both Native nvfs and P2PDMA modes) requires the open-source NVIDIA
    kernel driver. The proprietary closed-source driver does not support the
    peer-memory mapping that nvidia-fs and P2PDMA rely on.

    Parsed from gdscheck -p PLATFORM INFO:
      Nvidia Driver Info Status: Supported(Nvidia Open Driver Installed)
    """
    for line in gdscheck_output.splitlines():
        if "Nvidia Driver Info Status" in line:
            if "Open Driver" in line and "Supported" in line:
                return CheckResult(
                    check="NVIDIA Open Driver",
                    mode=mode,
                    status=Status.PASS,
                    why=(
                        "NVIDIA Open Kernel Driver is installed — required for "
                        "GDS Native (nvfs) and P2PDMA modes."
                    ),
                    evidence=line.strip(),
                )
            else:
                # Either "Not Supported" or shows proprietary driver
                return CheckResult(
                    check="NVIDIA Open Driver",
                    mode=mode,
                    status=Status.FAIL,
                    why=(
                        "NVIDIA Open Kernel Driver is NOT installed. "
                        "GDS (nvfs and P2PDMA) requires the open-source NVIDIA kernel driver — "
                        "the proprietary driver does not support GDS peer memory mapping."
                    ),
                    mitigation=(
                        "Install the NVIDIA Open Kernel Driver:\n"
                        "  Ubuntu: sudo apt-get install nvidia-open\n"
                        "  RHEL:   sudo dnf module install nvidia-driver:open-dkms\n"
                        "After install, reboot and verify:\n"
                        "  gdscheck -p | grep 'Nvidia Driver Info Status'"
                    ),
                    evidence=line.strip(),
                )

    # Line absent — older gdscheck version or output incomplete
    return CheckResult(
        check="NVIDIA Open Driver",
        mode=mode,
        status=Status.WARN,
        why=(
            "Could not find 'Nvidia Driver Info Status' in gdscheck output. "
            "GDS requires the NVIDIA Open Kernel Driver. "
            "This line is absent — may indicate an older gdscheck version."
        ),
        mitigation=(
            "Verify manually:\n"
            "  gdscheck -p | grep 'Nvidia Driver Info Status'\n"
            "Expected: Supported(Nvidia Open Driver Installed)"
        ),
    )


def check_gpu_presence_lspci() -> CheckResult:
    """
    Detect NVIDIA GPUs via lspci — works without the NVIDIA driver installed.
    Looks for 3D controller, VGA controller, Display controller, and
    Processing accelerators bearing the NVIDIA name (covers datacenter GPUs
    like A100/H100 that show up as '3D controller' or 'Processing accelerators').
    """
    try:
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return CheckResult(
            check="NVIDIA GPU (lspci)",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why="lspci not available — cannot scan for NVIDIA GPUs.",
            mitigation="Install pciutils: apt-get install pciutils  (or yum install pciutils)",
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return CheckResult(
            check="NVIDIA GPU (lspci)",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why="lspci failed, so GPU presence could not be verified.",
            mitigation="Run `lspci` directly and fix pciutils/PCI visibility in this environment.",
            evidence=detail or f"lspci exit code {result.returncode}",
        )

    gpu_classes = {"VGA", "3D", "Display", "Processing"}
    gpu_lines = [
        line for line in result.stdout.splitlines()
        if "NVIDIA" in line and any(cls in line for cls in gpu_classes)
    ]

    if gpu_lines:
        return CheckResult(
            check="NVIDIA GPU (lspci)",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"Found {len(gpu_lines)} NVIDIA GPU(s) via lspci.",
            evidence="\n".join(gpu_lines),
        )

    # No GPU-class device — check for any NVIDIA device at all (could be NIC etc.)
    any_nvidia = [l for l in result.stdout.splitlines() if "NVIDIA" in l]
    if any_nvidia:
        return CheckResult(
            check="NVIDIA GPU (lspci)",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why=(
                "NVIDIA device found on PCIe but not identified as a GPU class "
                "(VGA/3D/Display/Processing). GDS requires a NVIDIA GPU."
            ),
            evidence="\n".join(any_nvidia[:5]),
        )

    return CheckResult(
        check="NVIDIA GPU (lspci)",
        mode=GDSMode.NATIVE,
        status=Status.FAIL,
        why="No NVIDIA GPU found via lspci. GDS requires a NVIDIA GPU (Pascal/Volta/Ampere/Hopper or newer).",
        mitigation="Verify the GPU is seated and recognised by the system: lspci | grep -i nvidia",
    )


def coherent_gpu_memory_mode_value() -> Optional[str]:
    """Read CoherentGPUMemoryMode from /proc/driver/nvidia/params.

    Distinguishes three states:
      - None       file unreadable or param line missing — driver not loaded
                   (or this driver build does not expose the parameter at all)
      - ""         param present but empty — driver is loaded but no explicit
                   CDMM mode is set, so the driver uses its default behavior
                   (on Grace, default behavior is typically NUMA mode)
      - "driver"   CDMM (Coherent Device Memory Management) mode explicitly active
      - "numa"     GPU memory explicitly exposed as a NUMA node
      - other      unexpected token

    Requires the NVIDIA kernel driver to be loaded for any non-None result.
    """
    params_path = "/proc/driver/nvidia/params"
    try:
        with open(params_path) as fh:
            content = fh.read()
    except (FileNotFoundError, PermissionError):
        return None
    for line in content.splitlines():
        if line.startswith("CoherentGPUMemoryMode:"):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def check_coherent_gpu_memory_mode() -> Optional[CheckResult]:
    """
    Read CoherentGPUMemoryMode from /proc/driver/nvidia/params.
      "driver" → CDMM (Coherent Device Memory Management) mode
      "numa"   → NUMA mode (GPU memory exposed as a NUMA node)
      ""       → not applicable (no coherent memory support — skip)

    Only relevant on systems with coherent memory support (GH200, GB200).
    Returns None if the system does not have coherent memory (empty value),
    so the check is silently skipped on standard x86 / non-GH/GB systems.
    """
    params_path = "/proc/driver/nvidia/params"
    try:
        with open(params_path) as fh:
            content = fh.read()
    except (FileNotFoundError, PermissionError):
        return None  # Driver not loaded or no access — not applicable

    for line in content.splitlines():
        if line.startswith("CoherentGPUMemoryMode:"):
            value = line.split(":", 1)[1].strip().strip('"')
            if value == "":
                return None  # Not a coherent memory system — skip silently
            if value == "driver":
                return CheckResult(
                    check="CoherentGPUMemoryMode",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="CoherentGPUMemoryMode=driver — CDMM (Coherent Device Memory Management) mode active.",
                    evidence=line.strip(),
                )
            if value == "numa":
                return CheckResult(
                    check="CoherentGPUMemoryMode",
                    mode=GDSMode.NATIVE,
                    status=Status.PASS,
                    why="CoherentGPUMemoryMode=numa — GPU memory is exposed as a NUMA node.",
                    evidence=line.strip(),
                )
            return CheckResult(
                check="CoherentGPUMemoryMode",
                mode=GDSMode.NATIVE,
                status=Status.WARN,
                why=f"CoherentGPUMemoryMode has unexpected value: '{value}'.",
                evidence=line.strip(),
            )

    return None  # Param not present — not applicable


def _parse_registry_dwords(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in value.strip().strip('"').split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        parsed[key.strip()] = raw_value.strip()
    return parsed


def check_p2pdma_driver_registries() -> Optional[CheckResult]:
    """
    Check NVIDIA driver registry dwords required for PCI P2PDMA on x86.

    NVIDIA documents these as nvidia module options in /etc/modprobe.d/ and
    verifies the loaded state through /proc/driver/nvidia/params. cufile.json
    can request P2PDMA, but the route cannot activate if these loaded driver
    params are missing on x86.
    """
    arch = platform.machine().lower()
    if arch not in {"x86_64", "amd64"}:
        return None

    params_path = "/proc/driver/nvidia/params"
    try:
        with open(params_path) as fh:
            content = fh.read()
    except (FileNotFoundError, PermissionError):
        return CheckResult(
            check="NVIDIA P2PDMA driver registries",
            mode=GDSMode.P2PDMA,
            status=Status.WARN,
            why=(
                "Cannot read /proc/driver/nvidia/params, so the loaded NVIDIA "
                "driver registry dwords for PCI P2PDMA could not be verified."
            ),
            mitigation=(
                "After the NVIDIA driver is loaded, verify:\n"
                "  cat /proc/driver/nvidia/params | grep -i static\n"
                "For PCI P2PDMA on x86, configure /etc/modprobe.d/ with:\n"
                '  options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1;"\n'
                "For pre-Hopper GPUs such as L4, L40, A100, or A40, also include ForceP2P=0.\n"
                f"Then rebuild initramfs and reboot:\n{REBUILD_INITRAMFS_CMDS}"
            ),
        )

    registry_line = None
    for line in content.splitlines():
        if line.startswith("RegistryDwords:"):
            registry_line = line.strip()
            break

    if registry_line is None:
        return CheckResult(
            check="NVIDIA P2PDMA driver registries",
            mode=GDSMode.P2PDMA,
            status=Status.FAIL,
            why=(
                "The loaded NVIDIA driver params do not expose RegistryDwords. "
                "PCI P2PDMA requires NVIDIA driver registry dwords on x86, even "
                "when cufile.json requests P2PDMA."
            ),
            mitigation=(
                "Configure the NVIDIA module options and reboot:\n"
                '  echo \'options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1;"\' '
                "| sudo tee /etc/modprobe.d/nvidia-p2pdma.conf\n"
                "For pre-Hopper GPUs such as L4, L40, A100, or A40, include ForceP2P=0 in the same string.\n"
                f"Rebuild initramfs and reboot:\n{REBUILD_INITRAMFS_CMDS}\n"
                "Then verify:\n"
                "  cat /proc/driver/nvidia/params | grep -i static"
            ),
            evidence=params_path,
        )

    value = registry_line.split(":", 1)[1].strip()
    registry = _parse_registry_dwords(value)
    required = {
        "RMForceStaticBar1": "1",
        "RmForceDisableIomapWC": "1",
    }
    missing = [
        f"{key}={expected}"
        for key, expected in required.items()
        if registry.get(key) != expected
    ]

    if missing:
        return CheckResult(
            check="NVIDIA P2PDMA driver registries",
            mode=GDSMode.P2PDMA,
            status=Status.FAIL,
            why=(
                "The loaded NVIDIA driver is missing required x86 PCI P2PDMA "
                f"registry dword(s): {', '.join(missing)}. cufile.json can request "
                "P2PDMA, but the route cannot activate until these driver params "
                "are loaded."
            ),
            mitigation=(
                "Configure /etc/modprobe.d/nvidia-p2pdma.conf with:\n"
                '  options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;RmForceDisableIomapWC=1;"\n'
                "For pre-Hopper GPUs such as L4, L40, A100, or A40, use:\n"
                '  options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1;ForceP2P=0;RmForceDisableIomapWC=1;"\n'
                f"Then rebuild initramfs and reboot:\n{REBUILD_INITRAMFS_CMDS}\n"
                "Then verify:\n"
                "  cat /proc/driver/nvidia/params | grep -i static"
            ),
            evidence=registry_line,
        )

    return CheckResult(
        check="NVIDIA P2PDMA driver registries",
        mode=GDSMode.P2PDMA,
        status=Status.PASS,
        why=(
            "Loaded NVIDIA driver RegistryDwords include RMForceStaticBar1=1 "
            "and RmForceDisableIomapWC=1, which are required for PCI P2PDMA on x86."
        ),
        evidence=registry_line,
    )


def check_driver_type() -> CheckResult:
    """
    Detect whether the installed NVIDIA kernel driver is open-source or proprietary.
    Uses `modinfo nvidia` license field:
      - "NVIDIA"       → proprietary closed-source driver
      - "MIT" / "GPL"  → NVIDIA Open Kernel Driver (open-source)
    GDS direct modes require the NVIDIA Open Kernel Driver.
    """
    info = _modinfo("nvidia")
    if not info:
        return CheckResult(
            check="NVIDIA driver type",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                "NVIDIA kernel driver not installed (modinfo nvidia returned nothing). "
                "Install the NVIDIA driver before installing GDS."
            ),
            mitigation=_driver_install_mitigation(),
        )

    version = info.get("version", "unknown")
    license_ = info.get("license", "")
    license_lower = license_.lower()

    if "mit" in license_lower or "gpl" in license_lower:
        return CheckResult(
            check="NVIDIA driver type",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"NVIDIA Open Kernel Driver {version} installed — required for GDS (nvfs and P2PDMA).",
            evidence=f"modinfo nvidia: version={version}, license={license_}",
        )

    if "nvidia" in license_lower:
        return CheckResult(
            check="NVIDIA driver type",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                f"Proprietary NVIDIA driver {version} is installed. "
                "GDS (nvfs and P2PDMA) requires the NVIDIA Open Kernel Driver — "
                "the proprietary driver does not support GDS peer memory mapping."
            ),
            mitigation=_driver_install_mitigation(),
            evidence=f"modinfo nvidia: version={version}, license={license_}",
        )

    return CheckResult(
        check="NVIDIA driver type",
        mode=GDSMode.NATIVE,
        status=Status.WARN,
        why=f"NVIDIA driver {version} detected but type is unrecognised (license: {license_}). GDS requires the Open Kernel Driver.",
        evidence=f"modinfo nvidia: version={version}, license={license_}",
    )


def check_cuda_toolkit() -> CheckResult:
    """
    Check whether the CUDA Toolkit is installed (needed before installing the
    matching GDS packages such as gds-tools, libcufile, and nvidia-fs).
    Looks for nvcc in PATH and common CUDA installation paths.
    """
    info = _cuda_toolkit_info()
    nvcc = info.get("nvcc")
    if nvcc:
        version = info.get("version") or "unknown"
        return CheckResult(
            check="CUDA Toolkit",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"CUDA Toolkit {version} found at {nvcc}.",
            evidence=f"nvcc path: {nvcc}",
        )

    # No nvcc — also check for CUDA directories without nvcc in PATH.
    cuda_dir = info.get("cuda_dir")
    if cuda_dir:
        return CheckResult(
            check="CUDA Toolkit",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                f"CUDA directory found at {cuda_dir} but nvcc is not in PATH. "
                "The toolkit may be partially installed or PATH not configured."
            ),
            mitigation=(
                "Install or repair the CUDA Toolkit using the CUDA Downloads guide:\n"
                f"{CUDA_DOWNLOADS_URL}"
            ),
            evidence=f"Found {cuda_dir}, nvcc not in PATH",
        )

    return CheckResult(
        check="CUDA Toolkit",
        mode=GDSMode.NATIVE,
        status=Status.FAIL,
        why=(
            "CUDA Toolkit not found. Install CUDA Toolkit first, then install "
            "the matching GDS tools/packages."
        ),
        mitigation=(
            "Install the CUDA Toolkit using the CUDA Downloads guide:\n"
            f"{CUDA_DOWNLOADS_URL}"
        ),
    )


def check_nvidia_fs_preinstall() -> CheckResult:
    """
    Check nvidia-fs status for pre-install context.
    Not installed is the expected state at this stage (nvidia-fs ships with the
    GDS packages).
    Loaded = GDS already installed.
    Installed but not loaded = needs modprobe.
    Not installed = expected; will be resolved by installing matching GDS packages.
    """
    module_state = _kernel_module_state("nvidia_fs")
    if module_state.get("loaded"):
        ver = module_state.get("version") or "unknown"
        source = module_state.get("source") or "lsmod/sysfs"
        return CheckResult(
            check="nvidia-fs module",
            mode=GDSMode.NATIVE,
            status=Status.PASS,
            why=f"nvidia_fs loaded (version {ver}) — GDS kernel module is already present.",
            evidence=f"nvidia_fs loaded; version={ver}; source={source}",
        )

    if module_state.get("found"):
        ver = module_state.get("version") or "unknown"
        return CheckResult(
            check="nvidia-fs module",
            mode=GDSMode.NATIVE,
            status=Status.WARN,
            why=(
                f"nvidia_fs (version {ver}) is installed but not loaded. "
                "Load it before using GDS."
            ),
            mitigation=(
                "sudo modprobe nvidia_fs\n"
                "For persistence: echo 'nvidia_fs' | sudo tee /etc/modules-load.d/nvidia_fs.conf"
            ),
            evidence=f"{module_state.get('source') or 'module metadata'} found; module is not loaded",
        )

    return CheckResult(
        check="nvidia-fs module",
        mode=GDSMode.NATIVE,
        status=Status.WARN,
        why=(
            "nvidia-fs not installed — expected at pre-install stage. "
            "It will be installed as part of the NVIDIA GDS packages."
        ),
        mitigation=(
            _nvidia_fs_package_mitigation(
                "After installing CUDA Toolkit, install the matching GDS package."
            )
        ),
    )


def run_all() -> list[CheckResult]:
    results = []
    module_check = check_nvidia_fs_loaded()
    results.append(module_check)
    results.append(check_driver_version())
    results.append(check_gpu_compute_capability())
    # Only analyse the kernel log if the module is actually loaded — otherwise
    # the "no messages found" warning is redundant noise on top of the load
    # failure.
    if module_check.status == Status.PASS:
        results.extend(check_kmsg_nvidia_fs())
    results.extend(check_gdscheck())
    return results
