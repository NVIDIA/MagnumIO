# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import io
import shutil
import unittest
from unittest.mock import patch

from checks import cufile_version
from checks import kmsg
from checks import nvidia_fs
from checks.result import Status


class NvidiaFsPrerequisiteTests(unittest.TestCase):
    def _p2pdma_registry_result(self, params_content: str, arch: str = "x86_64"):
        with patch("checks.nvidia_fs.platform.machine", return_value=arch):
            with patch("builtins.open", return_value=io.StringIO(params_content)):
                return nvidia_fs.check_p2pdma_driver_registries()

    def test_p2pdma_driver_registries_pass_when_required_dwords_loaded(self):
        result = self._p2pdma_registry_result(
            'RegistryDwords: "RMForceStaticBar1=1;RmForceDisableIomapWC=1;"\n'
        )

        self.assertEqual(result.status, Status.PASS)
        self.assertIn("RMForceStaticBar1=1", result.why)
        self.assertIn("RmForceDisableIomapWC=1", result.why)

    def test_p2pdma_driver_registries_fail_when_registry_line_missing(self):
        result = self._p2pdma_registry_result("CoherentGPUMemoryMode: \n")

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("do not expose RegistryDwords", result.why)
        self.assertIn("NVreg_RegistryDwords", result.mitigation)
        self.assertIn("grep -i static", result.mitigation)

    def test_p2pdma_driver_registries_fail_when_required_dword_missing(self):
        result = self._p2pdma_registry_result(
            'RegistryDwords: "RMForceStaticBar1=1;"\n'
        )

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("RmForceDisableIomapWC=1", result.why)
        self.assertIn("ForceP2P=0", result.mitigation)

    def test_p2pdma_driver_registries_skip_non_x86(self):
        result = self._p2pdma_registry_result(
            'RegistryDwords: "RMForceStaticBar1=1;RmForceDisableIomapWC=1;"\n',
            arch="aarch64",
        )

        self.assertIsNone(result)

    def test_preinstall_open_driver_check_uses_modinfo_only(self):
        old_driver_version = nvidia_fs._nvidia_driver_version
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._nvidia_driver_version = lambda: self.fail("preinstall should not call nvidia-smi")
        nvidia_fs._modinfo = lambda module: {}
        try:
            result = nvidia_fs.check_open_driver_preinstall()
        finally:
            nvidia_fs._nvidia_driver_version = old_driver_version
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertIn("Open Kernel Module", result.mitigation)

    def test_missing_driver_installed_is_single_root_fail_with_cuda_download_link(self):
        old_driver_version = nvidia_fs._nvidia_driver_version
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._nvidia_driver_version = lambda: None
        nvidia_fs._modinfo = lambda module: {}
        try:
            result = nvidia_fs.check_driver_installed()
        finally:
            nvidia_fs._nvidia_driver_version = old_driver_version
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertIn("Open Kernel Module", result.mitigation)

    def test_missing_driver_version_is_fail_with_doc_link(self):
        old_driver_version = nvidia_fs._nvidia_driver_version
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._nvidia_driver_version = lambda: None
        nvidia_fs._modinfo = lambda module: {}
        try:
            result = nvidia_fs.check_driver_version()
        finally:
            nvidia_fs._nvidia_driver_version = old_driver_version
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertIn("Open Kernel Module", result.mitigation)
        self.assertNotIn("apt-get", result.mitigation)
        self.assertNotIn("dnf", result.mitigation)

    def test_driver_version_reports_installed_module_when_nvidia_smi_unusable(self):
        old_driver_version = nvidia_fs._nvidia_driver_version
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._nvidia_driver_version = lambda: None
        nvidia_fs._modinfo = lambda module: (
            {"version": "580.65.06", "license": "Dual MIT/GPL"}
            if module == "nvidia" else {}
        )
        try:
            result = nvidia_fs.check_driver_version()
        finally:
            nvidia_fs._nvidia_driver_version = old_driver_version
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("driver module is installed", result.why)
        self.assertIn("no NVIDIA GPU is visible", result.why)
        self.assertIn("nvidia-smi -L", result.mitigation)
        self.assertNotIn("CUDA Downloads", result.mitigation)

    def test_missing_driver_type_is_fail_with_open_kernel_doc_link(self):
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._modinfo = lambda module: {}
        try:
            result = nvidia_fs.check_driver_type()
        finally:
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertIn("Open Kernel Module", result.mitigation)
        self.assertNotIn("apt-get", result.mitigation)
        self.assertNotIn("dnf", result.mitigation)

    def test_missing_compute_capability_uses_driver_doc_link(self):
        old_caps = nvidia_fs._gpu_compute_caps
        nvidia_fs._gpu_compute_caps = lambda: []
        try:
            result = nvidia_fs.check_gpu_compute_capability()
        finally:
            nvidia_fs._gpu_compute_caps = old_caps

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertNotIn("nvidia-smi --query-gpu", result.mitigation)

    def test_missing_cuda_toolkit_is_fail_with_cuda_download_link(self):
        old_which = shutil.which
        old_glob = glob.glob
        shutil.which = lambda command: None
        glob.glob = lambda pattern: []
        try:
            result = nvidia_fs.check_cuda_toolkit()
        finally:
            shutil.which = old_which
            glob.glob = old_glob

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", result.mitigation)
        self.assertNotIn("apt-get", result.mitigation)
        self.assertNotIn("dnf", result.mitigation)

    def test_missing_nvidia_fs_names_gds_and_dkms_packages(self):
        old_lsmod = nvidia_fs._lsmod_has
        old_modinfo = nvidia_fs._modinfo
        old_loaded = nvidia_fs._kernel_module_loaded
        nvidia_fs._lsmod_has = lambda module: False
        nvidia_fs._modinfo = lambda module: {}
        nvidia_fs._kernel_module_loaded = lambda module: False
        try:
            result = nvidia_fs.check_nvidia_fs_loaded()
        finally:
            nvidia_fs._lsmod_has = old_lsmod
            nvidia_fs._modinfo = old_modinfo
            nvidia_fs._kernel_module_loaded = old_loaded

        self.assertEqual(result.status, Status.WARN)
        self.assertIn("nvidia-fs-dkms", result.why)
        self.assertIn("nvidia-gds", result.why)
        self.assertIn("not a global GDS blocker", result.why)
        self.assertIn("direct P2P", result.why)
        self.assertIn("C2C", result.why)
        self.assertIn("nvidia-fs-dkms", result.mitigation)
        self.assertIn("nvidia-gds", result.mitigation)
        self.assertIn(
            "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
            result.mitigation,
        )
        self.assertIn("sudo apt-get install nvidia-gds", result.mitigation)
        self.assertIn("sudo dnf install nvidia-gds", result.mitigation)
        self.assertIn("nvidia-gds-13-2", result.mitigation)
        self.assertIn("apt-cache search nvidia-gds", result.mitigation)
        self.assertIn("apt-cache search nvidia-fs", result.mitigation)
        self.assertIn(
            "dnf list --available 'nvidia-gds*' 'nvidia-fs*'",
            result.mitigation,
        )
        self.assertIn("sudo dnf install nvidia-fs-dkms", result.mitigation)
        self.assertNotIn("nvidia-fs/nvidia-fs-dkms", result.why)
        self.assertNotIn("install nvidia-fs\n", result.mitigation)

    def test_installed_but_unloaded_nvidia_fs_is_route_warning(self):
        old_lsmod = nvidia_fs._lsmod_has
        old_modinfo = nvidia_fs._modinfo
        old_loaded = nvidia_fs._kernel_module_loaded
        nvidia_fs._lsmod_has = lambda module: False
        nvidia_fs._modinfo = lambda module: (
            {"version": "2.26.6"} if module == "nvidia_fs" else {}
        )
        nvidia_fs._kernel_module_loaded = lambda module: False
        try:
            result = nvidia_fs.check_nvidia_fs_loaded()
        finally:
            nvidia_fs._lsmod_has = old_lsmod
            nvidia_fs._modinfo = old_modinfo
            nvidia_fs._kernel_module_loaded = old_loaded

        self.assertEqual(result.status, Status.WARN)
        self.assertIn("not a global GDS blocker", result.why)
        self.assertIn("Direct P2P", result.why)
        self.assertIn("C2C", result.why)

    def test_loaded_nvidia_fs_uses_sysfs_version_when_modinfo_missing(self):
        old_loaded = nvidia_fs._kernel_module_loaded
        old_sysfs = nvidia_fs._module_sysfs_attr
        old_modinfo = nvidia_fs._modinfo
        nvidia_fs._kernel_module_loaded = lambda module: module == "nvidia_fs"
        nvidia_fs._module_sysfs_attr = lambda module, attr: (
            "2.28.0" if module == "nvidia_fs" and attr == "version" else None
        )
        nvidia_fs._modinfo = lambda module: {}
        try:
            result = nvidia_fs.check_nvidia_fs_loaded()
        finally:
            nvidia_fs._kernel_module_loaded = old_loaded
            nvidia_fs._module_sysfs_attr = old_sysfs
            nvidia_fs._modinfo = old_modinfo

        self.assertEqual(result.status, Status.PASS)
        self.assertIn("2.28.0", result.why)
        self.assertIn("/sys/module/nvidia_fs/version", result.evidence)

    def test_libcufile_version_check_reports_api_and_file_version(self):
        old_detect = cufile_version.detect_libcufile
        try:
            cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.16.1",
                "api_version_int": 1160,
                "api_version": "1.16",
                "file_version": "1.16.1",
                "gds_release_version": "1.16.1.26",
            }
            result = nvidia_fs.check_libcufile_version("GDS release version: 1.16.1.26")
        finally:
            cufile_version.detect_libcufile = old_detect

        self.assertEqual(result.status, Status.PASS)
        self.assertIn("API=1.16", result.why)
        self.assertIn("file=1.16.1", result.why)

    def test_libcufile_version_check_passes_when_loaded_but_api_version_unavailable(self):
        # GDS 1.7.x (CUDA 12.2) doesn't export cuFileGetVersion, but the
        # library is genuinely installed and in use -- this must not report
        # FAIL/"install the matching libcufile package" for something that's
        # already correctly installed.
        old_detect = cufile_version.detect_libcufile
        try:
            cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.7.2",
                "api_version_int": None,
                "api_version": None,
                "file_version": "1.7.2",
                "gds_release_version": "1.7.2.10",
                "probe_errors": ["/usr/local/cuda/lib64/libcufile.so: undefined symbol: cuFileGetVersion"],
            }
            result = nvidia_fs.check_libcufile_version("GDS release version: 1.7.2.10")
        finally:
            cufile_version.detect_libcufile = old_detect

        self.assertEqual(result.status, Status.PASS)
        self.assertNotIn("was not found", result.why)
        self.assertIn("cuFileGetVersion unavailable", result.why)
        self.assertIn("file=1.7.2", result.why)
        self.assertIn("gdscheck release=1.7.2.10", result.why)
        self.assertIn("probe_error=", result.evidence)

    def test_kernel_log_permission_failure_reports_unperformed_check(self):
        old_kmsg = nvidia_fs._kmsg_nvidia_fs
        try:
            nvidia_fs._kmsg_nvidia_fs = lambda: kmsg.KernelLogResult(
                (),
                error="sudo -n dmesg: sudo: a password is required",
                permission_denied=True,
            )

            results = nvidia_fs.check_kmsg_nvidia_fs()
        finally:
            nvidia_fs._kmsg_nvidia_fs = old_kmsg

        self.assertEqual(results[0].status, Status.WARN)
        self.assertIn("Could not inspect", results[0].why)
        self.assertIn("not performed", results[0].why)
        self.assertIn("password is required", results[0].evidence)
        self.assertIn("sudo to enable the extra privileged validation checks", results[0].mitigation)
        self.assertIn("gds-diag.py", results[0].mitigation)
        self.assertIn("sudo journalctl -k -b | grep nvidia_fs", results[0].mitigation)
        self.assertIn("sudo dmesg | grep nvidia_fs", results[0].mitigation)

    def test_kernel_log_readable_with_no_nvidia_fs_lines_is_info(self):
        old_kmsg = nvidia_fs._kmsg_nvidia_fs
        try:
            nvidia_fs._kmsg_nvidia_fs = lambda: kmsg.KernelLogResult(
                (),
                source="sudo -n journalctl -k -b",
            )

            results = nvidia_fs.check_kmsg_nvidia_fs()
        finally:
            nvidia_fs._kmsg_nvidia_fs = old_kmsg

        self.assertEqual(results[0].status, Status.INFO)
        self.assertIn("Kernel logs were readable", results[0].why)
        self.assertIn("source=sudo -n journalctl", results[0].evidence)

    def test_empty_dmesg_fallback_is_warning_not_no_messages_info(self):
        old_kmsg = nvidia_fs._kmsg_nvidia_fs
        try:
            nvidia_fs._kmsg_nvidia_fs = lambda: kmsg.KernelLogResult(
                (),
                source="sudo -n dmesg",
                error="sudo -n journalctl -k -b: sudo: a password is required",
                permission_denied=True,
            )

            results = nvidia_fs.check_kmsg_nvidia_fs()
        finally:
            nvidia_fs._kmsg_nvidia_fs = old_kmsg

        self.assertEqual(results[0].status, Status.WARN)
        self.assertIn("Only the dmesg kernel ring buffer was readable", results[0].why)
        self.assertIn("dmesg can wrap", results[0].why)
        self.assertIn("fallback_reason=", results[0].evidence)

    def test_version_context_reports_cuda_driver_and_libcufile(self):
        old_cuda = nvidia_fs._cuda_toolkit_info
        old_driver_text = nvidia_fs._nvidia_driver_version_text
        old_modinfo = nvidia_fs._modinfo
        old_loaded = nvidia_fs._kernel_module_loaded
        old_detect = cufile_version.detect_libcufile
        try:
            nvidia_fs._cuda_toolkit_info = lambda: {
                "found": True,
                "complete": True,
                "version": "13.1",
                "source": "nvcc: /usr/local/cuda/bin/nvcc",
                "detail": "Cuda compilation tools, release 13.1",
            }
            nvidia_fs._nvidia_driver_version_text = lambda: ("575.57.08", "nvidia-smi")
            nvidia_fs._modinfo = lambda module: (
                {"version": "2.26.6"} if module == "nvidia_fs" else {}
            )
            nvidia_fs._kernel_module_loaded = lambda module: False
            cufile_version.detect_libcufile = lambda output=None: {
                "found": True,
                "path": "/usr/local/cuda/lib64/libcufile.so",
                "realpath": "/usr/local/cuda/lib64/libcufile.so.1.16.1",
                "api_version": "1.16",
                "file_version": "1.16.1",
                "gds_release_version": "1.16.1.26",
            }
            rows = nvidia_fs.collect_version_context(include_runtime_driver=True)
        finally:
            nvidia_fs._cuda_toolkit_info = old_cuda
            nvidia_fs._nvidia_driver_version_text = old_driver_text
            nvidia_fs._modinfo = old_modinfo
            nvidia_fs._kernel_module_loaded = old_loaded
            cufile_version.detect_libcufile = old_detect

        rendered = {row["component"]: row for row in rows}
        self.assertEqual(rendered["CUDA Toolkit"]["version"], "13.1")
        self.assertEqual(rendered["NVIDIA driver"]["version"], "575.57.08")
        self.assertEqual(rendered["nvidia-fs"]["version"], "2.26.6")
        self.assertEqual(rendered["libcufile"]["version"], "1.16.1")

    def test_version_context_uses_loaded_sysfs_nvidia_fs_when_modinfo_missing(self):
        old_cuda = nvidia_fs._cuda_toolkit_info
        old_loaded = nvidia_fs._kernel_module_loaded
        old_sysfs = nvidia_fs._module_sysfs_attr
        old_modinfo = nvidia_fs._modinfo
        old_detect = cufile_version.detect_libcufile
        try:
            nvidia_fs._cuda_toolkit_info = lambda: {
                "found": True,
                "version": "13.0",
                "source": "nvcc",
                "detail": "Cuda compilation tools, release 13.0",
            }
            nvidia_fs._kernel_module_loaded = lambda module: module == "nvidia_fs"
            nvidia_fs._module_sysfs_attr = lambda module, attr: (
                "2.28.0" if module == "nvidia_fs" and attr == "version" else None
            )
            nvidia_fs._modinfo = lambda module: (
                {"version": "580.82.07"} if module == "nvidia" else {}
            )
            cufile_version.detect_libcufile = lambda output=None: {"found": False}
            rows = nvidia_fs.collect_version_context(include_runtime_driver=False)
        finally:
            nvidia_fs._cuda_toolkit_info = old_cuda
            nvidia_fs._kernel_module_loaded = old_loaded
            nvidia_fs._module_sysfs_attr = old_sysfs
            nvidia_fs._modinfo = old_modinfo
            cufile_version.detect_libcufile = old_detect

        rendered = {row["component"]: row for row in rows}
        self.assertTrue(rendered["nvidia-fs"]["found"])
        self.assertEqual(rendered["nvidia-fs"]["version"], "2.28.0")
        self.assertEqual(rendered["nvidia-fs"]["source"], "/sys/module/nvidia_fs/version")

    def test_preinstall_version_context_avoids_nvidia_smi(self):
        old_cuda = nvidia_fs._cuda_toolkit_info
        old_driver_text = nvidia_fs._nvidia_driver_version_text
        old_modinfo = nvidia_fs._modinfo
        old_loaded = nvidia_fs._kernel_module_loaded
        old_detect = cufile_version.detect_libcufile
        try:
            nvidia_fs._cuda_toolkit_info = lambda: {
                "found": False,
                "version": None,
                "source": None,
                "detail": "missing",
            }
            nvidia_fs._nvidia_driver_version_text = lambda: self.fail("pre-install version context should not call nvidia-smi")
            nvidia_fs._modinfo = lambda module: {"version": "575.57.08"} if module == "nvidia" else {}
            nvidia_fs._kernel_module_loaded = lambda module: False
            cufile_version.detect_libcufile = lambda output=None: {"found": False}
            rows = nvidia_fs.collect_version_context(include_runtime_driver=False)
        finally:
            nvidia_fs._cuda_toolkit_info = old_cuda
            nvidia_fs._nvidia_driver_version_text = old_driver_text
            nvidia_fs._modinfo = old_modinfo
            nvidia_fs._kernel_module_loaded = old_loaded
            cufile_version.detect_libcufile = old_detect

        driver = next(row for row in rows if row["component"] == "NVIDIA driver")
        self.assertEqual(driver["version"], "575.57.08")
        self.assertEqual(driver["source"], "modinfo nvidia")


if __name__ == "__main__":
    unittest.main()
