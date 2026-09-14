# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks import fs_matrix
from checks import iommu
from checks import kernel
from checks import nvidia_fs
from checks import pcie
from checks import rdma
from checks.result import CheckResult, GDSMode, Status
from subcommands import pre_install


class PreInstallTests(unittest.TestCase):
    def test_preinstall_decodes_proc_mounts_mountpoint_before_probing(self):
        old_check_odirect = fs_matrix.check_odirect
        try:
            probed_paths = []

            def fake_check_odirect(path):
                probed_paths.append(path)
                return False, "O_DIRECT rejected"

            fs_matrix.check_odirect = fake_check_odirect

            results = pre_install._collect_mounted_filesystem_results([
                "/dev/nvme0n1p1 /mnt/gds\\040data xfs rw,relatime 0 0\n",
            ])
        finally:
            fs_matrix.check_odirect = old_check_odirect

        self.assertEqual(probed_paths, ["/mnt/gds data"])
        self.assertEqual(results[0].check, "Mount: /mnt/gds data")

    def test_preinstall_reframes_missing_p2pdma_as_nvfs_route_advisory(self):
        results = pre_install._preinstall_p2pdma_results([
            CheckResult(
                check="P2PDMA kernel support",
                mode=GDSMode.P2PDMA,
                status=Status.FAIL,
                why="No known P2PDMA symbol was found.",
                mitigation="Upgrade the kernel.",
                evidence="checked P2PDMA symbols",
            ),
            CheckResult(
                check="CONFIG_PCI_P2PDMA",
                mode=GDSMode.P2PDMA,
                status=Status.FAIL,
                why="CONFIG_PCI_P2PDMA is not set.",
                mitigation="Upgrade the kernel.",
                evidence="CONFIG_PCI_P2PDMA=not set",
            ),
        ])

        self.assertEqual(len(results), 1)
        advisory = results[0]
        self.assertEqual(advisory.check, "PCI P2PDMA route")
        self.assertEqual(advisory.status, Status.WARN)
        self.assertIn("does not include PCI P2PDMA support", advisory.why)
        self.assertNotIn(
            "Upstream NVMe↔GPU P2PDMA is not available on this host",
            advisory.why,
        )
        self.assertIn("Use the nvidia-fs/nvfs route", advisory.mitigation)
        self.assertIn("MLNX_OFED/DOCA", advisory.mitigation)
        self.assertIn("./gds-diag.py post-install -v", advisory.mitigation)
        self.assertIn("./gds-diag.py mount-check <path> -v", advisory.mitigation)
        self.assertIn(
            "Only change kernels if this deployment specifically requires "
            "upstream PCI P2PDMA",
            advisory.mitigation,
        )
        self.assertIn("checked P2PDMA symbols", advisory.evidence)
        self.assertIn("CONFIG_PCI_P2PDMA=not set", advisory.evidence)

    def test_preinstall_preserves_passing_p2pdma_checks(self):
        original = [
            CheckResult(
                check="P2PDMA kernel support",
                mode=GDSMode.P2PDMA,
                status=Status.PASS,
                why="P2PDMA symbol found.",
            ),
            CheckResult(
                check="CONFIG_PCI_P2PDMA",
                mode=GDSMode.P2PDMA,
                status=Status.PASS,
                why="CONFIG_PCI_P2PDMA=y.",
            ),
        ]

        self.assertIs(pre_install._preinstall_p2pdma_results(original), original)

    def test_preinstall_suppresses_runtime_tmpfs_mounts(self):
        old_check_odirect = fs_matrix.check_odirect
        try:
            probed_paths = []

            def fake_check_odirect(path):
                probed_paths.append(path)
                return False, "O_DIRECT rejected"

            fs_matrix.check_odirect = fake_check_odirect

            results = pre_install._collect_mounted_filesystem_results([
                "tmpfs /dev/shm tmpfs rw,nosuid,nodev,inode64 0 0\n",
                "tmpfs /run/lock tmpfs rw,nosuid,nodev,noexec 0 0\n",
                "tmpfs /run/user/151648 tmpfs rw,nosuid,nodev,relatime 0 0\n",
                "/dev/md2 / xfs rw,noatime 0 0\n",
            ])
        finally:
            fs_matrix.check_odirect = old_check_odirect

        self.assertEqual(probed_paths, ["/"])
        self.assertEqual([result.check for result in results], ["Mount: /"])

    def test_preinstall_ext4_data_mode_is_mount_specific_warning(self):
        old_ext4 = fs_matrix.check_ext4_data_mode
        old_odirect = fs_matrix.check_odirect
        try:
            fs_matrix.check_ext4_data_mode = lambda path: (
                "default",
                f"{path} opts:rw,relatime,stripe=96",
            )
            fs_matrix.check_odirect = lambda path: self.fail("ext4 data warning should skip O_DIRECT probe")

            results = pre_install._collect_mounted_filesystem_results([
                "/dev/md3 /raid ext4 rw,relatime,stripe=96 0 0\n",
            ])
        finally:
            fs_matrix.check_ext4_data_mode = old_ext4
            fs_matrix.check_odirect = old_odirect

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check, "ext4 data mode (/raid)")
        self.assertEqual(results[0].status, Status.WARN)
        self.assertIn("See the following for more information once GDS is installed", results[0].mitigation)
        self.assertIn("./gds-diag.py mount-check /raid -v", results[0].mitigation)

    def test_preinstall_odirect_failure_is_mount_specific_warning(self):
        old_check_odirect = fs_matrix.check_odirect
        try:
            fs_matrix.check_odirect = lambda path: (False, "O_DIRECT rejected")

            results = pre_install._collect_mounted_filesystem_results([
                "/dev/md0 /boot ext2 rw,noatime 0 0\n",
            ])
        finally:
            fs_matrix.check_odirect = old_check_odirect

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check, "Mount: /boot")
        self.assertEqual(results[0].status, Status.WARN)
        self.assertIn("If this mount will be used for GDS workloads", results[0].why)
        self.assertIn("./gds-diag.py mount-check /boot -v", results[0].mitigation)

    def test_missing_driver_skips_dependent_gpu_checks(self):
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_preinstall = nvidia_fs.check_open_driver_preinstall
        old_version = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        old_compute = nvidia_fs.check_gpu_compute_capability
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_query_bdfs = pcie._query_gpu_bdfs
        old_is_grace = iommu.is_grace
        try:
            iommu.is_grace = lambda: False
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_open_driver_preinstall = lambda: CheckResult(
                check="NVIDIA Open Driver install",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="Driver missing.",
                mitigation="Install driver.",
            )
            nvidia_fs.check_driver_version = lambda: self.fail("driver version should be skipped")
            nvidia_fs.check_driver_type = lambda: self.fail("driver type should be skipped")
            nvidia_fs.check_gpu_compute_capability = lambda: self.fail("compute capability should be skipped")
            pcie._query_gpu_bdfs = lambda: self.fail("pre-install should not call nvidia-smi GPU BDF query")
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="CUDA found.",
            )

            sections = pre_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_open_driver_preinstall = old_preinstall
            nvidia_fs.check_driver_version = old_version
            nvidia_fs.check_driver_type = old_type
            nvidia_fs.check_gpu_compute_capability = old_compute
            nvidia_fs.check_cuda_toolkit = old_cuda
            pcie._query_gpu_bdfs = old_query_bdfs
            iommu.is_grace = old_is_grace

        checks = [result.check for result in sections["GPU"]]
        self.assertEqual(checks, ["NVIDIA GPU (lspci)", "NVIDIA Open Driver install"])
        self.assertEqual(sections["GPU"][1].status, Status.FAIL)
        self.assertNotIn("NVIDIA driver version", checks)
        self.assertNotIn("NVIDIA driver type", checks)
        self.assertNotIn("GPU compute capability", checks)

    def test_preinstall_gpu_section_uses_install_prerequisite_only(self):
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_preinstall = nvidia_fs.check_open_driver_preinstall
        old_version = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        old_compute = nvidia_fs.check_gpu_compute_capability
        old_is_grace = iommu.is_grace
        try:
            iommu.is_grace = lambda: False
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_open_driver_preinstall = lambda: CheckResult(
                check="NVIDIA Open Driver install",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="Open driver installed.",
            )
            nvidia_fs.check_driver_version = lambda: self.fail("driver version should be post-install only")
            nvidia_fs.check_driver_type = lambda: self.fail("driver type should be post-install only")
            nvidia_fs.check_gpu_compute_capability = lambda: self.fail("compute capability should be post-install only")

            sections = pre_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_open_driver_preinstall = old_preinstall
            nvidia_fs.check_driver_version = old_version
            nvidia_fs.check_driver_type = old_type
            nvidia_fs.check_gpu_compute_capability = old_compute
            iommu.is_grace = old_is_grace

        self.assertEqual(
            [result.check for result in sections["GPU"]],
            ["NVIDIA GPU (lspci)", "NVIDIA Open Driver install"],
        )

    def test_missing_cuda_toolkit_is_preinstall_failure(self):
        old_cuda = nvidia_fs.check_cuda_toolkit
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="CUDA Toolkit not found.",
                mitigation="Install CUDA Toolkit.",
            )

            sections = pre_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda

        self.assertEqual(sections["CUDA Toolkit"][0].status, Status.FAIL)

    def test_preinstall_includes_doca_mofed_advisory_warning(self):
        old_check = rdma.check_ofed_preinstall
        try:
            rdma.check_ofed_preinstall = lambda: CheckResult(
                check="MLNX_OFED / DOCA",
                mode=GDSMode.RDMA,
                status=Status.WARN,
                why="MLNX_OFED/DOCA not detected.",
                mitigation="Install MLNX_OFED or DOCA if using RDMA routes.",
            )

            sections = pre_install._collect_sections(verbose=False)
        finally:
            rdma.check_ofed_preinstall = old_check

        self.assertIn("DOCA / MLNX_OFED", sections)
        self.assertEqual(sections["DOCA / MLNX_OFED"][0].status, Status.WARN)


class CdmmDetectionTests(unittest.TestCase):
    """_check_cdmm() — Grace-only CDMM detection."""

    def setUp(self):
        self._old_is_grace = iommu.is_grace
        self._old_cdmm_value = nvidia_fs.coherent_gpu_memory_mode_value

    def tearDown(self):
        iommu.is_grace = self._old_is_grace
        nvidia_fs.coherent_gpu_memory_mode_value = self._old_cdmm_value

    def test_skipped_on_non_grace(self):
        iommu.is_grace = lambda: False
        nvidia_fs.coherent_gpu_memory_mode_value = lambda: self.fail(
            "coherent_gpu_memory_mode_value should not be called on non-Grace"
        )
        self.assertIsNone(pre_install._check_cdmm())

    def test_grace_with_cdmm_active(self):
        iommu.is_grace = lambda: True
        nvidia_fs.coherent_gpu_memory_mode_value = lambda: "driver"
        result = pre_install._check_cdmm()
        self.assertIsNotNone(result)
        self.assertEqual(result.check, "CDMM mode")
        self.assertEqual(result.status, Status.PASS)
        self.assertIn("CDMM", result.why)
        self.assertIn("driver", result.evidence)

    def test_grace_with_numa_mode(self):
        iommu.is_grace = lambda: True
        nvidia_fs.coherent_gpu_memory_mode_value = lambda: "numa"
        result = pre_install._check_cdmm()
        self.assertEqual(result.status, Status.PASS)
        self.assertIn("NUMA", result.why)
        self.assertIn("numa", result.evidence)

    def test_grace_driver_not_loaded(self):
        iommu.is_grace = lambda: True
        nvidia_fs.coherent_gpu_memory_mode_value = lambda: None
        result = pre_install._check_cdmm()
        self.assertEqual(result.status, Status.WARN)
        self.assertIn("driver", result.mitigation)

    def test_grace_param_empty_means_default_behavior(self):
        # Driver loaded but module param is empty — CDMM is NOT active and the
        # driver is in its default coherent-memory behavior. This must not be
        # treated as "driver not loaded".
        iommu.is_grace = lambda: True
        nvidia_fs.coherent_gpu_memory_mode_value = lambda: ""
        result = pre_install._check_cdmm()
        self.assertEqual(result.status, Status.PASS)
        self.assertIn("not explicitly enabled", result.why)
        self.assertIn("NVreg_RegistryDwords", result.mitigation)


class KernelVersionInfoTests(unittest.TestCase):
    """_check_kernel_version_info() — informational kernel version report."""

    def setUp(self):
        self._old_kernel_version = kernel._kernel_version

    def tearDown(self):
        kernel._kernel_version = self._old_kernel_version

    def test_reports_running_kernel_version(self):
        kernel._kernel_version = lambda: (6, 11, 3)
        results = pre_install._check_kernel_version_info()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check, "Kernel version")
        self.assertEqual(results[0].status, Status.PASS)
        self.assertIn("6.11.3", results[0].why)


class IommuMitigationTests(unittest.TestCase):
    def setUp(self):
        self._old_cpu_vendor = iommu._cpu_vendor
        self._old_detect = iommu.detect_iommu_state
        self._old_arch = iommu._arch
        self._old_is_grace = iommu._is_grace
        self._old_has_local_nvme = iommu._has_local_nvme_devices

    def tearDown(self):
        iommu._cpu_vendor = self._old_cpu_vendor
        iommu.detect_iommu_state = self._old_detect
        iommu._arch = self._old_arch
        iommu._is_grace = self._old_is_grace
        iommu._has_local_nvme_devices = self._old_has_local_nvme

    def test_x86_strict_mitigation_uses_intel_cmdline_only(self):
        iommu._cpu_vendor = lambda: iommu.CpuVendor.INTEL

        mitigation = iommu._strict_mitigation(iommu.Arch.X86)

        self.assertIn("intel_iommu=on iommu=pt", mitigation)
        self.assertIn("intel_iommu=off", mitigation)
        self.assertNotIn("amd_iommu=on", mitigation)
        self.assertNotIn("amd_iommu=off", mitigation)

    def test_x86_strict_mitigation_uses_amd_cmdline_only(self):
        iommu._cpu_vendor = lambda: iommu.CpuVendor.AMD

        mitigation = iommu._strict_mitigation(iommu.Arch.X86)

        self.assertIn("amd_iommu=on iommu=pt", mitigation)
        self.assertIn("amd_iommu=off", mitigation)
        self.assertNotIn("intel_iommu=on", mitigation)
        self.assertNotIn("intel_iommu=off", mitigation)

    def test_x86_strict_mitigation_keeps_both_options_when_vendor_unknown(self):
        iommu._cpu_vendor = lambda: iommu.CpuVendor.UNKNOWN

        mitigation = iommu._strict_mitigation(iommu.Arch.X86)

        self.assertIn("CPU vendor could not be detected", mitigation)
        self.assertIn("Intel: intel_iommu=on iommu=pt", mitigation)
        self.assertIn("AMD:   amd_iommu=on iommu=pt", mitigation)

    def test_p2pdma_strict_iommu_is_not_warning_without_local_nvme(self):
        iommu.detect_iommu_state = lambda: (
            iommu.IommuState.STRICT,
            "cmdline: BOOT_IMAGE=/vmlinuz",
        )
        iommu._arch = lambda: iommu.Arch.X86
        iommu._is_grace = lambda: False
        iommu._has_local_nvme_devices = lambda: False

        result = iommu.check_iommu_for_p2pdma()

        self.assertEqual(result.status, Status.PASS)
        self.assertIn("no local NVMe PCI devices", result.why)
        self.assertIn("local_nvme_bdfs=none", result.evidence)

    def test_native_strict_iommu_still_warns_without_local_nvme(self):
        iommu.detect_iommu_state = lambda: (
            iommu.IommuState.STRICT,
            "cmdline: BOOT_IMAGE=/vmlinuz",
        )
        iommu._arch = lambda: iommu.Arch.X86
        iommu._is_grace = lambda: False
        iommu._has_local_nvme_devices = lambda: False

        result = iommu.check_iommu_for_native()

        self.assertEqual(result.status, Status.WARN)
        self.assertIn("Native GDS", result.why)


if __name__ == "__main__":
    unittest.main()
