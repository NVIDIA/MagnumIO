# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks.result import CheckResult, GDSMode, Status
from checks import iommu, nvidia_fs, pcie
from subcommands import post_install


GDSCHECK_DRIVER_CONFIG = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : compat
  NVMeOF             : p2pdma, compat
  NFS                : p2pdma, compat
  IBM Spectrum Scale : p2pdma, compat
  ScaTeFS            : p2pdma, compat
=====================
"""


GDSCHECK_DRIVER_CONFIG_C2C = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : c2c, nvfs, compat
  NVMeOF             : compat
=====================
"""


GDSCHECK_DRIVER_CONFIG_OLD_P2PDMA = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe P2PDMA        : Supported
  NVMe               : Supported
  DDN EXAScaler      : Supported
=====================
"""


class PostInstallP2pdmaRouteTests(unittest.TestCase):
    def test_unsupported_p2pdma_tokens_are_not_active_routes(self):
        results = post_install._p2pdma_routes_from_gdscheck(GDSCHECK_DRIVER_CONFIG)

        active = next(result for result in results if result.check == "Active P2PDMA/C2C routes")
        self.assertEqual(active.status.value, "PASS")
        self.assertEqual(active.evidence, "NVMeOF: p2pdma, compat")

        unsupported = next(result for result in results if result.check == "Unsupported P2PDMA tokens")
        self.assertEqual(unsupported.status.value, "WARN")
        self.assertIn("NFS: p2pdma, compat", unsupported.evidence)
        self.assertIn("IBM Spectrum Scale: p2pdma, compat", unsupported.evidence)
        self.assertIn("ScaTeFS: p2pdma, compat", unsupported.evidence)

    def test_has_active_p2pdma_route_ignores_unsupported_only(self):
        unsupported_only = """\
=====================
DRIVER CONFIGURATION:
=====================
  IBM Spectrum Scale : p2pdma, compat
  NFS                : p2pdma, compat
  ScaTeFS            : p2pdma, compat
=====================
"""
        self.assertFalse(post_install._has_active_p2pdma_route(unsupported_only))
        self.assertTrue(post_install._has_active_p2pdma_route(GDSCHECK_DRIVER_CONFIG))

    def test_c2c_nvme_token_is_active_direct_route(self):
        results = post_install._p2pdma_routes_from_gdscheck(GDSCHECK_DRIVER_CONFIG_C2C)

        active = next(result for result in results if result.check == "Active P2PDMA/C2C routes")
        self.assertEqual(active.status.value, "PASS")
        self.assertEqual(active.evidence, "NVMe: c2c, nvfs, compat")
        self.assertIn("P2PDMA/C2C", active.why)
        self.assertTrue(post_install._has_active_p2pdma_route(GDSCHECK_DRIVER_CONFIG_C2C))

    def test_old_nvme_p2pdma_supported_status_is_active_direct_route(self):
        results = post_install._p2pdma_routes_from_gdscheck(GDSCHECK_DRIVER_CONFIG_OLD_P2PDMA)

        active = next(result for result in results if result.check == "Active P2PDMA/C2C routes")
        self.assertEqual(active.status.value, "PASS")
        self.assertEqual(active.evidence, "NVMe P2PDMA: Supported")
        self.assertTrue(post_install._has_active_p2pdma_route(GDSCHECK_DRIVER_CONFIG_OLD_P2PDMA))

    def test_p2pdma_sections_include_nvidia_driver_registry_check(self):
        from checks import kernel

        old_kernel_run_all = kernel.run_all
        old_topology = post_install._p2pdma_topology_candidates
        old_config = post_install._p2pdma_config_routes
        old_acs = pcie.check_acs
        old_registry = nvidia_fs.check_p2pdma_driver_registries
        try:
            kernel.run_all = lambda mode: []
            nvidia_fs.check_p2pdma_driver_registries = lambda: CheckResult(
                check="NVIDIA P2PDMA driver registries",
                mode=GDSMode.P2PDMA,
                status=Status.FAIL,
                why="Missing RMForceStaticBar1=1.",
            )
            post_install._p2pdma_config_routes = lambda: []
            post_install._p2pdma_topology_candidates = lambda: []
            pcie.check_acs = lambda: CheckResult(
                check="PCIe ACS redirect",
                mode=GDSMode.P2PDMA,
                status=Status.PASS,
                why="ACS redirect disabled.",
            )

            results = post_install._collect_p2pdma_sections("")
        finally:
            kernel.run_all = old_kernel_run_all
            post_install._p2pdma_topology_candidates = old_topology
            post_install._p2pdma_config_routes = old_config
            pcie.check_acs = old_acs
            nvidia_fs.check_p2pdma_driver_registries = old_registry

        checks = [result.check for result in results]
        self.assertIn("NVIDIA P2PDMA driver registries", checks)

    def test_post_install_stops_when_base_prerequisites_are_missing(self):
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_driver_installed = nvidia_fs.check_driver_installed
        old_driver = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.WARN,
                why="CUDA Toolkit not found.",
                mitigation="install cuda",
            )
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_driver_installed = lambda: CheckResult(
                check="NVIDIA driver installed",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="NVIDIA driver not installed.",
                mitigation="install driver",
            )
            nvidia_fs.check_driver_version = lambda: self.fail("driver version should be skipped")
            nvidia_fs.check_driver_type = lambda: self.fail("driver type should be skipped")

            sections = post_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_driver_installed = old_driver_installed
            nvidia_fs.check_driver_version = old_driver
            nvidia_fs.check_driver_type = old_type

        self.assertEqual(list(sections), ["Prerequisites"])
        self.assertTrue(any(result.status == Status.FAIL for result in sections["Prerequisites"]))
        self.assertNotIn(
            "NVIDIA driver version",
            [result.check for result in sections["Prerequisites"]],
        )
        gate = sections["Prerequisites"][-1]
        self.assertEqual(gate.check, "Post-install prerequisite gate")
        self.assertIn("rerun", gate.mitigation)

        rows = post_install._prerequisite_rows(sections["Prerequisites"])
        self.assertEqual(rows[0]["component"], "CUDA Toolkit")
        self.assertIn("CUDA Toolkit is missing", rows[0]["missing_or_issue"])
        self.assertEqual(rows[0]["docs"], "[1]")
        self.assertIn("gdscheck", rows[-1]["recommendation"])

        table = "\n".join(post_install._render_prerequisite_table(sections["Prerequisites"]))
        self.assertIn("Prerequisite Remediation Table", table)
        self.assertIn("Missing / Issue", table)
        self.assertIn("https://developer.nvidia.com/cuda-downloads", table)

    def test_post_install_warns_system_memory_only_without_gpu_or_driver(self):
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_driver_installed = nvidia_fs.check_driver_installed
        old_driver = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="CUDA Toolkit found.",
            )
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="No NVIDIA GPU found.",
                mitigation="install GPU",
            )
            nvidia_fs.check_driver_installed = lambda: CheckResult(
                check="NVIDIA driver installed",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="NVIDIA driver not installed.",
                mitigation="install driver",
            )
            nvidia_fs.check_driver_version = lambda: self.fail("driver version should be skipped")
            nvidia_fs.check_driver_type = lambda: self.fail("driver type should be skipped")

            sections = post_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_driver_installed = old_driver_installed
            nvidia_fs.check_driver_version = old_driver
            nvidia_fs.check_driver_type = old_type

        memory_result = next(
            result for result in sections["Prerequisites"]
            if result.check == "libcufile GPU memory support"
        )
        self.assertEqual(memory_result.status, Status.WARN)
        self.assertIn("Only system-memory buffers are supported", memory_result.why)
        self.assertIn("GPU-memory buffers are not supported with libcufile", memory_result.why)

    def test_post_install_warns_system_memory_only_when_runtime_driver_unhealthy(self):
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_driver_installed = nvidia_fs.check_driver_installed
        old_driver = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="CUDA Toolkit found.",
            )
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_driver_installed = lambda: CheckResult(
                check="NVIDIA driver installed",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="modinfo nvidia found.",
            )
            nvidia_fs.check_driver_version = lambda: CheckResult(
                check="NVIDIA driver version",
                mode=GDSMode.NATIVE,
                status=Status.FAIL,
                why="nvidia-smi unavailable.",
                mitigation="repair driver",
            )
            nvidia_fs.check_driver_type = lambda: CheckResult(
                check="NVIDIA driver type",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="open driver.",
            )

            sections = post_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_driver_installed = old_driver_installed
            nvidia_fs.check_driver_version = old_driver
            nvidia_fs.check_driver_type = old_type

        checks = [result.check for result in sections["Prerequisites"]]
        self.assertIn("NVIDIA driver version", checks)
        self.assertIn("libcufile GPU memory support", checks)

    def test_prerequisite_table_distinguishes_installed_driver_from_missing_gpu_runtime(self):
        result = CheckResult(
            check="NVIDIA driver version",
            mode=GDSMode.NATIVE,
            status=Status.FAIL,
            why=(
                "NVIDIA driver module is installed (version 580.65.06), but "
                "nvidia-smi did not return a runtime driver version."
            ),
            mitigation="verify nvidia-smi -L",
        )

        rows = post_install._prerequisite_rows([result])

        self.assertEqual(
            rows[0]["missing_or_issue"],
            "Driver installed, but runtime GPU validation is unavailable",
        )
        self.assertIn("nvidia-smi -L", rows[0]["mitigation"])
        self.assertNotIn("driver is missing", rows[0]["missing_or_issue"].lower())

    def test_post_install_continues_after_prerequisites_pass(self):
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_driver_installed = nvidia_fs.check_driver_installed
        old_driver = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        old_find = post_install._find_gdscheck
        old_loaded = nvidia_fs.check_nvidia_fs_loaded
        old_open = nvidia_fs.check_open_driver
        old_run = post_install._run_gdscheck_raw
        old_p2p = post_install._collect_p2pdma_sections
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="CUDA Toolkit found.",
            )
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_driver_installed = lambda: CheckResult(
                check="NVIDIA driver installed",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="driver installed.",
            )
            nvidia_fs.check_driver_version = lambda: CheckResult(
                check="NVIDIA driver version",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="driver found.",
            )
            nvidia_fs.check_driver_type = lambda: CheckResult(
                check="NVIDIA driver type",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="open driver found.",
            )
            post_install._find_gdscheck = lambda: "/usr/local/cuda/gds/tools/gdscheck"
            post_install._run_gdscheck_raw = lambda path, apply_env=True: (GDSCHECK_DRIVER_CONFIG, None)
            nvidia_fs.check_nvidia_fs_loaded = lambda: CheckResult(
                check="nvidia-fs module",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="loaded.",
            )
            nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver",
                mode=GDSMode.NATIVE,
                status=Status.PASS,
                why="open.",
            )
            post_install._collect_p2pdma_sections = lambda raw: []

            sections = post_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_driver_installed = old_driver_installed
            nvidia_fs.check_driver_version = old_driver
            nvidia_fs.check_driver_type = old_type
            post_install._find_gdscheck = old_find
            nvidia_fs.check_nvidia_fs_loaded = old_loaded
            nvidia_fs.check_open_driver = old_open
            post_install._run_gdscheck_raw = old_run
            post_install._collect_p2pdma_sections = old_p2p

        self.assertIn("Prerequisites", sections)
        self.assertIn("Installation", sections)
        self.assertNotEqual(list(sections), ["Prerequisites"])

    def _collect_sections_with_gdscheck_output(self, gds_raw: str) -> dict:
        old_cuda = nvidia_fs.check_cuda_toolkit
        old_gpu = nvidia_fs.check_gpu_presence_lspci
        old_driver_installed = nvidia_fs.check_driver_installed
        old_driver = nvidia_fs.check_driver_version
        old_type = nvidia_fs.check_driver_type
        old_find = post_install._find_gdscheck
        old_loaded = nvidia_fs.check_nvidia_fs_loaded
        old_open = nvidia_fs.check_open_driver
        old_run = post_install._run_gdscheck_raw
        old_p2p = post_install._collect_p2pdma_sections
        try:
            nvidia_fs.check_cuda_toolkit = lambda: CheckResult(
                check="CUDA Toolkit", mode=GDSMode.NATIVE, status=Status.PASS,
                why="CUDA Toolkit found.",
            )
            nvidia_fs.check_gpu_presence_lspci = lambda: CheckResult(
                check="NVIDIA GPU (lspci)", mode=GDSMode.NATIVE, status=Status.PASS,
                why="GPU found.",
            )
            nvidia_fs.check_driver_installed = lambda: CheckResult(
                check="NVIDIA driver installed", mode=GDSMode.NATIVE, status=Status.PASS,
                why="driver installed.",
            )
            nvidia_fs.check_driver_version = lambda: CheckResult(
                check="NVIDIA driver version", mode=GDSMode.NATIVE, status=Status.PASS,
                why="driver found.",
            )
            nvidia_fs.check_driver_type = lambda: CheckResult(
                check="NVIDIA driver type", mode=GDSMode.NATIVE, status=Status.PASS,
                why="open driver found.",
            )
            post_install._find_gdscheck = lambda: "/usr/local/cuda/gds/tools/gdscheck"
            post_install._run_gdscheck_raw = lambda path, apply_env=True: (gds_raw, None)
            nvidia_fs.check_nvidia_fs_loaded = lambda: CheckResult(
                check="nvidia-fs module", mode=GDSMode.NATIVE, status=Status.PASS,
                why="loaded.",
            )
            nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver", mode=GDSMode.NATIVE, status=Status.PASS,
                why="open.",
            )
            post_install._collect_p2pdma_sections = lambda raw: []

            return post_install._collect_sections(verbose=False)
        finally:
            nvidia_fs.check_cuda_toolkit = old_cuda
            nvidia_fs.check_gpu_presence_lspci = old_gpu
            nvidia_fs.check_driver_installed = old_driver_installed
            nvidia_fs.check_driver_version = old_driver
            nvidia_fs.check_driver_type = old_type
            post_install._find_gdscheck = old_find
            nvidia_fs.check_nvidia_fs_loaded = old_loaded
            nvidia_fs.check_open_driver = old_open
            post_install._run_gdscheck_raw = old_run
            post_install._collect_p2pdma_sections = old_p2p

    def test_cufile_checks_receive_gdscheck_output(self):
        from checks import cufile_config

        calls = []
        old_run_all = cufile_config.run_all
        try:
            cufile_config.run_all = lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: (
                calls.append(gdscheck_output) or []
            )
            self._collect_sections_with_gdscheck_output(GDSCHECK_DRIVER_CONFIG)
        finally:
            cufile_config.run_all = old_run_all

        self.assertEqual(calls, [GDSCHECK_DRIVER_CONFIG])

    def test_gdscheck_iommu_warning_includes_grub_mitigation_when_strict(self):
        gds_raw_with_iommu_warning = (
            GDSCHECK_DRIVER_CONFIG
            + "WARN: GDS is not guaranteed to work functionally or in a "
              "performant way with iommu=on/pt\n"
        )

        old_detect = iommu.detect_iommu_state
        try:
            iommu.detect_iommu_state = lambda: (iommu.IommuState.STRICT, "mocked strict")
            sections = self._collect_sections_with_gdscheck_output(gds_raw_with_iommu_warning)
        finally:
            iommu.detect_iommu_state = old_detect

        driver_results = sections["GDS Driver Configuration"]
        warning = next(r for r in driver_results if r.check == "gdscheck warning")
        self.assertEqual(warning.status, Status.WARN)
        self.assertIsNotNone(warning.mitigation)
        self.assertIn("GRUB_CMDLINE_LINUX", warning.mitigation)
        expected_token = "iommu.passthrough=1" if iommu.arch() == iommu.Arch.ARM else "iommu=pt"
        self.assertIn(expected_token, warning.mitigation)

    def test_gdscheck_iommu_warning_is_advisory_when_already_passthrough_x86(self):
        gds_raw_with_iommu_warning = (
            GDSCHECK_DRIVER_CONFIG
            + "WARN: GDS is not guaranteed to work functionally or in a "
              "performant way with iommu=on/pt\n"
        )

        old_detect = iommu.detect_iommu_state
        old_arch = iommu.arch
        try:
            iommu.detect_iommu_state = lambda: (iommu.IommuState.PASSTHROUGH, "mocked passthrough")
            iommu.arch = lambda: iommu.Arch.X86
            sections = self._collect_sections_with_gdscheck_output(gds_raw_with_iommu_warning)
        finally:
            iommu.detect_iommu_state = old_detect
            iommu.arch = old_arch

        driver_results = sections["GDS Driver Configuration"]
        warning = next(r for r in driver_results if r.check == "gdscheck warning")
        # Stays WARN so it's still visible in non-verbose output (PASS results
        # are hidden by render_sections), but must not tell the operator to
        # set passthrough mode they've already set.
        self.assertEqual(warning.status, Status.WARN)
        self.assertIn("already in passthrough mode", warning.why)
        self.assertIsNotNone(warning.mitigation)
        self.assertNotIn("GRUB_CMDLINE_LINUX", warning.mitigation)
        self.assertIn("intel_iommu=off", warning.mitigation)

    def test_gdscheck_iommu_warning_downgrades_to_pass_when_already_passthrough_arm(self):
        gds_raw_with_iommu_warning = (
            GDSCHECK_DRIVER_CONFIG
            + "WARN: GDS is not guaranteed to work functionally or in a "
              "performant way with iommu=on/pt\n"
        )

        old_detect = iommu.detect_iommu_state
        old_arch = iommu.arch
        try:
            iommu.detect_iommu_state = lambda: (iommu.IommuState.PASSTHROUGH, "mocked passthrough")
            iommu.arch = lambda: iommu.Arch.ARM
            sections = self._collect_sections_with_gdscheck_output(gds_raw_with_iommu_warning)
        finally:
            iommu.detect_iommu_state = old_detect
            iommu.arch = old_arch

        driver_results = sections["GDS Driver Configuration"]
        warning = next(r for r in driver_results if r.check == "gdscheck warning")
        # No cmdline knob exists to disable the ARM SMMU entirely, so there is
        # nothing actionable to surface — downgrade to PASS so it stays quiet
        # in non-verbose output instead of showing a WARN with no useful action.
        self.assertEqual(warning.status, Status.PASS)
        self.assertIn("already in passthrough mode", warning.why)
        self.assertIsNone(warning.mitigation)

    def test_cross_root_p2pdma_candidate_is_performance_warning(self):
        old_gpu = pcie.get_gpu_bdfs_with_error
        old_nvme = pcie.get_nvme_bdfs
        old_same = pcie.same_root_complex
        try:
            pcie.get_gpu_bdfs_with_error = lambda: (["0000:65:00.0"], None)
            pcie.get_nvme_bdfs = lambda: ["0000:ca:00.0"]
            pcie.same_root_complex = lambda gpu, nvme: (
                False,
                "No common ancestor",
            )

            results = post_install._p2pdma_topology_candidates()
        finally:
            pcie.get_gpu_bdfs_with_error = old_gpu
            pcie.get_nvme_bdfs = old_nvme
            pcie.same_root_complex = old_same

        result = results[0]
        self.assertEqual(result.status, Status.INFO)
        self.assertIn("GDS can still operate across root ports", result.why)
        self.assertIn("not treat cross-root-port topology alone as a GDS blocker", result.mitigation)
        self.assertNotIn("compat", result.mitigation.lower())


if __name__ == "__main__":
    unittest.main()
