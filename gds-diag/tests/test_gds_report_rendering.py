# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

from checks import gds_report
from checks.result import CheckResult, GDSMode, ModeReport, Status


GDSCHECK_NVME_NVFS_ONLY = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : nvfs, compat
=====================
"""


GDSCHECK_NVME_COMPAT_ONLY = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : compat
=====================
"""


GDSCHECK_NVME_C2C = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : c2c, nvfs, compat
=====================
"""


GDSCHECK_NVME_COMPAT_WITH_P2P_CONFIG_AND_ACS = """\
=====================
DRIVER CONFIGURATION:
=====================
  NVMe               : compat
=====================
CUFILE CONFIGURATION:
=====================
  properties.use_pci_p2pdma : false
  block.nvme.use_pci_p2pdma : false
=====================
PLATFORM INFO:
=====================
  Found ACS enabled for switch 0000:40:01.1
=====================
"""


GDSCHECK_DDN_EXASCALER = """\
=====================
DRIVER CONFIGURATION:
=====================
  DDN EXAScaler      : nvfs, compat
=====================
"""


GDSCHECK_GPFS_MISLEADING_P2PDMA = """\
=====================
DRIVER CONFIGURATION:
=====================
  IBM Spectrum Scale : p2pdma, compat
=====================
"""


GDSCHECK_DDN_EXASCALER_OLD_SUPPORTED = """\
=====================
DRIVER CONFIGURATION:
=====================
  DDN EXAScaler      : Supported
=====================
"""


GDSCHECK_BEEGFS_ONLY = """\
=====================
DRIVER CONFIGURATION:
=====================
  BeeGFS             : nvfs, compat
=====================
"""


class GdsReportRenderingTests(unittest.TestCase):
    def _enrich(self, reports, output, fs_type, parse_fs_type=None, nvme_backed=True,
                unsupported_raid_level=None):
        gds_report._enrich_with_gdscheck(
            reports, output, fs_type, parse_fs_type or fs_type, nvme_backed, unsupported_raid_level
        )
        return {r.mode: r for r in reports}

    @staticmethod
    def _gdscheck_results(report):
        return [r for r in report.results if r.check.startswith("gdscheck")]

    def test_normal_mount_report_shows_info_findings(self):
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="PCIe topology",
                        mode=GDSMode.P2PDMA,
                        status=Status.INFO,
                        why="GDS can still operate across root ports.",
                    )
                ],
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: None
            text = gds_report.render_text_report("/", "ext4", reports)
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("P2P DMA/C2C (NVMe", text)
        self.assertIn("INFO", text)
        self.assertIn("PCIe topology", text)
        self.assertNotIn("No warnings or errors.", text)

    def test_inactive_p2pdma_with_active_nvfs_reports_alternate_route(self):
        reports = [
            ModeReport(mode=GDSMode.NATIVE, applicable=True),
            ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[CheckResult(
                check="PCIe topology",
                mode=GDSMode.P2PDMA,
                status=Status.INFO,
                why="GDS can still operate across root ports.",
            )]),
        ]

        by_mode = self._enrich(reports, GDSCHECK_NVME_NVFS_ONLY, "ext4")

        whys = [r.why for r in self._gdscheck_results(by_mode[GDSMode.P2PDMA])]
        self.assertTrue(any("Direct GDS is still available via nvidia-fs/nvfs" in w for w in whys))

    def test_inactive_nvfs_on_nvme_reports_doca_mitigation(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="nvidia-fs module",
                        mode=GDSMode.NATIVE,
                        status=Status.WARN,
                        why="gdscheck reports NVMe: compat.",
                    )
                ],
            )
        ]

        by_mode = self._enrich(reports, GDSCHECK_NVME_COMPAT_ONLY, "xfs")

        results = self._gdscheck_results(by_mode[GDSMode.NATIVE])
        self.assertEqual(len(results), 1)
        self.assertIn("gdscheck reports NVMe: compat", results[0].why)
        self.assertIn("GDS storage-stack patches from MLNX_OFED or DOCA", results[0].mitigation)
        self.assertIn("doca-host-installation-and-upgrade", results[0].mitigation)

    def test_gdscheck_acs_blocker_is_suppressed_when_static_acs_passes(self):
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="PCIe ACS redirect",
                        mode=GDSMode.P2PDMA,
                        status=Status.PASS,
                        why="No PCIe bridges have ACS P2P Request Redirect enabled.",
                    ),
                    CheckResult(
                        check="cufile.json P2PDMA settings",
                        mode=GDSMode.P2PDMA,
                        status=Status.FAIL,
                        why="P2PDMA/C2C is disabled in /etc/cufile.json.",
                    ),
                ],
            )
        ]

        by_mode = self._enrich(reports, GDSCHECK_NVME_COMPAT_WITH_P2P_CONFIG_AND_ACS, "xfs")

        whys = "\n".join(r.why for r in self._gdscheck_results(by_mode[GDSMode.P2PDMA]))
        self.assertNotIn("Found ACS enabled", whys)

    def test_gdscheck_explainer_leaves_cufile_and_raid_to_static_checks(self):
        # These conditions are reported by static checks (cufile_config and the
        # RAID checks) from the same effective configuration; gdscheck must not
        # repeat them. Only PLATFORM INFO hardware blockers come from gdscheck.
        output = (
            GDSCHECK_NVME_COMPAT_WITH_P2P_CONFIG_AND_ACS
            + "CUFILE CONFIGURATION:\n=====================\n"
            + "  block.raid.use_pci_p2pdma : false\n"
            + "  fs.virtiofs.use_pci_p2pdma : false\n=====================\n"
        )
        with mock.patch.object(gds_report.iommu, "is_grace", return_value=False), \
                mock.patch.object(gds_report.kernel, "_kernel_version", return_value=(6, 8, 0)):
            whys = [why for why, _ in gds_report._gdscheck_p2pdma_blockers(output)]

        self.assertEqual(whys, ["Found ACS enabled for switch 0000:40:01.1"])

    def test_c2c_token_marks_p2pdma_active(self):
        self.assertIs(
            gds_report._gdscheck_parse(GDSCHECK_NVME_C2C, "ext4", True)[GDSMode.P2PDMA], True
        )
        reports = [ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[CheckResult(
                check="PCIe topology",
                mode=GDSMode.P2PDMA,
                status=Status.INFO,
                why="GDS can still operate across root ports.",
            )])]

        by_mode = self._enrich(reports, GDSCHECK_NVME_C2C, "ext4")

        self.assertEqual(self._gdscheck_results(by_mode[GDSMode.P2PDMA]), [])

    def test_gpfs_p2pdma_token_does_not_suppress_native_warning(self):
        result = gds_report._gdscheck_native_inactive_result(
            GDSCHECK_GPFS_MISLEADING_P2PDMA,
            "gpfs",
            False,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.status, Status.WARN)
        self.assertIn("no native GDS token is active", result.why)
        self.assertNotIn("direct P2PDMA/C2C is active", result.why)

    def test_native_warn_does_not_exempt_p2pdma_failure(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="gdscheck nvidia-fs/nvfs route",
                        mode=GDSMode.NATIVE,
                        status=Status.WARN,
                        why="gdscheck reports NVMe: compat; no nvfs/native token is active.",
                    )
                ],
            ),
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="cufile.json P2PDMA settings",
                        mode=GDSMode.P2PDMA,
                        status=Status.FAIL,
                        why="P2PDMA/C2C is disabled in cufile.json.",
                    )
                ],
            ),
        ]

        self.assertTrue(gds_report.has_blocking_failures(reports, "ext4"))

    def test_active_p2pdma_signal_exempts_native_failure(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="gdscheck nvidia-fs/nvfs route",
                        mode=GDSMode.NATIVE,
                        status=Status.INFO,
                        why=(
                            "gdscheck reports NVMe: c2c, compat; nvidia-fs/nvfs is not "
                            "the active route because direct P2PDMA/C2C is active."
                        ),
                    ),
                    CheckResult(
                        check="nvidia-fs module",
                        mode=GDSMode.NATIVE,
                        status=Status.FAIL,
                        why="nvidia_fs is not loaded.",
                    ),
                ],
            ),
            ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[]),
        ]

        self.assertFalse(gds_report.has_blocking_failures(reports, "ext4"))

    def test_active_p2pdma_signal_does_not_exempt_native_mount_failures(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="gdscheck nvidia-fs/nvfs route",
                        mode=GDSMode.NATIVE,
                        status=Status.INFO,
                        why=(
                            "gdscheck reports NVMe: c2c, compat; nvidia-fs/nvfs is not "
                            "the active route because direct P2PDMA/C2C is active."
                        ),
                    ),
                    CheckResult(
                        check="ext4 data mode",
                        mode=GDSMode.NATIVE,
                        status=Status.FAIL,
                        why="ext4 is not explicitly mounted with data=ordered.",
                    ),
                    CheckResult(
                        check="O_DIRECT support",
                        mode=GDSMode.NATIVE,
                        status=Status.FAIL,
                        why="O_DIRECT is NOT supported on this filesystem mount.",
                    ),
                ],
            ),
            ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[]),
        ]

        self.assertTrue(gds_report.has_blocking_failures(reports, "ext4"))

    def test_raid0_p2pdma_token_is_not_supported_on_non_grace(self):
        with mock.patch.object(gds_report.iommu, "is_grace", return_value=False), \
                mock.patch.object(gds_report.kernel, "_kernel_version", return_value=(7, 0, 0)):
            verdict = gds_report._gdscheck_parse(GDSCHECK_NVME_C2C, "raid0", True)

        self.assertIs(verdict[GDSMode.P2PDMA], False)

    def test_raid0_p2pdma_token_is_supported_on_non_grace_kernel_7_1(self):
        with mock.patch.object(gds_report.iommu, "is_grace", return_value=False), \
                mock.patch.object(gds_report.kernel, "_kernel_version", return_value=(7, 1, 0)):
            verdict = gds_report._gdscheck_parse(GDSCHECK_NVME_C2C, "raid0", True)

        self.assertIs(verdict[GDSMode.P2PDMA], True)

    def test_raid0_p2pdma_token_is_supported_on_grace(self):
        with mock.patch.object(gds_report.iommu, "is_grace", return_value=True), \
                mock.patch.object(gds_report.kernel, "_kernel_version", return_value=(6, 8, 0)):
            verdict = gds_report._gdscheck_parse(GDSCHECK_NVME_C2C, "raid0", True)

        self.assertIs(verdict[GDSMode.P2PDMA], True)

    def test_non_raid0_nvme_token_does_not_override_unsupported_raid_level(self):
        def raid5_fail(mode):
            return CheckResult(
                check="RAID level GDS support",
                mode=mode,
                status=Status.FAIL,
                why="RAID5 is not a supported direct GDS RAID route.",
                mitigation="Use RAID0 or compat mode.",
            )

        reports = [
            ModeReport(mode=GDSMode.NATIVE, applicable=True, results=[raid5_fail(GDSMode.NATIVE)]),
            ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[raid5_fail(GDSMode.P2PDMA)]),
        ]

        by_mode = self._enrich(
            reports, GDSCHECK_NVME_C2C, "ext4", parse_fs_type="raid5", unsupported_raid_level="raid5"
        )

        self.assertEqual(by_mode[GDSMode.NATIVE].status, Status.FAIL)
        self.assertEqual(by_mode[GDSMode.P2PDMA].status, Status.FAIL)
        for report in by_mode.values():
            for r in report.results:
                self.assertNotIn("Direct GDS is still available via nvidia-fs/nvfs", r.why)

    def test_device_mapper_failure_suppresses_nvfs_available_info(self):
        # gdscheck sees native nvfs as active for ext4/NVMe, but the static
        # device-mapper FAIL in the Native report means nvfs is not actually
        # available for this mount, so the INFO must not claim it is.
        def dm_fail(mode):
            return CheckResult(
                check="Device-mapper backing device",
                mode=mode,
                status=Status.FAIL,
                why="Backed by device-mapper (LVM).",
                mitigation="Remove the device-mapper layer.",
            )

        reports = [
            ModeReport(mode=GDSMode.NATIVE, applicable=True, results=[dm_fail(GDSMode.NATIVE)]),
            ModeReport(mode=GDSMode.P2PDMA, applicable=True, results=[dm_fail(GDSMode.P2PDMA)]),
        ]

        by_mode = self._enrich(reports, GDSCHECK_NVME_NVFS_ONLY, "ext4")

        for r in self._gdscheck_results(by_mode[GDSMode.P2PDMA]):
            self.assertNotIn("Direct GDS is still available via nvidia-fs/nvfs", r.why)

    def test_lustre_accepts_ddn_exascaler_gdscheck_key(self):
        reports = [ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="nvidia-fs module",
                        mode=GDSMode.NATIVE,
                        status=Status.PASS,
                        why="nvidia_fs loaded.",
                    )
                ],
            )]

        by_mode = self._enrich(reports, GDSCHECK_DDN_EXASCALER, "lustre", nvme_backed=False)

        self.assertIs(
            gds_report._gdscheck_parse(GDSCHECK_DDN_EXASCALER, "lustre", False)[GDSMode.NATIVE], True
        )
        self.assertEqual(self._gdscheck_results(by_mode[GDSMode.NATIVE]), [])

    def test_old_supported_status_marks_lustre_native_active(self):
        verdict = gds_report._gdscheck_parse(GDSCHECK_DDN_EXASCALER_OLD_SUPPORTED, "lustre", False)

        self.assertIs(verdict[GDSMode.NATIVE], True)

    def test_lustre_cannot_verify_mentions_lustre_and_ddn_exascaler(self):
        reports = [ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="nvidia-fs module",
                        mode=GDSMode.NATIVE,
                        status=Status.PASS,
                        why="nvidia_fs loaded.",
                    )
                ],
            )]

        by_mode = self._enrich(reports, GDSCHECK_BEEGFS_ONLY, "lustre", nvme_backed=False)

        results = self._gdscheck_results(by_mode[GDSMode.NATIVE])
        self.assertEqual([r.check for r in results], ["gdscheck client detection"])
        self.assertIn("No active Lustre or DDN EXAScaler client found by gdscheck", results[0].why)

    def test_nfs_p2pdma_is_not_applicable_and_rdma_is_checked(self):
        # NFS is Native-applicable (nvidia-fs/nvfs has to be loaded to activate
        # NFSoRDMA, even though the direct-path mechanism itself is NFSoRDMA,
        # not the NVMe nvfs route) — so shared Native infra checks (kernel,
        # iommu, nvidia_fs.run_all) legitimately run here. P2PDMA remains
        # inapplicable for NFS, so its P2PDMA-only checks must still not fire.
        old_kernel = gds_report.kernel.run_all
        old_iommu = gds_report.iommu.run_all
        old_nvidia_fs_run_all = gds_report.nvidia_fs.run_all
        old_pcie_run_all = gds_report.pcie.run_all
        old_check_acs = gds_report.pcie.check_acs
        old_open_driver = gds_report.nvidia_fs.check_open_driver
        old_p2pdma_registries = gds_report.nvidia_fs.check_p2pdma_driver_registries
        old_cufile_run_all = gds_report.cufile_config.run_all
        old_compat = gds_report.cufile_config.run_compat_checks
        old_rdma = gds_report.rdma.run_all
        old_nfs_rdma = gds_report._check_nfs_rdma_mount
        old_run_gdscheck = gds_report._run_gdscheck_raw
        old_check_odirect = gds_report.check_odirect
        try:
            gds_report.kernel.run_all = lambda mode: []
            gds_report.iommu.run_all = lambda mode: []
            gds_report.nvidia_fs.run_all = lambda: []
            gds_report.pcie.run_all = lambda: self.fail("P2PDMA-only local NVMe topology should not run for NFS")
            gds_report.pcie.check_acs = lambda: self.fail("P2PDMA-only ACS checks should not run for NFS")
            gds_report.nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver",
                mode=mode,
                status=Status.PASS,
                why="Open driver.",
            )
            gds_report.nvidia_fs.check_p2pdma_driver_registries = lambda: self.fail("P2PDMA registry checks should not run for NFS")
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: []
            gds_report.cufile_config.run_compat_checks = lambda: []
            gds_report.rdma.run_all = lambda fs_type: []
            gds_report._check_nfs_rdma_mount = lambda path: CheckResult(
                check="NFS rdma mount option",
                mode=GDSMode.RDMA,
                status=Status.FAIL,
                why="NFS is not mounted with rdma.",
            )
            gds_report._run_gdscheck_raw = lambda: ""
            gds_report.check_odirect = lambda path: (True, "mocked")

            reports = gds_report.build_mode_reports("/home", "nfs")
        finally:
            gds_report.kernel.run_all = old_kernel
            gds_report.iommu.run_all = old_iommu
            gds_report.nvidia_fs.run_all = old_nvidia_fs_run_all
            gds_report.pcie.run_all = old_pcie_run_all
            gds_report.pcie.check_acs = old_check_acs
            gds_report.nvidia_fs.check_open_driver = old_open_driver
            gds_report.nvidia_fs.check_p2pdma_driver_registries = old_p2pdma_registries
            gds_report.cufile_config.run_all = old_cufile_run_all
            gds_report.cufile_config.run_compat_checks = old_compat
            gds_report.rdma.run_all = old_rdma
            gds_report._check_nfs_rdma_mount = old_nfs_rdma
            gds_report._run_gdscheck_raw = old_run_gdscheck
            gds_report.check_odirect = old_check_odirect

        native = next(report for report in reports if report.mode == GDSMode.NATIVE)
        self.assertTrue(native.applicable)

        p2pdma = next(report for report in reports if report.mode == GDSMode.P2PDMA)
        self.assertFalse(p2pdma.applicable)
        self.assertEqual([result.check for result in p2pdma.results], ["Filesystem P2PDMA support"])

        rdma = next(report for report in reports if report.mode == GDSMode.RDMA)
        self.assertIn("NFS rdma mount option", [result.check for result in rdma.results])

    def test_nfs_rdma_mount_accepts_proto_rdma(self):
        mounts = (
            "tmpfs /mnt tmpfs rw,relatime 0 0\n"
            "192.168.4.11:/ /mnt/vast nfs "
            "rw,relatime,vers=3,rsize=1048576,wsize=1048576,proto=rdma,"
            "nconnect=16,port=20049,localports=192.168.4.210,"
            "remoteports=192.168.4.11-192.168.4.26 0 0\n"
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=mounts)):
            result = gds_report._check_nfs_rdma_mount("/mnt/vast/file")

        self.assertEqual(result.status, Status.PASS)
        self.assertIn("RDMA transport", result.why)

    def test_nfs_rdma_mount_rejects_tcp_proto(self):
        mounts = (
            "server:/ /mnt/nfs nfs "
            "rw,relatime,vers=3,rsize=1048576,wsize=1048576,proto=tcp,port=2049 0 0\n"
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=mounts)):
            result = gds_report._check_nfs_rdma_mount("/mnt/nfs/file")

        self.assertEqual(result.status, Status.FAIL)
        self.assertIn("proto=rdma", result.mitigation)

    def test_nfs_rdma_mount_blank_path_is_unverifiable(self):
        result = gds_report._check_nfs_rdma_mount(" ")

        self.assertEqual(result.status, Status.WARN)
        self.assertIn("Path was not provided", result.why)

    def test_ext4_on_raid0_uses_raid_p2pdma_config_key(self):
        calls = []
        old_get_nvme_transport = gds_report.get_nvme_transport
        old_get_raid_level = gds_report.get_raid_level
        old_kernel = gds_report.kernel.run_all
        old_iommu = gds_report.iommu.run_all
        old_pcie_run_all = gds_report.pcie.run_all
        old_open_driver = gds_report.nvidia_fs.check_open_driver
        old_p2pdma_registries = gds_report.nvidia_fs.check_p2pdma_driver_registries
        old_nvidia_run_all = gds_report.nvidia_fs.run_all
        old_cufile_run_all = gds_report.cufile_config.run_all
        old_compat = gds_report.cufile_config.run_compat_checks
        old_odirect = gds_report.check_odirect
        old_ext4_data = gds_report.check_ext4_data_mode
        old_run_gdscheck = gds_report._run_gdscheck_raw
        old_nvme_backed = gds_report.is_nvme_backed
        old_grace = gds_report.iommu.is_grace
        try:
            gds_report.get_nvme_transport = lambda path: None
            gds_report.get_raid_level = lambda path: "raid0"
            gds_report.kernel.run_all = lambda mode: []
            gds_report.iommu.run_all = lambda mode: []
            gds_report.pcie.run_all = lambda: []
            gds_report.nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver",
                mode=mode,
                status=Status.PASS,
                why="Open driver.",
            )
            gds_report.nvidia_fs.check_p2pdma_driver_registries = lambda: None
            gds_report.nvidia_fs.run_all = lambda: []
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: (
                calls.append((fs_type, p2pdma_block_key)) or []
            )
            gds_report.cufile_config.run_compat_checks = lambda: []
            gds_report.check_odirect = lambda path: (True, "ok")
            gds_report.check_ext4_data_mode = lambda path: None
            gds_report._run_gdscheck_raw = lambda: ""
            gds_report.is_nvme_backed = lambda path: True
            gds_report.iommu.is_grace = lambda: False

            gds_report.build_mode_reports("/raid", "ext4")
        finally:
            gds_report.get_nvme_transport = old_get_nvme_transport
            gds_report.get_raid_level = old_get_raid_level
            gds_report.kernel.run_all = old_kernel
            gds_report.iommu.run_all = old_iommu
            gds_report.pcie.run_all = old_pcie_run_all
            gds_report.nvidia_fs.check_open_driver = old_open_driver
            gds_report.nvidia_fs.check_p2pdma_driver_registries = old_p2pdma_registries
            gds_report.nvidia_fs.run_all = old_nvidia_run_all
            gds_report.cufile_config.run_all = old_cufile_run_all
            gds_report.cufile_config.run_compat_checks = old_compat
            gds_report.check_odirect = old_odirect
            gds_report.check_ext4_data_mode = old_ext4_data
            gds_report._run_gdscheck_raw = old_run_gdscheck
            gds_report.is_nvme_backed = old_nvme_backed
            gds_report.iommu.is_grace = old_grace

        self.assertEqual(calls, [("ext4", "raid")])

    def test_ext4_implicit_data_ordered_blocks_native_gds_for_root(self):
        old_kernel = gds_report.kernel.run_all
        old_iommu = gds_report.iommu.run_all
        old_nvidia_run_all = gds_report.nvidia_fs.run_all
        old_open_driver = gds_report.nvidia_fs.check_open_driver
        old_cufile_run_all = gds_report.cufile_config.run_all
        old_compat = gds_report.cufile_config.run_compat_checks
        old_odirect = gds_report.check_odirect
        old_ext4_data = gds_report.check_ext4_data_mode
        old_run_gdscheck = gds_report._run_gdscheck_raw
        old_transport = gds_report.get_nvme_transport
        old_raid = gds_report.get_raid_level
        try:
            gds_report.kernel.run_all = lambda mode: []
            gds_report.iommu.run_all = lambda mode: []
            gds_report.nvidia_fs.run_all = lambda: []
            gds_report.nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver",
                mode=mode,
                status=Status.PASS,
                why="Open driver.",
            )
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: []
            gds_report.cufile_config.run_compat_checks = lambda: []
            gds_report.check_odirect = lambda path: (True, "O_DIRECT ok")
            gds_report.check_ext4_data_mode = lambda path: (
                "default",
                "/ opts: rw,relatime (implicit ext4 default)",
            )
            gds_report._run_gdscheck_raw = lambda: ""
            gds_report.get_nvme_transport = lambda path: None
            gds_report.get_raid_level = lambda path: None

            reports = gds_report.build_mode_reports("/", "ext4")
        finally:
            gds_report.kernel.run_all = old_kernel
            gds_report.iommu.run_all = old_iommu
            gds_report.nvidia_fs.run_all = old_nvidia_run_all
            gds_report.nvidia_fs.check_open_driver = old_open_driver
            gds_report.cufile_config.run_all = old_cufile_run_all
            gds_report.cufile_config.run_compat_checks = old_compat
            gds_report.check_odirect = old_odirect
            gds_report.check_ext4_data_mode = old_ext4_data
            gds_report._run_gdscheck_raw = old_run_gdscheck
            gds_report.get_nvme_transport = old_transport
            gds_report.get_raid_level = old_raid

        native = next(report for report in reports if report.mode == GDSMode.NATIVE)
        ext4_result = next(result for result in native.results if result.check == "ext4 data mode")
        self.assertEqual(ext4_result.status, Status.FAIL)
        self.assertIn("not explicitly mounted with data=ordered", ext4_result.why)
        self.assertIn("rootflags=data=ordered", ext4_result.mitigation)

    def test_local_path_failure_overrides_gdscheck_native_token(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="ext4 data mode",
                        mode=GDSMode.NATIVE,
                        status=Status.FAIL,
                        why="ext4 is not explicitly mounted with data=ordered.",
                        mitigation="Add rootflags=data=ordered.",
                    )
                ],
            )
        ]

        by_mode = self._enrich(reports, GDSCHECK_NVME_NVFS_ONLY, "ext4")

        native = by_mode[GDSMode.NATIVE]
        self.assertEqual(native.status, Status.FAIL)
        self.assertIn("rootflags=data=ordered", native.blockers()[0].mitigation)

    def test_ext4_explicit_data_ordered_passes_native_gds_check(self):
        old_kernel = gds_report.kernel.run_all
        old_iommu = gds_report.iommu.run_all
        old_nvidia_run_all = gds_report.nvidia_fs.run_all
        old_open_driver = gds_report.nvidia_fs.check_open_driver
        old_cufile_run_all = gds_report.cufile_config.run_all
        old_compat = gds_report.cufile_config.run_compat_checks
        old_odirect = gds_report.check_odirect
        old_ext4_data = gds_report.check_ext4_data_mode
        old_run_gdscheck = gds_report._run_gdscheck_raw
        old_transport = gds_report.get_nvme_transport
        old_raid = gds_report.get_raid_level
        try:
            gds_report.kernel.run_all = lambda mode: []
            gds_report.iommu.run_all = lambda mode: []
            gds_report.nvidia_fs.run_all = lambda: []
            gds_report.nvidia_fs.check_open_driver = lambda raw, mode: CheckResult(
                check="NVIDIA Open Driver",
                mode=mode,
                status=Status.PASS,
                why="Open driver.",
            )
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: []
            gds_report.cufile_config.run_compat_checks = lambda: []
            gds_report.check_odirect = lambda path: (True, "O_DIRECT ok")
            gds_report.check_ext4_data_mode = lambda path: (
                "ordered",
                "/ opts: rw,relatime,data=ordered",
            )
            gds_report._run_gdscheck_raw = lambda: ""
            gds_report.get_nvme_transport = lambda path: None
            gds_report.get_raid_level = lambda path: None

            reports = gds_report.build_mode_reports("/", "ext4")
        finally:
            gds_report.kernel.run_all = old_kernel
            gds_report.iommu.run_all = old_iommu
            gds_report.nvidia_fs.run_all = old_nvidia_run_all
            gds_report.nvidia_fs.check_open_driver = old_open_driver
            gds_report.cufile_config.run_all = old_cufile_run_all
            gds_report.cufile_config.run_compat_checks = old_compat
            gds_report.check_odirect = old_odirect
            gds_report.check_ext4_data_mode = old_ext4_data
            gds_report._run_gdscheck_raw = old_run_gdscheck
            gds_report.get_nvme_transport = old_transport
            gds_report.get_raid_level = old_raid

        native = next(report for report in reports if report.mode == GDSMode.NATIVE)
        ext4_result = next(result for result in native.results if result.check == "ext4 data mode")
        self.assertEqual(ext4_result.status, Status.PASS)
        self.assertIn("explicitly mounted with data=ordered", ext4_result.why)

    def test_pcie_topology_pair_evidence_is_verbose_only(self):
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="PCIe topology",
                        mode=GDSMode.P2PDMA,
                        status=Status.INFO,
                        why=(
                            "2 of 2 visible GPU↔NVMe pair(s) are on different PCIe root "
                            "complexes. Rerun mount-check with --verbose to view the affected pair list."
                        ),
                        evidence=(
                            "GPU 0000:65:00.0 ↔ NVMe 0000:ca:00.0\n"
                            "GPU 0000:66:00.0 ↔ NVMe 0000:cb:00.0"
                        ),
                    )
                ],
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: None
            normal = gds_report.render_text_report("/", "ext4", reports, verbose=False)
            verbose = gds_report.render_text_report("/", "ext4", reports, verbose=True)
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("--verbose", normal)
        self.assertNotIn("GPU 0000:65:00.0 ↔ NVMe 0000:ca:00.0", normal)
        self.assertIn("GPU 0000:65:00.0 ↔ NVMe 0000:ca:00.0", verbose)
        self.assertIn("GPU 0000:66:00.0 ↔ NVMe 0000:cb:00.0", verbose)


if __name__ == "__main__":
    unittest.main()
