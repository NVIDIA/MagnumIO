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


GDSCHECK_VIRTIOFS_FS_KEY_ONLY = """\
=====================
DRIVER CONFIGURATION:
=====================
  VIRTIOFS           : compat
=====================
CUFILE CONFIGURATION:
=====================
  properties.use_pci_p2pdma : false
  fs.virtiofs.use_pci_p2pdma : true
=====================
"""


GDSCHECK_VIRTIOFS_BOTH_KEYS = """\
=====================
DRIVER CONFIGURATION:
=====================
  VIRTIOFS           : p2pdma, compat
=====================
CUFILE CONFIGURATION:
=====================
  properties.use_pci_p2pdma : true
  fs.virtiofs.use_pci_p2pdma : true
=====================
"""


class GdscheckP2pdmaBlockersVirtiofsTests(unittest.TestCase):
    """
    Regression coverage for a real gap found while validating gds-diag against
    a live virtiofs P2PDMA test VM: gdscheck only reports the VIRTIOFS route
    as active when *both* properties.use_pci_p2pdma and
    fs.virtiofs.use_pci_p2pdma are true, but the checker used to only look at
    the fs-specific key — so it would report "config OK" even though the
    route was not actually active.
    """

    def test_fs_key_alone_is_reported_as_a_blocker(self):
        blockers = gds_report._gdscheck_p2pdma_blockers(
            GDSCHECK_VIRTIOFS_FS_KEY_ONLY, fs_type="virtiofs"
        )

        why = "\n".join(b[0] for b in blockers)
        self.assertIn("properties.use_pci_p2pdma = false", why)

    def test_both_keys_true_reports_no_config_blocker(self):
        blockers = gds_report._gdscheck_p2pdma_blockers(
            GDSCHECK_VIRTIOFS_BOTH_KEYS, fs_type="virtiofs"
        )

        why = "\n".join(b[0] for b in blockers)
        self.assertNotIn("use_pci_p2pdma", why)


class GdsReportRenderingTests(unittest.TestCase):
    def _mode_support_line(self, text: str, mode: GDSMode) -> str:
        for line in text.splitlines():
            if mode.value in line:
                return line
        self.fail(f"missing mode support line for {mode.value!r}")

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

    def test_applicable_inactive_p2pdma_renders_not_active(self):
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
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_NVFS_ONLY
            text = gds_report._build_mode_support(reports, "ext4", "")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("Not active", text)
        self.assertNotIn("Not supported", text)

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

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_COMPAT_ONLY
            text = gds_report._build_mode_support(reports, "xfs", "")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("Native GDS (nvidia-fs)", text)
        self.assertIn("Not active", self._mode_support_line(text, GDSMode.NATIVE))
        self.assertIn("gdscheck reports NVMe: compat", text)
        self.assertIn("GDS storage-stack patches from MLNX_OFED or DOCA", text)
        self.assertIn("doca-host-installation-and-upgrade", text)

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

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_COMPAT_WITH_P2P_CONFIG_AND_ACS
            text = gds_report._build_mode_support(reports, "xfs", "")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("properties.use_pci_p2pdma = false", text)
        self.assertNotIn("Found ACS enabled", text)

    def test_c2c_token_renders_p2pdma_supported(self):
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
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_C2C
            text = gds_report._build_mode_support(reports, "ext4", "")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertNotIn("Not active", text)

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
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="RAID0 P2PDMA architecture",
                        mode=GDSMode.P2PDMA,
                        status=Status.FAIL,
                        why="RAID0 P2PDMA requires NVIDIA Grace or Linux kernel >= 7.1.",
                        mitigation="Use nvidia-fs/nvfs or compat mode.",
                    )
                ],
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        old_raid = gds_report.get_raid_level
        old_transport = gds_report.get_nvme_transport
        old_grace = gds_report.iommu.is_grace
        old_kernel = gds_report.kernel._kernel_version
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_C2C
            gds_report.get_raid_level = lambda path: "raid0"
            gds_report.get_nvme_transport = lambda path: "pcie"
            gds_report.iommu.is_grace = lambda: False
            gds_report.kernel._kernel_version = lambda: (7, 0, 0)
            text = gds_report._build_mode_support(reports, "ext4", "/raid")
        finally:
            gds_report._run_gdscheck_raw = old_run
            gds_report.get_raid_level = old_raid
            gds_report.get_nvme_transport = old_transport
            gds_report.iommu.is_grace = old_grace
            gds_report.kernel._kernel_version = old_kernel

        self.assertIn("Not active", text)
        self.assertIn("RAID0 P2PDMA requires NVIDIA Grace or Linux kernel >= 7.1", text)
        p2pdma_line = self._mode_support_line(text, GDSMode.P2PDMA)
        self.assertIn("Not active", p2pdma_line)
        self.assertNotIn("Supported", p2pdma_line)

    def test_raid0_p2pdma_token_is_supported_on_non_grace_kernel_7_1(self):
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="RAID0 P2PDMA architecture",
                        mode=GDSMode.P2PDMA,
                        status=Status.PASS,
                        why="Linux kernel 7.1.0 detected.",
                    )
                ],
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        old_raid = gds_report.get_raid_level
        old_transport = gds_report.get_nvme_transport
        old_grace = gds_report.iommu.is_grace
        old_kernel = gds_report.kernel._kernel_version
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_C2C
            gds_report.get_raid_level = lambda path: "raid0"
            gds_report.get_nvme_transport = lambda path: "pcie"
            gds_report.iommu.is_grace = lambda: False
            gds_report.kernel._kernel_version = lambda: (7, 1, 0)
            text = gds_report._build_mode_support(reports, "ext4", "/raid")
        finally:
            gds_report._run_gdscheck_raw = old_run
            gds_report.get_raid_level = old_raid
            gds_report.get_nvme_transport = old_transport
            gds_report.iommu.is_grace = old_grace
            gds_report.kernel._kernel_version = old_kernel

        self.assertNotIn("Not active", text)
        self.assertNotIn("Grace-based only", text)

    def test_raid0_p2pdma_token_is_supported_on_grace(self):
        reports = [
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="RAID0 P2PDMA architecture",
                        mode=GDSMode.P2PDMA,
                        status=Status.PASS,
                        why="Grace platform detected.",
                    )
                ],
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        old_raid = gds_report.get_raid_level
        old_transport = gds_report.get_nvme_transport
        old_grace = gds_report.iommu.is_grace
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_C2C
            gds_report.get_raid_level = lambda path: "raid0"
            gds_report.get_nvme_transport = lambda path: "pcie"
            gds_report.iommu.is_grace = lambda: True
            text = gds_report._build_mode_support(reports, "ext4", "/raid")
        finally:
            gds_report._run_gdscheck_raw = old_run
            gds_report.get_raid_level = old_raid
            gds_report.get_nvme_transport = old_transport
            gds_report.iommu.is_grace = old_grace

        self.assertNotIn("Grace-based only", text)

    def test_non_raid0_nvme_token_does_not_override_unsupported_raid_level(self):
        reports = [
            ModeReport(
                mode=GDSMode.NATIVE,
                applicable=True,
                results=[
                    CheckResult(
                        check="RAID level GDS support",
                        mode=GDSMode.NATIVE,
                        status=Status.FAIL,
                        why="RAID5 is not a supported direct GDS RAID route.",
                        mitigation="Use RAID0 or compat mode.",
                    )
                ],
            ),
            ModeReport(
                mode=GDSMode.P2PDMA,
                applicable=True,
                results=[
                    CheckResult(
                        check="RAID level GDS support",
                        mode=GDSMode.P2PDMA,
                        status=Status.FAIL,
                        why="RAID5 is not a supported direct GDS RAID route.",
                        mitigation="Use RAID0 or compat mode.",
                    )
                ],
            ),
        ]

        old_run = gds_report._run_gdscheck_raw
        old_raid = gds_report.get_raid_level
        old_transport = gds_report.get_nvme_transport
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_C2C
            gds_report.get_raid_level = lambda path: "raid5"
            gds_report.get_nvme_transport = lambda path: "pcie"
            text = gds_report._build_mode_support(reports, "ext4", "/raid")
        finally:
            gds_report._run_gdscheck_raw = old_run
            gds_report.get_raid_level = old_raid
            gds_report.get_nvme_transport = old_transport

        self.assertIn("Not active", self._mode_support_line(text, GDSMode.NATIVE))
        self.assertIn("Not active", self._mode_support_line(text, GDSMode.P2PDMA))
        self.assertIn("RAID5 is not a supported direct GDS RAID route", text)
        self.assertNotIn("Direct GDS is still available via nvidia-fs/nvfs", text)

    def test_lustre_accepts_ddn_exascaler_gdscheck_key(self):
        reports = [
            ModeReport(
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
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_DDN_EXASCALER
            text = gds_report._build_mode_support(reports, "lustre", "/lustre")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertNotIn("No active Lustre or DDN EXAScaler client found by gdscheck", text)

    def test_old_supported_status_renders_lustre_native_supported(self):
        reports = [
            ModeReport(
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
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_DDN_EXASCALER_OLD_SUPPORTED
            text = gds_report._build_mode_support(reports, "lustre", "/lustre")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertNotIn("Not active", text)

    def test_lustre_cannot_verify_mentions_lustre_and_ddn_exascaler(self):
        reports = [
            ModeReport(
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
            )
        ]

        old_run = gds_report._run_gdscheck_raw
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_BEEGFS_ONLY
            text = gds_report._build_mode_support(reports, "lustre", "/lustre")
        finally:
            gds_report._run_gdscheck_raw = old_run

        self.assertIn("Cannot verify", text)
        self.assertIn("No active Lustre or DDN EXAScaler client found by gdscheck", text)

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
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None: []
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
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None: (
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
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None: []
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
        old_run = gds_report._run_gdscheck_raw
        old_nvme = gds_report.is_nvme_backed
        old_transport = gds_report.get_nvme_transport
        old_raid = gds_report.get_raid_level
        try:
            gds_report._run_gdscheck_raw = lambda: GDSCHECK_NVME_NVFS_ONLY
            gds_report.is_nvme_backed = lambda path: True
            gds_report.get_nvme_transport = lambda path: "pcie"
            gds_report.get_raid_level = lambda path: None

            text = gds_report._build_mode_support(reports, "ext4", "/")
        finally:
            gds_report._run_gdscheck_raw = old_run
            gds_report.is_nvme_backed = old_nvme
            gds_report.get_nvme_transport = old_transport
            gds_report.get_raid_level = old_raid

        self.assertIn("Native GDS", text)
        self.assertIn("Not active", text)
        self.assertIn("rootflags=data=ordered", text)
        self.assertNotIn("Supported", self._mode_support_line(text, GDSMode.NATIVE))

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
            gds_report.cufile_config.run_all = lambda fs_type, p2pdma_block_key=None: []
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
