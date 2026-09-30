# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-filesystem tests for build_mode_reports().

Each test relies on the gds_infrastructure fixture (conftest.py) to patch
all system-level calls to neutral values.  Tests only assert on the
filesystem-specific logic that changes with fs_type.
"""

import pytest

from checks import cufile_config
from checks.gds_report import build_mode_reports
from checks.result import CheckResult, GDSMode, Status

# Captured before the gds_infrastructure fixture replaces it, for tests that
# need the real cufile.json checks.
_real_cufile_run_all = cufile_config.run_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _by_mode(reports):
    return {r.mode: r for r in reports}


def _check_names(report):
    return [r.check for r in report.results]


def _find(report, check_name):
    return next((r for r in report.results if r.check == check_name), None)


# ---------------------------------------------------------------------------
# Applicability smoke tests — one parametrized test covers every supported FS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fs_type,expect_native,expect_p2pdma,expect_rdma", [
    # Local block-backed
    ("ext4",        True,  True,  False),
    ("xfs",         True,  True,  False),
    ("nvme-of",     True,  True,  True),
    ("raid0",       True,  True,  False),
    # Network / HPC
    ("nfs",         True,  False, True),
    ("nfs4",        True,  False, True),
    ("lustre",      True,  False, True),
    ("gpfs",        True,  False, True),
    ("mmfs",        True,  False, True),
    ("wekafs",      True,  False, True),
    ("beegfs",      True,  False, True),
    ("fhgfs",       True,  False, True),
    # Specialty
    ("virtiofs",    False, True,  False),
    ("scatefs",     True,  False, True),
    ("nvmesh",      True,  False, False),
    ("scsi",        False, False, False),
    ("scaleflux",   True,  False, False),
    # Compat-only
    ("tmpfs",       False, False, False),
    ("ramfs",       False, False, False),
    ("overlay",     False, False, False),
    ("btrfs",       False, False, False),
    ("zfs",         False, False, False),
])
def test_mode_applicability(gds_infrastructure, fs_type, expect_native, expect_p2pdma, expect_rdma):
    reports = build_mode_reports("/tmp", fs_type)
    modes = _by_mode(reports)
    assert modes[GDSMode.NATIVE].applicable  == expect_native,  f"{fs_type}: native"
    assert modes[GDSMode.P2PDMA].applicable == expect_p2pdma, f"{fs_type}: p2pdma"
    assert modes[GDSMode.RDMA].applicable   == expect_rdma,   f"{fs_type}: rdma"


# ---------------------------------------------------------------------------
# ext4
# ---------------------------------------------------------------------------

class TestExt4:
    def test_data_ordered_passes(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr(
            "checks.gds_report.check_ext4_data_mode",
            lambda path: ("ordered", "/mnt opts: rw,data=ordered"),
        )
        reports = build_mode_reports("/mnt", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "ext4 data mode")
        assert result is not None
        assert result.status == Status.PASS

    def test_data_journal_fails(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr(
            "checks.gds_report.check_ext4_data_mode",
            lambda path: ("journal", "/mnt opts: rw,data=journal"),
        )
        reports = build_mode_reports("/mnt", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "ext4 data mode")
        assert result is not None
        assert result.status == Status.FAIL
        assert "data=journal" in result.why

    def test_data_implicit_default_fails(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr(
            "checks.gds_report.check_ext4_data_mode",
            lambda path: ("default", "/mnt opts: rw,relatime (implicit ext4 default)"),
        )
        reports = build_mode_reports("/mnt", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "ext4 data mode")
        assert result is not None
        assert result.status == Status.FAIL

    def test_root_fs_journal_mitigation_mentions_grub(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr(
            "checks.gds_report.check_ext4_data_mode",
            lambda path: ("journal", "/ opts: rw,data=journal"),
        )
        reports = build_mode_reports("/", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "ext4 data mode")
        assert result is not None
        assert "grub" in result.mitigation.lower() or "rootflags" in result.mitigation

    def test_odirect_check_fires(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "ext4")
        assert _find(_by_mode(reports)[GDSMode.NATIVE], "O_DIRECT support") is not None

    def test_p2pdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "ext4")
        assert _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_rdma_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "ext4")
        assert not _by_mode(reports)[GDSMode.RDMA].applicable


# ---------------------------------------------------------------------------
# XFS
# ---------------------------------------------------------------------------

class TestXFS:
    def test_no_ext4_data_mode_check(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "xfs")
        assert _find(_by_mode(reports)[GDSMode.NATIVE], "ext4 data mode") is None

    def test_odirect_check_fires(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "xfs")
        assert _find(_by_mode(reports)[GDSMode.NATIVE], "O_DIRECT support") is not None

    def test_p2pdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "xfs")
        assert _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_rdma_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt", "xfs")
        assert not _by_mode(reports)[GDSMode.RDMA].applicable


# ---------------------------------------------------------------------------
# NFS / NFS4
# ---------------------------------------------------------------------------

class TestNFS:
    @pytest.mark.parametrize("fs_type", ["nfs", "nfs4"])
    def test_native_applicable(self, gds_infrastructure, fs_type):
        # nvidia-fs (nvfs) still has to be loaded to activate NFSoRDMA, so NFS
        # is Native-applicable the same way Lustre is, even though the actual
        # direct-path mechanism (NFSoRDMA) is different from the NVMe nvfs path.
        reports = build_mode_reports("/mnt/nfs", fs_type)
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    @pytest.mark.parametrize("fs_type", ["nfs", "nfs4"])
    def test_p2pdma_not_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/nfs", fs_type)
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    @pytest.mark.parametrize("fs_type", ["nfs", "nfs4"])
    def test_rdma_mount_check_fires(self, gds_infrastructure, monkeypatch, fs_type):
        called = []

        def _mock(path):
            called.append(path)
            return CheckResult(
                check="NFS rdma mount option", mode=GDSMode.RDMA,
                status=Status.PASS, why="mocked",
            )

        monkeypatch.setattr("checks.gds_report._check_nfs_rdma_mount", _mock)
        build_mode_reports("/mnt/nfs", fs_type)
        assert called, f"_check_nfs_rdma_mount was not called for {fs_type}"

    @pytest.mark.parametrize("fs_type", ["nfs", "nfs4"])
    def test_rdma_mount_fail_appears_in_rdma_results(self, gds_infrastructure, monkeypatch, fs_type):
        monkeypatch.setattr(
            "checks.gds_report._check_nfs_rdma_mount",
            lambda path: CheckResult(
                check="NFS rdma mount option", mode=GDSMode.RDMA,
                status=Status.FAIL, why="proto=tcp, not rdma",
                mitigation="remount with proto=rdma,port=20049",
            ),
        )
        reports = build_mode_reports("/mnt/nfs", fs_type)
        rdma = _by_mode(reports)[GDSMode.RDMA]
        assert rdma.applicable
        result = _find(rdma, "NFS rdma mount option")
        assert result is not None
        assert result.status == Status.FAIL


# ---------------------------------------------------------------------------
# Lustre
# ---------------------------------------------------------------------------

class TestLustre:
    def test_native_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/lustre", "lustre")
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_rdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/lustre", "lustre")
        assert _by_mode(reports)[GDSMode.RDMA].applicable

    def test_p2pdma_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/lustre", "lustre")
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_nfs_rdma_check_does_not_fire(self, gds_infrastructure, monkeypatch):
        called = []
        monkeypatch.setattr(
            "checks.gds_report._check_nfs_rdma_mount",
            lambda path: (called.append(path) or
                          CheckResult(check="NFS rdma mount option", mode=GDSMode.RDMA,
                                      status=Status.PASS, why="mocked")),
        )
        build_mode_reports("/mnt/lustre", "lustre")
        assert not called, "_check_nfs_rdma_mount should not fire for Lustre"


# ---------------------------------------------------------------------------
# GPFS / mmfs
# ---------------------------------------------------------------------------

class TestGPFS:
    @pytest.mark.parametrize("fs_type", ["gpfs", "mmfs"])
    def test_native_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/gpfs", fs_type)
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    @pytest.mark.parametrize("fs_type", ["gpfs", "mmfs"])
    def test_rdma_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/gpfs", fs_type)
        assert _by_mode(reports)[GDSMode.RDMA].applicable

    @pytest.mark.parametrize("fs_type", ["gpfs", "mmfs"])
    def test_p2pdma_not_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/gpfs", fs_type)
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    @pytest.mark.parametrize("fs_type", ["gpfs", "mmfs"])
    def test_mmfs_and_gpfs_produce_same_applicability(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/gpfs", fs_type)
        modes = _by_mode(reports)
        assert modes[GDSMode.NATIVE].applicable
        assert not modes[GDSMode.P2PDMA].applicable
        assert modes[GDSMode.RDMA].applicable


# ---------------------------------------------------------------------------
# WekaFS
# ---------------------------------------------------------------------------

class TestWekaFS:
    def test_native_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/weka", "wekafs")
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_rdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/weka", "wekafs")
        assert _by_mode(reports)[GDSMode.RDMA].applicable

    def test_p2pdma_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/weka", "wekafs")
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_write_path_check_fires(self, gds_infrastructure):
        # fs_type == "wekafs" gates a call to
        # cufile_config.check_weka_write_support() (mocked in gds_infrastructure).
        reports = build_mode_reports("/mnt/weka", "wekafs")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "WekaFS write path")
        assert result is not None
        assert result.status == Status.WARN


# ---------------------------------------------------------------------------
# BeeGFS / fhgfs
# ---------------------------------------------------------------------------

class TestBeeGFS:
    @pytest.mark.parametrize("fs_type", ["beegfs", "fhgfs"])
    def test_native_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/beegfs", fs_type)
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    @pytest.mark.parametrize("fs_type", ["beegfs", "fhgfs"])
    def test_p2pdma_not_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/beegfs", fs_type)
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    @pytest.mark.parametrize("fs_type", ["beegfs", "fhgfs"])
    def test_rdma_applicable(self, gds_infrastructure, fs_type):
        # BeeGFS's native path depends on kernel-level RDMA (nvidia-fs has to
        # be loaded to activate it), same as Lustre — not a standalone route.
        reports = build_mode_reports("/mnt/beegfs", fs_type)
        assert _by_mode(reports)[GDSMode.RDMA].applicable

    @pytest.mark.parametrize("fs_type", ["beegfs", "fhgfs"])
    def test_fhgfs_alias_matches_beegfs(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt/beegfs", fs_type)
        modes = _by_mode(reports)
        assert modes[GDSMode.NATIVE].applicable
        assert not modes[GDSMode.P2PDMA].applicable
        assert modes[GDSMode.RDMA].applicable


# ---------------------------------------------------------------------------
# ScaTeFS
# ---------------------------------------------------------------------------

class TestScaTeFS:
    def test_native_applicable(self, gds_infrastructure):
        # nvidia-fs (nvfs) has to be loaded to activate ScaTeFS's kernel RDMA
        # path, same shape as Lustre and BeeGFS — not a standalone route.
        reports = build_mode_reports("/mnt/scatefs", "scatefs")
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_p2pdma_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/scatefs", "scatefs")
        assert not _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_rdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/scatefs", "scatefs")
        assert _by_mode(reports)[GDSMode.RDMA].applicable


# ---------------------------------------------------------------------------
# VirtioFS
# ---------------------------------------------------------------------------

class TestVirtioFS:
    def test_native_not_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/virtiofs", "virtiofs")
        assert not _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_p2pdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/virtiofs", "virtiofs")
        assert _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_pcie_run_all_not_called(self, gds_infrastructure, monkeypatch):
        # virtiofs is not a local block FS — check_acs() is used, not run_all()
        run_all_called = []
        monkeypatch.setattr("checks.pcie.run_all", lambda: run_all_called.append(1) or [])
        build_mode_reports("/mnt/virtiofs", "virtiofs")
        assert not run_all_called, "pcie.run_all() should not be called for virtiofs"

    def test_pcie_check_acs_called(self, gds_infrastructure, monkeypatch):
        check_acs_called = []
        monkeypatch.setattr(
            "checks.pcie.check_acs",
            lambda: (
                check_acs_called.append(1) or
                CheckResult(check="PCIe ACS", mode=GDSMode.P2PDMA, status=Status.PASS, why="mocked")
            ),
        )
        build_mode_reports("/mnt/virtiofs", "virtiofs")
        assert check_acs_called, "pcie.check_acs() should be called for virtiofs"

    def test_fs_key_without_global_key_fails_p2pdma(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.cufile_config.run_all", _real_cufile_run_all)
        # File-only scenario: ignore any CUFILE_* overrides in the caller's environment.
        monkeypatch.setattr("checks.cufile_config._env_override_map", lambda: {})
        monkeypatch.setattr(
            "checks.cufile_config._load_cufile_json",
            lambda: {
                "properties": {"use_pci_p2pdma": False},
                "fs": {"virtiofs": {"use_pci_p2pdma": True}},
            },
        )
        reports = build_mode_reports("/mnt/virtiofs", "virtiofs")
        result = _find(_by_mode(reports)[GDSMode.P2PDMA], "P2PDMA config key")
        assert result is not None
        assert result.status == Status.FAIL
        assert "properties.use_pci_p2pdma = False" in result.why


# ---------------------------------------------------------------------------
# NVMe-oF
# ---------------------------------------------------------------------------

class TestNVMeOF:
    def test_native_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/nvmeof", "nvme-of")
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_p2pdma_applicable(self, gds_infrastructure):
        reports = build_mode_reports("/mnt/nvmeof", "nvme-of")
        assert _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_rdma_applicable(self, gds_infrastructure):
        # nvidia-fs (nvfs) has to be loaded to activate NVMe-oF's kernel RDMA
        # path (MLNX_OFED/DOCA), same shape as Lustre/BeeGFS/NFS/ScaTeFS.
        reports = build_mode_reports("/mnt/nvmeof", "nvme-of")
        assert _by_mode(reports)[GDSMode.RDMA].applicable

    def test_ext4_on_nvmeof_transport_uses_nvmeof_p2pdma_key(self, gds_infrastructure, monkeypatch):
        # NVMe-oF is not detectable as an fs_type from /proc/mounts — the filesystem
        # type is always ext4 or xfs.  The NVMe-oF transport is detected separately
        # via get_nvme_transport(), which triggers p2pdma_block_key="nvmeof" so that
        # block.nvmeof.use_pci_p2pdma is checked rather than block.nvme.use_pci_p2pdma.
        calls = []
        monkeypatch.setattr(
            "checks.cufile_config.run_all",
            lambda fs_type, p2pdma_block_key=None, gdscheck_output=None: calls.append((fs_type, p2pdma_block_key)) or [],
        )
        monkeypatch.setattr("checks.gds_report.get_nvme_transport", lambda path: "rdma")
        build_mode_reports("/mnt/data", "ext4")
        assert any(key == "nvmeof" for _, key in calls), (
            f"Expected p2pdma_block_key='nvmeof' for ext4 on NVMe-oF transport, got: {calls}"
        )


# ---------------------------------------------------------------------------
# RAID0
# ---------------------------------------------------------------------------

class TestRAID0:
    def test_native_applicable(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid0")
        reports = build_mode_reports("/mnt/raid", "ext4")
        assert _by_mode(reports)[GDSMode.NATIVE].applicable

    def test_p2pdma_applicable(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid0")
        reports = build_mode_reports("/mnt/raid", "ext4")
        assert _by_mode(reports)[GDSMode.P2PDMA].applicable

    def test_arch_check_fires_for_raid0(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid0")
        reports = build_mode_reports("/mnt/raid", "ext4")
        p2pdma = _by_mode(reports)[GDSMode.P2PDMA]
        assert _find(p2pdma, "RAID0 P2PDMA architecture") is not None

    def test_old_kernel_arch_check_fails(self, gds_infrastructure, monkeypatch):
        # kernel 6.8 < 7.1, not Grace → RAID0 P2PDMA should fail the arch check
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid0")
        monkeypatch.setattr("checks.iommu.is_grace", lambda: False)
        monkeypatch.setattr("checks.kernel._kernel_version", lambda: (6, 8, 0))
        reports = build_mode_reports("/mnt/raid", "ext4")
        result = _find(_by_mode(reports)[GDSMode.P2PDMA], "RAID0 P2PDMA architecture")
        assert result is not None
        assert result.status == Status.FAIL

    def test_grace_arch_check_passes(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid0")
        monkeypatch.setattr("checks.iommu.is_grace", lambda: True)
        reports = build_mode_reports("/mnt/raid", "ext4")
        result = _find(_by_mode(reports)[GDSMode.P2PDMA], "RAID0 P2PDMA architecture")
        assert result is not None
        assert result.status == Status.PASS

    def test_unsupported_raid_level_fails_native(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: "raid5")
        reports = build_mode_reports("/mnt/raid", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "RAID level GDS support")
        assert result is not None
        assert result.status == Status.FAIL


# ---------------------------------------------------------------------------
# Device-mapper (LVM, dm-crypt, dm-multipath, ...)
# ---------------------------------------------------------------------------

class TestDeviceMapper:
    def test_dm_backing_fails_native(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_dm_info", lambda path: "LVM")
        reports = build_mode_reports("/mnt/lv0", "ext4")
        result = _find(_by_mode(reports)[GDSMode.NATIVE], "Device-mapper backing device")
        assert result is not None
        assert result.status == Status.FAIL

    def test_dm_backing_fails_p2pdma(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_dm_info", lambda path: "dm-crypt")
        reports = build_mode_reports("/mnt/lv0", "ext4")
        result = _find(_by_mode(reports)[GDSMode.P2PDMA], "Device-mapper backing device")
        assert result is not None
        assert result.status == Status.FAIL

    def test_dm_backing_fails_native_mode_overall_status(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_dm_info", lambda path: "device-mapper multipath")
        reports = build_mode_reports("/mnt/lv0", "ext4")
        assert _by_mode(reports)[GDSMode.NATIVE].status == Status.FAIL
        assert _by_mode(reports)[GDSMode.P2PDMA].status == Status.FAIL

    def test_no_dm_backing_check_when_not_present(self, gds_infrastructure, monkeypatch):
        monkeypatch.setattr("checks.gds_report.get_dm_info", lambda path: None)
        reports = build_mode_reports("/mnt/data", "ext4")
        assert _find(_by_mode(reports)[GDSMode.NATIVE], "Device-mapper backing device") is None
        assert _find(_by_mode(reports)[GDSMode.P2PDMA], "Device-mapper backing device") is None


# ---------------------------------------------------------------------------
# Compat-only filesystems
# ---------------------------------------------------------------------------

class TestCompatOnly:
    @pytest.mark.parametrize("fs_type", ["tmpfs", "ramfs", "overlay", "btrfs", "zfs"])
    def test_all_direct_modes_not_applicable(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt", fs_type)
        modes = _by_mode(reports)
        assert not modes[GDSMode.NATIVE].applicable,  f"{fs_type}: native should not be applicable"
        assert not modes[GDSMode.P2PDMA].applicable, f"{fs_type}: p2pdma should not be applicable"
        assert not modes[GDSMode.RDMA].applicable,   f"{fs_type}: rdma should not be applicable"

    @pytest.mark.parametrize("fs_type", ["tmpfs", "ramfs", "overlay", "btrfs", "zfs"])
    def test_compat_mode_always_present(self, gds_infrastructure, fs_type):
        reports = build_mode_reports("/mnt", fs_type)
        assert _by_mode(reports)[GDSMode.COMPAT].applicable
