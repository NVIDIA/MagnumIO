# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def gds_infrastructure(monkeypatch):
    """
    Patch all system-level calls made by build_mode_reports() to neutral values,
    leaving only filesystem-specific dispatch logic active.

    Tests that need different behaviour for a specific call can override individual
    patches with additional monkeypatch.setattr() calls — the last setattr wins.
    """
    from checks.result import CheckResult, GDSMode, Status

    monkeypatch.setattr("checks.kernel.run_all", lambda mode: [])
    monkeypatch.setattr("checks.iommu.run_all", lambda mode: [])
    monkeypatch.setattr("checks.iommu.is_grace", lambda: False)
    monkeypatch.setattr("checks.kernel._kernel_version", lambda: (6, 8, 0))

    monkeypatch.setattr("checks.nvidia_fs.run_all", lambda: [])
    monkeypatch.setattr(
        "checks.nvidia_fs.check_open_driver",
        lambda raw, mode: CheckResult(
            check="NVIDIA Open Driver", mode=mode, status=Status.PASS, why="mocked"
        ),
    )
    monkeypatch.setattr("checks.nvidia_fs.check_p2pdma_driver_registries", lambda: None)

    monkeypatch.setattr("checks.pcie.run_all", lambda: [])
    monkeypatch.setattr(
        "checks.pcie.check_acs",
        lambda: CheckResult(
            check="PCIe ACS", mode=GDSMode.P2PDMA, status=Status.PASS, why="mocked"
        ),
    )

    monkeypatch.setattr("checks.gds_report._run_gdscheck_raw", lambda: None)

    monkeypatch.setattr(
        "checks.cufile_config.run_all", lambda fs_type, p2pdma_block_key=None: []
    )
    monkeypatch.setattr("checks.cufile_config.run_compat_checks", lambda: [])
    monkeypatch.setattr(
        "checks.cufile_config.check_weka_write_support",
        lambda config=None: CheckResult(
            check="WekaFS write path", mode=GDSMode.NATIVE, status=Status.WARN, why="mocked"
        ),
    )

    monkeypatch.setattr("checks.rdma.run_all", lambda fs_type: [])
    monkeypatch.setattr(
        "checks.gds_report._check_nfs_rdma_mount",
        lambda path: CheckResult(
            check="NFS rdma mount option", mode=GDSMode.RDMA, status=Status.PASS, why="mocked"
        ),
    )

    monkeypatch.setattr("checks.gds_report.check_odirect", lambda path: (True, "O_DIRECT ok"))
    monkeypatch.setattr("checks.gds_report.check_ext4_data_mode", lambda path: None)
    monkeypatch.setattr("checks.gds_report.get_nvme_transport", lambda path: None)
    monkeypatch.setattr("checks.gds_report.get_raid_level", lambda path: None)
    monkeypatch.setattr("checks.gds_report.is_nvme_backed", lambda path: True)
