# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks import gds_report, rdma
from checks.result import CheckResult, GDSMode, Status


IBV_DEVINFO_STATE_OUTPUT = """\
hca_id:\tmlx5_0
\ttransport:\t\t\tInfiniBand (0)
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
\t\t\tlink_layer:\t\tInfiniBand

hca_id:\tmlx5_1
\ttransport:\t\t\tInfiniBand (0)
\tphys_port_cnt:\t\t\t1
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_DOWN (1)
\t\t\tlink_layer:\t\tInfiniBand
"""


class RDMACheckTests(unittest.TestCase):
    def test_gpfs_dmabuf_driver_line_wins_over_generic_userspace_rdma_unsupported(self):
        output = """\
=====================
DRIVER CONFIGURATION:
=====================
 IBM Spectrum Scale : dmabuf, compat
 Userspace RDMA     : Unsupported
 --DmaBuf support   : Enabled
 --rdma library     : Not Loaded (libcufile_rdma.so)
=====================
"""
        verdict = gds_report._gdscheck_parse(output, "gpfs", nvme_backed=False)
        diagnosis = gds_report._gdscheck_rdma_diagnosis(output, "gpfs")

        self.assertIs(verdict[GDSMode.NATIVE], True)
        self.assertIs(verdict[GDSMode.RDMA], True)
        self.assertEqual(diagnosis, [])

    def test_ibv_devinfo_state_field_is_parsed(self):
        old_run = rdma._run
        try:
            rdma._run = lambda *cmd, **kwargs: (0, IBV_DEVINFO_STATE_OUTPUT, "")
            result = rdma.check_ib_devices()
        finally:
            rdma._run = old_run

        self.assertEqual(result.status.value, "PASS")
        self.assertIn("mlx5_0:PORT_ACTIVE", result.why)

    def test_validate_rdma_client_addresses_rejects_nonlocal_and_invalid(self):
        result = rdma.validate_rdma_client_addresses(
            ["192.0.2.10", "not-an-ip", "10.1.2.3"],
            local_ip_map={"10.1.2.3": "ib0"},
            rdma_netdevs={"ib0"},
        )

        self.assertEqual(result["matched"], ["10.1.2.3"])
        self.assertEqual(result["nonlocal"], ["192.0.2.10"])
        self.assertEqual(result["invalid"], ["not-an-ip"])

    def test_validate_rdma_client_addresses_flags_non_rdma_netdev(self):
        result = rdma.validate_rdma_client_addresses(
            ["10.1.2.3"],
            local_ip_map={"10.1.2.3": "eth0"},
            rdma_netdevs={"ib0"},
        )

        self.assertEqual(result["non_rdma_iface"], ["10.1.2.3 (eth0)"])

    def test_check_rdma_dev_addr_list_uses_client_ips(self):
        from checks import cufile_config

        old_loader = cufile_config._load_cufile_json
        old_local = rdma._local_ipv4_map
        old_rdma_netdevs = rdma._rdma_netdevs
        try:
            cufile_config._load_cufile_json = lambda: {
                "properties": {"rdma_dev_addr_list": ["10.1.2.3"]}
            }
            rdma._local_ipv4_map = lambda: {"10.1.2.3": "ib0"}
            rdma._rdma_netdevs = lambda: {"ib0"}
            result = rdma.check_rdma_dev_addr_list("gpfs")
        finally:
            cufile_config._load_cufile_json = old_loader
            rdma._local_ipv4_map = old_local
            rdma._rdma_netdevs = old_rdma_netdevs

        self.assertEqual(result.status.value, "PASS")
        self.assertIn("client-local", result.why)

    def test_check_rdma_dev_addr_list_uses_gpfs_specific_ips(self):
        from checks import cufile_config

        old_loader = cufile_config._load_cufile_json
        old_local = rdma._local_ipv4_map
        old_rdma_netdevs = rdma._rdma_netdevs
        try:
            cufile_config._load_cufile_json = lambda: {
                "fs": {"gpfs": {"rdma_dev_addr_list": ["10.1.2.3"]}}
            }
            rdma._local_ipv4_map = lambda: {"10.1.2.3": "ib0"}
            rdma._rdma_netdevs = lambda: {"ib0"}
            result = rdma.check_rdma_dev_addr_list("gpfs")
        finally:
            cufile_config._load_cufile_json = old_loader
            rdma._local_ipv4_map = old_local
            rdma._rdma_netdevs = old_rdma_netdevs

        self.assertEqual(result.status.value, "PASS")
        self.assertIn("client-local", result.why)
        self.assertIn("fs.gpfs.rdma_dev_addr_list", result.evidence)

    def test_check_rdma_dev_addr_list_warns_for_server_side_ip(self):
        from checks import cufile_config

        old_loader = cufile_config._load_cufile_json
        old_local = rdma._local_ipv4_map
        old_rdma_netdevs = rdma._rdma_netdevs
        try:
            cufile_config._load_cufile_json = lambda: {
                "properties": {"rdma_dev_addr_list": ["192.0.2.10"]}
            }
            rdma._local_ipv4_map = lambda: {"10.1.2.3": "ib0"}
            rdma._rdma_netdevs = lambda: {"ib0"}
            result = rdma.check_rdma_dev_addr_list("gpfs")
        finally:
            cufile_config._load_cufile_json = old_loader
            rdma._local_ipv4_map = old_local
            rdma._rdma_netdevs = old_rdma_netdevs

        self.assertEqual(result.status.value, "WARN")
        self.assertIn("not local to this client", result.why)

    def test_preinstall_ofed_doca_missing_is_warning(self):
        old_check = rdma.check_ofed
        try:
            rdma.check_ofed = lambda: CheckResult(
                check="MLNX_OFED / DOCA",
                mode=GDSMode.RDMA,
                status=Status.FAIL,
                why="Neither MLNX_OFED nor DOCA found.",
                mitigation="Install RDMA stack.",
            )

            result = rdma.check_ofed_preinstall()
        finally:
            rdma.check_ofed = old_check

        self.assertEqual(result.status, Status.WARN)
        self.assertIn("Pre-install advisory", result.why)
        self.assertIn("DOCA storage installation", result.mitigation)
        self.assertIn("ofed_info -s || doca_version", result.mitigation)


if __name__ == "__main__":
    unittest.main()
