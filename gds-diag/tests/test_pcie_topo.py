# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks import pcie
from checks.result import CheckResult, GDSMode, Status


SPLIT_TOPO = """\
        GPU0    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
    GPU0         X      NODE    0-47    0               N/A
    NIC0        NODE     X
    Legend:
      X    = Self
      SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
      NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
      PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
      PXB  = Connection traversing multiple PCIe bridges
      PIX  = Connection traversing at most a single PCIe bridge
      NV#  = Connection traversing a bonded set of # NVLinks
    NIC Legend:

      NIC0: mlx5_0

            NVMe0   NVMe1
    GPU0    PHB     PHB
"""


SPLIT_TOPO_WITH_TRAILING_NVME_LEGEND = """\
        GPU0    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
    GPU0         X      NODE    0-47    0               N/A
    NIC0        NODE     X
    Legend:
      X    = Self
      SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
      NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
      PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
      PXB  = Connection traversing multiple PCIe bridges
      PIX  = Connection traversing at most a single PCIe bridge
      NV#  = Connection traversing a bonded set of # NVLinks
    NIC Legend:
      NIC0: mlx5_0
        NVMe0   NVMe1
    GPU0        PHB     PHB
    Legend:
      X    = Self
      SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
      NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
      PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
      PXB  = Connection traversing multiple PCIe bridges
      PIX  = Connection traversing at most a single PCIe bridge
    NVMe Legend:
      NVMe0: nvme0n1
      NVMe1: nvme1n1
"""


SPLIT_TOPO_WITH_ANSI_HEADERS = """\
\x1b[4mGPU0\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m
GPU0\t X \tNODE\t0-47\t0\t\tN/A
NIC0\tNODE\t X \t\t\t\t

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
  PXB  = Connection traversing multiple PCIe bridges
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0

\x1b[4mNVMe0\tNVMe1\t\x1b[0m
GPU0\tPHB\tPHB\t

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
  PXB  = Connection traversing multiple PCIe bridges
  PIX  = Connection traversing at most a single PCIe bridge

NVMe Legend:

  NVMe0: nvme0n1
  NVMe1: nvme1n1
"""


COMBINED_TOPO = """\
        GPU0    GPU1    NVMe0   NVMe1   CPU Affinity    NUMA Affinity
GPU0     X      SYS     PIX     SYS     0-31            0
GPU1    SYS      X      SYS     PIX     32-63           1
NVMe0   PIX     SYS      X      SYS
NVMe1   SYS     PIX     SYS      X
"""


class PcieTopoTests(unittest.TestCase):
    def test_split_nvme_table_after_nic_legend(self):
        parsed = pcie.parse_topo_nvme(SPLIT_TOPO)
        self.assertEqual(parsed["gpu_to_nvme"], {"GPU0": {"NVMe0": "PHB", "NVMe1": "PHB"}})
        self.assertEqual(pcie._target_nvme_labels(parsed, "/dev/nvme0n1"), ["NVMe0"])

    def test_split_nvme_table_with_trailing_nvme_legend(self):
        parsed = pcie.parse_topo_nvme(SPLIT_TOPO_WITH_TRAILING_NVME_LEGEND)
        self.assertEqual(parsed["gpu_to_nvme"], {"GPU0": {"NVMe0": "PHB", "NVMe1": "PHB"}})
        self.assertEqual(parsed["nvme_details"]["NVMe0"]["devices"], ["nvme0n1"])

    def test_split_nvme_table_with_ansi_headers(self):
        parsed = pcie.parse_topo_nvme(SPLIT_TOPO_WITH_ANSI_HEADERS)
        self.assertEqual(parsed["gpu_to_nvme"], {"GPU0": {"NVMe0": "PHB", "NVMe1": "PHB"}})
        self.assertEqual(parsed["nvme_details"]["NVMe1"]["devices"], ["nvme1n1"])

    def test_combined_gpu_nvme_matrix(self):
        parsed = pcie.parse_topo_nvme(COMBINED_TOPO)
        self.assertEqual(parsed["gpu_to_nvme"]["GPU0"]["NVMe0"], "PIX")
        self.assertEqual(parsed["gpu_to_nvme"]["GPU1"]["NVMe1"], "PIX")

    def test_recommendations_report_equal_best_nvmes(self):
        old_run = pcie._run_nvidia_smi_topo_nvme
        try:
            pcie._run_nvidia_smi_topo_nvme = lambda: (SPLIT_TOPO, None)
            recs, err = pcie.topo_nvme_recommendations("/dev/nvme0n1")
        finally:
            pcie._run_nvidia_smi_topo_nvme = old_run

        self.assertIsNone(err)
        self.assertIn("For NVMe0, prefer GPU(s): GPU0 (PHB)", recs)
        self.assertIn("Best NVMe per GPU: GPU0 -> NVMe0, NVMe1 (PHB)", recs)

    def test_cross_root_topology_is_advisory_info_not_failure(self):
        old_gpu = pcie._query_gpu_bdfs
        old_nvme = pcie.get_nvme_bdfs
        old_same = pcie.same_root_complex
        old_topo = pcie.check_topo_nvme
        try:
            pcie._query_gpu_bdfs = lambda: (["0000:65:00.0"], None)
            pcie.get_nvme_bdfs = lambda: ["0000:ca:00.0"]
            pcie.same_root_complex = lambda gpu, nvme: (
                False,
                "No common ancestor",
            )
            pcie.check_topo_nvme = lambda target_nvme=None: CheckResult(
                check="nvidia-smi topo -m -nvme",
                mode=GDSMode.P2PDMA,
                status=Status.WARN,
                why="topo unavailable",
            )

            results = pcie.check_pcie_topology()
        finally:
            pcie._query_gpu_bdfs = old_gpu
            pcie.get_nvme_bdfs = old_nvme
            pcie.same_root_complex = old_same
            pcie.check_topo_nvme = old_topo

        topo = next(result for result in results if result.check == "PCIe topology")
        self.assertEqual(topo.status, Status.INFO)
        self.assertIn("GDS can still operate across root ports", topo.why)
        self.assertIn("1 of 1 visible GPU↔NVMe pair", topo.why)
        self.assertIn("--verbose", topo.why)
        self.assertIsNone(topo.mitigation)
        self.assertNotIn("cannot cross root complex", topo.why.lower())
        self.assertNotIn("GPU 0000:65:00.0 ↔ NVMe 0000:ca:00.0", topo.why)
        self.assertIn("GPU 0000:65:00.0 ↔ NVMe 0000:ca:00.0", topo.evidence)

    def test_cross_root_topology_pair_list_moves_to_evidence(self):
        old_gpu = pcie._query_gpu_bdfs
        old_nvme = pcie.get_nvme_bdfs
        old_same = pcie.same_root_complex
        old_topo = pcie.check_topo_nvme
        try:
            pcie._query_gpu_bdfs = lambda: (["0000:65:00.0", "0000:66:00.0"], None)
            pcie.get_nvme_bdfs = lambda: [
                "0000:ca:00.0",
                "0000:cb:00.0",
                "0000:cc:00.0",
                "0000:cd:00.0",
            ]
            pcie.same_root_complex = lambda gpu, nvme: (
                False,
                f"No common ancestor for {gpu} and {nvme}",
            )
            pcie.check_topo_nvme = lambda target_nvme=None: CheckResult(
                check="nvidia-smi topo -m -nvme",
                mode=GDSMode.P2PDMA,
                status=Status.WARN,
                why="topo unavailable",
            )

            results = pcie.check_pcie_topology()
        finally:
            pcie._query_gpu_bdfs = old_gpu
            pcie.get_nvme_bdfs = old_nvme
            pcie.same_root_complex = old_same
            pcie.check_topo_nvme = old_topo

        topo = next(result for result in results if result.check == "PCIe topology")
        self.assertIn("8 of 8 visible GPU↔NVMe pair", topo.why)
        self.assertIn("--verbose", topo.why)
        self.assertNotIn("0000:cc:00.0", topo.why)
        self.assertNotIn("GPU 0000:66:00.0 ↔ NVMe 0000:cd:00.0", topo.why)
        self.assertIn("GPU 0000:66:00.0 ↔ NVMe 0000:cd:00.0", topo.evidence)


if __name__ == "__main__":
    unittest.main()
