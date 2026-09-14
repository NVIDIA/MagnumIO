# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks import rdma


SINGLE_GPU_SINGLE_NIC = """\
\x1b[4mGPU0\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m
GPU0\t X \tNODE\t0-47\t0\t\tN/A
NIC0\tNODE\t X

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
  PXB  = Connection traversing multiple PCIe bridges
  PIX  = Connection traversing at most a single PCIe bridge

NIC Legend:
  NIC0: mlx5_0
"""


DUAL_GPU_DUAL_NIC = """\
        GPU0    GPU1    NIC0    NIC1    CPU Affinity    NUMA Affinity
GPU0     X      NV1     PIX     SYS     0-31            0
GPU1    NV1      X      SYS     PIX     32-63           1
NIC0    PIX     SYS      X      SYS
NIC1    SYS     PIX     SYS      X

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
  PXB  = Connection traversing multiple PCIe bridges
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:
  NIC0: mlx5_0
  NIC1: mlx5_1
"""


SINGLE_GPU_DUAL_NIC = """\
        GPU0    NIC0    NIC1    CPU Affinity    NUMA Affinity
GPU0     X      NODE    NODE    0-47            0
NIC0    NODE     X      PIX
NIC1    NODE    PIX      X

NIC Legend:
  NIC0: mlx5_0
  NIC1: mlx5_1
"""


class RDMAPolicyRecommendationTests(unittest.TestCase):
    def test_parse_gpu_nic_topology_with_ansi_header(self):
        parsed = rdma.parse_gpu_nic_topology(SINGLE_GPU_SINGLE_NIC)

        self.assertEqual(parsed["gpus"], ["GPU0"])
        self.assertEqual(parsed["nics"], ["NIC0"])
        self.assertEqual(parsed["gpu_to_nic"], {"GPU0": {"NIC0": "NODE"}})
        self.assertEqual(parsed["nic_details"], {"NIC0": "mlx5_0"})

    def test_single_gpu_single_nic_keeps_round_robin(self):
        recs = rdma.rdma_policy_recommendations_for_topology(
            "gpfs",
            SINGLE_GPU_SINGLE_NIC,
            config={"properties": {"rdma_load_balancing_policy": "RoundRobin"}},
        )

        self.assertTrue(any("1 GPU(s), 1 NIC(s)" in rec for rec in recs))
        self.assertTrue(any("RoundRobin" in rec and "matches the recommendation" in rec for rec in recs))
        self.assertTrue(any("Dynamic routing: not needed" in rec for rec in recs))

    def test_dual_gpu_dual_nic_recommends_round_robin_max_min_and_dynamic_routing(self):
        recs = rdma.rdma_policy_recommendations_for_topology(
            "wekafs",
            DUAL_GPU_DUAL_NIC,
            config={"properties": {"rdma_load_balancing_policy": "RoundRobin"}},
        )

        self.assertTrue(any("GPU0 -> NIC0/mlx5_0 (PIX)" in rec for rec in recs))
        self.assertTrue(any("GPU1 -> NIC1/mlx5_1 (PIX)" in rec for rec in recs))
        self.assertTrue(any("RoundRobinMaxMin" in rec and "current: 'RoundRobin'" in rec for rec in recs))
        self.assertTrue(any("properties.rdma_dynamic_routing=true" in rec for rec in recs))

    def test_single_gpu_dual_nic_round_robin_rotates_tied_nearest_nics(self):
        recs = rdma.rdma_policy_recommendations_for_topology(
            "gpfs",
            SINGLE_GPU_DUAL_NIC,
            config={"properties": {"rdma_load_balancing_policy": "RoundRobin"}},
        )

        self.assertTrue(any("GPU0 -> NIC0/mlx5_0, NIC1/mlx5_1 (NODE)" in rec for rec in recs))
        self.assertTrue(any("single GPU with multiple visible NICs" in rec for rec in recs))

    def test_non_gpfs_weka_filesystem_is_ignored(self):
        self.assertEqual(
            rdma.rdma_policy_recommendations_for_topology("lustre", DUAL_GPU_DUAL_NIC),
            [],
        )


if __name__ == "__main__":
    unittest.main()
