# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from checks import fs_matrix, pcie
from subcommands import mount_check


class MountCheckPerformanceRecommendationTests(unittest.TestCase):
    def test_raid_mount_does_not_request_single_nvme_topology_recommendation(self):
        old_backing = fs_matrix.get_backing_device
        old_raid = fs_matrix.get_raid_level
        old_nvme = fs_matrix.is_nvme_backed
        old_topo = pcie.topo_nvme_recommendations
        old_exists = mount_check.os.path.exists
        try:
            fs_matrix.get_backing_device = lambda path: "/dev/md0"
            fs_matrix.get_raid_level = lambda path: "raid0"
            fs_matrix.is_nvme_backed = lambda path: True
            pcie.topo_nvme_recommendations = (
                lambda target_nvme=None: self.fail("RAID mount should not look up md0 in nvidia-smi topo")
            )
            mount_check.os.path.exists = lambda path: False

            recs = mount_check._perf_recommendations("/raid", "ext4")
        finally:
            fs_matrix.get_backing_device = old_backing
            fs_matrix.get_raid_level = old_raid
            fs_matrix.is_nvme_backed = old_nvme
            pcie.topo_nvme_recommendations = old_topo
            mount_check.os.path.exists = old_exists

        text = "\n".join(recs)
        self.assertIn("RAID0 spans multiple NVMe-backed devices", text)
        self.assertIn("GPU locality may be uneven", text)
        self.assertIn("topology in mind", text)
        self.assertNotIn("nvidia-smi topo", text)
        self.assertNotIn("Best NVMe per GPU", text)


if __name__ == "__main__":
    unittest.main()
